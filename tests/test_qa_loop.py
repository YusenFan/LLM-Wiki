"""Offline behavioral checks of the unified QA loop; no model API calls."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import llm_wiki_bench
from wiki_agent import WikiAgent, RetrievalResult
from wiki_documents import archive_article, read_article
from wiki_retriever import WikiRetriever
from qa_contract import validate_answer
from evidence_snapshots import evidence_id
import run_qa


def call(name, **args):
    return {'id': name, 'type': 'function', 'function': {
        'name': name, 'arguments': json.dumps(args)}}


def reply(*calls):
    return {'role': 'assistant', 'content': None, 'tool_calls': list(calls)}


class QALoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'wiki'
        self.root.mkdir()
        source = self.base / 'input.md'
        source.write_text('Alpha founded Beta.\nBeta is in Paris.\n')
        self.article = archive_article(self.root, source)['article']
        self.first = read_article(self.root, self.article, 1, 1)
        self.second = read_article(self.root, self.article, 2, 2)
        self.retriever = WikiRetriever(self.root)

    def req(self, rid, citation=None):
        return {'id': rid, 'question': 'Founder?' if rid == 'founder' else 'Location?',
                'status': 'supported' if citation else 'unresolved',
                'evidence_ids': [self.reference_id(citation)] if citation else []}

    def reference_id(self, citation):
        if 'page' in citation:
            body = (self.root / citation['page']).read_text()
            start = body.find(citation['quote'])
            return evidence_id({**citation, 'page_version': hashlib.sha256(body.encode()).hexdigest(),
                                'start_offset': start, 'end_offset': start + len(citation['quote'])})
        return evidence_id({k: citation[k] for k in ('article', 'version', 'start_line', 'end_line', 'quote')})

    def finish(self, requirements=None, both=True):
        citations = [('founder', self.first), ('location', self.second)] if both else [('founder', self.first)]
        return call('finish_answer', answer='Paris', requirements=requirements or [], evidence_chain=[
            {'requirement_id': rid, 'claim': cit['quote'], 'evidence_ids': [self.reference_id(cit)]} for rid, cit in citations])

    def read(self, start, end=None):
        return call('source_read', article=self.article, start_line=start, end_line=end or start)

    def run_script(self, replies, budget=12):
        snapshots = []
        script = iter(replies)
        def model(messages, **kwargs):
            snapshots.append((copy.deepcopy(messages), kwargs))
            return next(script)
        result = WikiAgent(self.retriever, call_llm_with_tools=model, t_max=budget).retrieve('Where is the company Alpha founded?')
        return result, snapshots

    def knowledge_page(self, text='Alpha founded Beta.\nBeta is in Paris.'):
        path = self.root / 'entities/alpha.md'
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
        return 'entities/alpha.md'

    def page_finish(self, page, quote):
        citation = {'page': page, 'quote': quote}
        return call('finish_answer', answer='Paris', requirements=[self.req('location', citation)],
                    evidence_chain=[{'requirement_id': 'location', 'claim': quote,
                                     'evidence_ids': [self.reference_id(citation)]}])

    def test_knowledge_page_answers_without_original_read(self):
        page = self.knowledge_page()
        result, _ = self.run_script([
            reply(call('wiki_read', paths=[page])),
            reply(self.page_finish(page, 'Beta is in Paris.')),
        ], budget=2)
        self.assertEqual(result.answer['prediction'], 'Paris')
        self.assertEqual(result.evidence, [])
        self.assertEqual(len(result.page_evidence), 1)
        actual = result.answer['evidence_chain'][0]['citations'][0]
        self.assertEqual((actual['page'], actual['quote']), (page, 'Beta is in Paris.'))
        self.assertIn('page_version', actual)

    def test_unread_knowledge_page_requires_read_before_submission(self):
        page = self.knowledge_page()
        result, _ = self.run_script([
            reply(self.page_finish(page, 'Beta is in Paris.')),
            reply(call('wiki_read', paths=[page])),
            reply(self.page_finish(page, 'Beta is in Paris.')),
        ])
        self.assertIn('unread', result.tool_calls[0]['result'])
        self.assertEqual(result.answer['prediction'], 'Paris')

    def test_page_evidence_can_be_supplemented_with_original(self):
        page = self.knowledge_page('Alpha founded Beta.')
        first = {'page': page, 'quote': 'Alpha founded Beta.'}
        result, _ = self.run_script([
            reply(call('wiki_read', paths=[page])),
            reply(self.read(2)),
            reply(call('finish_answer', answer='Paris',
                       requirements=[self.req('founder', first), self.req('location', self.second)],
                       evidence_chain=[{'requirement_id': rid, 'claim': cit['quote'], 'evidence_ids': [self.reference_id(cit)]}
                                       for rid, cit in [('founder', first), ('location', self.second)]])),
        ])
        self.assertEqual(result.answer['prediction'], 'Paris')
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(len(result.page_evidence), 1)

    def test_truncated_unread_text_rejected_until_continuation(self):
        body = 'Alpha founded Beta.\n' * 1000 + 'Beta is in Paris.'
        page = self.knowledge_page(body)
        self.retriever.summary_token_budget = 400
        result, _ = self.run_script([
            reply(call('wiki_read', paths=[page])),
            reply(self.page_finish(page, 'Beta is in Paris.')),
            reply(call('wiki_read', paths=[page], offset=len(body) - len('Beta is in Paris.'))),
            reply(self.page_finish(page, 'Beta is in Paris.')),
        ])
        self.assertIn('unread', result.tool_calls[1]['result'])
        self.assertEqual(result.answer['prediction'], 'Paris')
        self.assertEqual(len(result.page_evidence), 2)

    def test_page_quote_cannot_be_paraphrased_or_disguised_as_article(self):
        page = self.knowledge_page()
        result, _ = self.run_script([
            reply(call('wiki_read', paths=[page])),
            reply(self.page_finish(page, 'Beta is located in Paris.')),
            reply(self.page_finish(page, 'Beta is in Paris.')),
        ])
        self.assertIn('unknown evidence_ids', result.tool_calls[1]['result'])
        proposal = copy.deepcopy(result.answer)
        proposal['answer'] = proposal.pop('prediction')
        proposal['evidence_chain'][0]['citations'][0]['article'] = self.article
        with self.assertRaisesRegex(ValueError, 'do not mix'):
            validate_answer(proposal, [], result.page_evidence)

    def test_summary_and_directory_reads_are_not_answer_evidence(self):
        page = self.knowledge_page()
        self.retriever.load()
        summary = 'summaries/example.md'
        self.retriever.pages[summary] = self.retriever._parse_page(
            'example', Path(summary), 'Beta is in Paris.')
        (self.root / 'summaries').mkdir(exist_ok=True)
        (self.root / summary).write_text('Beta is in Paris.')
        result, _ = self.run_script([
            reply(call('wiki_read', paths=[summary, 'entities'])),
            reply(self.page_finish(summary, 'Beta is in Paris.')),
            reply(self.page_finish(page, 'Beta is in Paris.')),
            reply(call('finish_answer', answer='unknown', requirements=[], evidence_chain=[])),
        ])
        self.assertEqual(result.page_evidence, [])
        self.assertIn('unread', result.tool_calls[1]['result'])
        self.assertIn('unread', result.tool_calls[2]['result'])

    def test_gap_rejects_early_answer_then_second_hop_completes(self):
        result, snapshots = self.run_script([
            reply(self.read(1)),
            reply(call('update_evidence_state', requirements=[self.req('founder', self.first), self.req('location')])),
            reply(self.finish(both=False)),
            reply(self.read(2)),
            reply(self.finish([self.req('location', self.second)])),
        ])
        self.assertEqual(result.answer['prediction'], 'Paris')
        self.assertEqual(result.total_calls, 5)
        self.assertEqual(result.stop_reason, 'submitted')
        self.assertIn('Unresolved', result.tool_calls[2]['result'])
        statuses = [json.loads(messages[-1]['content']) for messages, _ in snapshots]
        self.assertEqual([s['remaining_tool_calls'] for s in statuses], [12, 11, 10, 9, 8])
        self.assertEqual(statuses[3]['requirements'][1]['status'], 'unresolved')
        self.assertTrue(all(s['status'] == 'supported' for s in result.requirements))

    def test_unread_citation_rejected_and_corrected_in_same_conversation(self):
        invalid = self.finish([self.req('founder', self.first), self.req('location', self.second)])
        result, _ = self.run_script([reply(self.read(1)), reply(invalid), reply(self.read(2)), reply(invalid)])
        self.assertIn('unread', result.tool_calls[1]['result'])
        self.assertEqual(result.answer['prediction'], 'Paris')

    def test_answer_must_cover_all_recorded_requirements(self):
        result, _ = self.run_script([
            reply(self.read(1, 2)),
            reply(self.finish([self.req('founder', self.first), self.req('location', self.second)], both=False)),
            reply(self.finish()),
        ])
        self.assertIn('cover every', result.tool_calls[1]['result'])
        self.assertEqual(result.answer['prediction'], 'Paris')

    def test_plain_text_does_not_finish_and_last_slot_is_submission_only(self):
        result, snapshots = self.run_script([
            reply(self.read(1, 2)), {'role': 'assistant', 'content': 'Paris'},
            reply(self.finish([self.req('founder', self.first), self.req('location', self.second)])),
        ], budget=2)
        self.assertEqual(result.answer['prediction'], 'Paris')
        self.assertEqual(result.llm_calls, 3)
        self.assertEqual([t['function']['name'] for t in snapshots[-1][1]['tools']], ['finish_answer'])

    def test_budget_enforced_for_oversized_batch_and_every_call_acknowledged(self):
        batch = reply(self.read(1), self.read(2), self.read(1), self.read(2))
        captured = []
        def model(messages, **kwargs):
            captured.append(messages)
            return batch
        result = WikiAgent(self.retriever, call_llm_with_tools=model, t_max=2).retrieve('q')
        self.assertEqual(result.total_calls, 2)
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.answer['prediction'], 'unknown')
        self.assertEqual(result.stop_reason, 'budget_exhausted')
        self.assertEqual(sum(m['role'] == 'tool' for m in captured[0]), 4)

    def test_malformed_submission_recoverable_and_no_evidence_unknown(self):
        bad = call('finish_answer')
        bad['function']['arguments'] = 'not json'
        result, _ = self.run_script([reply(bad), reply(call('finish_answer', answer='unknown', requirements=[self.req('location')], evidence_chain=[]))])
        self.assertIn('error', result.tool_calls[0]['result'])
        self.assertEqual(result.answer['prediction'], 'unknown')
        self.assertEqual(result.requirements[0]['status'], 'unresolved')

    def test_non_tool_responses_have_bounded_retries(self):
        model = Mock(return_value={'role': 'assistant', 'content': 'done'})
        result = WikiAgent(self.retriever, call_llm_with_tools=model, t_max=2).retrieve('q')
        self.assertEqual(model.call_count, 4)
        self.assertEqual(result.stop_reason, 'turn_limit')
        self.assertEqual(result.answer['prediction'], 'unknown')

    def test_tree_navigation_is_available_without_search_tools(self):
        result, snapshots = self.run_script([
            reply(call('wiki_tree', path='sources/articles')),
            reply(self.read(1, 2)),
            reply(self.finish([self.req('founder', self.first), self.req('location', self.second)])),
        ])
        self.assertEqual(result.answer['prediction'], 'Paris')
        self.assertIn('tree sources/articles', result.trace[0])
        names = [t['function']['name'] for t in snapshots[0][1]['tools']]
        self.assertIn('wiki_tree', names)
        self.assertNotIn('wiki_search', names)

    def test_runner_uses_agent_answer_and_persists_state(self):
        self.knowledge_page()
        (self.root / self.article).unlink()  # knowledge-only Wiki is a valid QA input
        qa_dir = self.base / 'data/hotpotqa'
        qa_dir.mkdir(parents=True)
        (qa_dir / 'qa_pairs.jsonl').write_text(json.dumps({'id': 'q1', 'question': 'q', 'answer': 'Paris'}) + '\n')
        output = self.base / 'predictions.jsonl'
        result = RetrievalResult(answer={'prediction': 'Paris', 'evidence_chain': [], 'evidence_status': 'citations_validated'},
                                 requirements=[self.req('location')], stop_reason='submitted',
                                 page_evidence=[{'page': 'entities/alpha.md', 'text': 'Beta is in Paris.', 'start_offset': 0}])
        with patch.object(run_qa.config, 'BASE_DIR', self.base), \
             patch.object(run_qa.config, 'WIKI_DIR', self.root), \
             patch.object(run_qa.config, 'set_dataset'), patch.object(run_qa.config, 'ensure_wiki_dirs'), \
             patch.object(run_qa, 'WikiAgent') as agent, \
             patch('sys.argv', ['run_qa', '--dataset', 'hotpotqa', '--output', str(output)]):
            agent.return_value.retrieve.return_value = result
            run_qa.main()
        saved = json.loads(output.read_text())
        self.assertEqual(saved['prediction'], 'Paris')
        self.assertEqual(saved['evidence_gaps'], result.requirements)
        self.assertEqual(saved['stop_reason'], 'submitted')
        self.assertEqual(saved['knowledge_evidence'], result.page_evidence)
        self.assertEqual(saved['article_evidence'], [])


if __name__ == '__main__':
    unittest.main()
