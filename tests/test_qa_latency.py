"""Offline latency-path checks, including HotpotQA/2Wiki runner integration."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import llm_wiki_bench
import llm_client
import run_qa
from build_summaries import build_summaries
from benchmark_metrics import latency_percentiles
from embedding_client import EmbeddingClient
from evidence_snapshots import evidence_id
from request_budget import DeadlineExceeded, request_budget
from wiki_agent import WikiAgent
from wiki_documents import archive_article, read_article
from wiki_retriever import WikiRetriever
from token_budget import dumps
from test_qa_loop import call, reply
from test_summary_retrieval import FakeEmbeddings
from test_azure_models import completion


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'wiki'
        self.root.mkdir()
        self.fake = FakeEmbeddings()
        self.facts = {}

    def page(self, name, fact, aliases=()):
        path = f'entities/{name.lower().replace(" ", "-")}.md'
        target = self.root / path
        target.parent.mkdir(exist_ok=True)
        target.write_text('---\naliases: ' + json.dumps(list(aliases)) + '\n---\n# ' + name +
                          '\n\n## Core Facts\n- ' + fact + '\n', encoding='utf-8')
        self.facts[path] = fact
        return path

    def summaries(self):
        with contextlib.redirect_stdout(io.StringIO()):
            build_summaries(self.root, generate=Mock(side_effect=AssertionError('singletons need no model')))

    def retriever(self, **kwargs):
        return WikiRetriever(self.root, embedder=self.fake, **kwargs)

    def finish(self, retriever, paths, answer='Paris'):
        retriever.load()
        requirements, chain = [], []
        for i, path in enumerate(paths):
            page, fact = retriever.pages[path], self.facts[path]
            left = page.body.index(fact)
            eid = evidence_id({'page': path, 'page_version': hashlib.sha256(page.text.encode()).hexdigest(),
                               'start_offset': left, 'end_offset': left + len(fact), 'quote': fact})
            rid = f'hop{i}'
            requirements.append({'id': rid, 'question': fact, 'status': 'supported', 'evidence_ids': [eid]})
            chain.append({'requirement_id': rid, 'claim': fact, 'evidence_ids': [eid]})
        return call('finish_answer', answer=answer, requirements=requirements, evidence_chain=chain)


class SourceBoundaryTest(Fixture):
    def test_eof_clamping_returns_actual_citable_range(self):
        source = self.base / 'article.md'
        source.write_text('Alpha founded Beta.\nBeta is in Paris.\n')
        article = archive_article(self.root, source)['article']
        retriever = self.retriever()
        passage = retriever.source_read(article, 1, 80)
        self.assertEqual((passage['start_line'], passage['end_line'], passage['total_lines']), (1, 2, 2))
        self.assertEqual(passage['quote'], source.read_text().rstrip('\n'))
        self.assertTrue(passage['range_clamped'])
        with self.assertRaises(ValueError):
            read_article(self.root, article, 1, 80)
        for start, end in [(0, 1), (3, 4), (2, 1), (True, 2), (1, True), (1, 201)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                retriever.source_read(article, start, end)
        first = retriever.source_read(article, 1, 1)
        self.assertEqual(first['next_start_line'], 2)


class PreparedIndexTest(Fixture):
    def setUp(self):
        super().setUp()
        self.page('Alpha', 'solar power systems')
        self.page('Beta', 'Ocean currents')
        self.summaries()

    def test_restart_loads_documents_and_only_embeds_query(self):
        first = self.retriever()
        prepared = first.prepare_index()
        self.assertEqual(prepared['status'], 'built')
        self.assertEqual(len(self.fake.calls), 1)
        expected = first.summary_search('sunlight')
        fresh = FakeEmbeddings()
        second = WikiRetriever(self.root, embedder=fresh)
        self.assertEqual(second.prepare_index()['status'], 'loaded')
        actual = second.summary_search('sunlight')
        self.assertEqual(actual['results'], expected['results'])
        self.assertEqual(fresh.calls, [['sunlight']])

    def test_changed_corpus_and_provider_invalidate_snapshot(self):
        first = self.retriever()
        fingerprint = first.prepare_index()['fingerprint']
        self.page('Alpha', 'solar power changed')
        self.summaries()
        changed = self.retriever()
        updated = changed.prepare_index()
        self.assertEqual(updated['status'], 'built')
        self.assertNotEqual(updated['fingerprint'], fingerprint)
        self.fake.model = 'different-provider-model'
        provider = self.retriever().prepare_index()
        self.assertEqual(provider['status'], 'built')
        self.assertNotEqual(provider['fingerprint'], updated['fingerprint'])

    def test_corrupt_binary_or_manifest_is_rebuilt(self):
        report = self.retriever().prepare_index()
        binary = self.root / '.build' / f"summary-index-{report['fingerprint']}.bin"
        binary.write_bytes(b'broken')
        self.assertEqual(self.retriever().prepare_index()['status'], 'built')
        (self.root / '.build/summary-index.json').write_text('[]')
        self.assertEqual(self.retriever().prepare_index()['status'], 'built')
        manifest_path = self.root / '.build/summary-index.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['owners'] = list(reversed(manifest['owners']))
        manifest_path.write_text(json.dumps(manifest))
        self.assertEqual(self.retriever().prepare_index()['status'], 'built')


class NavigationTest(Fixture):
    def test_title_lookup_rejects_ambiguous_aliases_and_nested_matches(self):
        alpha = self.page('Alpha', 'Alpha fact.', ['Shared'])
        self.page('Beta', 'Beta fact.', ['Shared'])
        longer = self.page('Alpha Centauri', 'Longer entity fact.')
        retriever = self.retriever()
        self.assertEqual(retriever.lookup_pages('Shared'), [])
        self.assertEqual([m['path'] for m in retriever.lookup_pages('Alpha Centauri')], [longer])
        self.assertEqual([m['path'] for m in retriever.lookup_pages('Is Alpha here?')], [alpha])
        self.assertEqual(retriever.lookup_pages('Alphabet'), [])

    def test_prefetch_delivers_both_comparison_entities_and_saves_a_turn(self):
        paths = [self.page('Alpha', 'Alpha is French.'), self.page('Beta', 'Beta is French.')]
        retriever = self.retriever()
        model = Mock(return_value=reply(self.finish(retriever, paths, 'yes')))
        result = WikiAgent(retriever, call_llm_with_tools=model, prefetch_pages=2, t_max=2).retrieve(
            'Are Alpha and Beta both French?')
        self.assertEqual(result.answer['prediction'], 'yes')
        self.assertEqual(result.llm_calls, 1)
        self.assertEqual(result.total_calls, 2)
        self.assertEqual({p['page'] for p in result.page_evidence}, set(paths))
        self.assertEqual(result.tool_calls[0]['origin'], 'bootstrap')
        initial = model.call_args.args[0][1]['content']
        for hop in result.answer['evidence_chain']:
            self.assertIn(hop['evidence_ids'][0], initial)

    def test_prefetch_reserves_the_only_slot_for_submission(self):
        self.page('Alpha', 'Alpha is French.')
        model = Mock(return_value=reply(call('finish_answer', answer='unknown', requirements=[], evidence_chain=[])))
        result = WikiAgent(self.retriever(), call_llm_with_tools=model, prefetch_pages=2, t_max=1).retrieve('Alpha?')
        self.assertEqual(result.page_evidence, [])
        self.assertEqual(result.total_calls, 1)

    def test_prefetch_does_not_register_truncated_unseen_text(self):
        path = self.page('Alpha', 'Alpha fact. ' * 3000 + 'The unseen answer is Paris.')
        retriever = self.retriever(read_token_budget=512)
        script = iter([reply(self.finish(retriever, [path])),
                       reply(call('finish_answer', answer='unknown', requirements=[], evidence_chain=[]))])
        result = WikiAgent(retriever, call_llm_with_tools=lambda *a, **k: next(script),
                           prefetch_pages=1, t_max=3).retrieve('Alpha?')
        self.assertIn('unread', result.tool_calls[1]['result'])
        self.assertNotIn('The unseen answer is Paris.', result.page_evidence[0]['text'])

    def test_compound_read_is_charged_two_slots_and_unread_evidence_rejected(self):
        alpha = self.page('Alpha', 'Alpha founded Beta.')
        beta = self.page('Beta', 'Beta is in Paris.')
        retriever = self.retriever()
        script = iter([reply(call('retrieve_evidence', query='Alpha')),
                       reply(self.finish(retriever, [alpha, beta])),
                       reply(call('finish_answer', answer='unknown', requirements=[], evidence_chain=[]))])
        result = WikiAgent(retriever, call_llm_with_tools=lambda *a, **k: next(script),
                           adaptive_retrieval=True, t_max=4).retrieve('Unlisted question?')
        self.assertEqual(result.tool_calls[0]['call_cost'], 2)
        self.assertIn('unread', result.tool_calls[1]['result'])
        self.assertEqual(result.total_calls, 4)
        self.assertEqual({p['page'] for p in result.page_evidence}, {alpha})

    def test_compound_cannot_spend_submission_slot_on_read(self):
        self.page('Alpha', 'Alpha is French.')
        script = iter([reply(call('retrieve_evidence', query='Alpha')),
                       reply(call('finish_answer', answer='unknown', requirements=[], evidence_chain=[]))])
        result = WikiAgent(self.retriever(), call_llm_with_tools=lambda *a, **k: next(script),
                           adaptive_retrieval=True, t_max=2).retrieve('q')
        self.assertEqual(result.total_calls, 2)
        self.assertEqual(result.page_evidence, [])
        self.assertEqual(result.tool_calls[0]['call_cost'], 1)

    def test_failed_compound_read_still_charges_both_operations(self):
        self.page('Alpha', 'Alpha is French.')
        retriever = self.retriever()
        script = iter([reply(call('retrieve_evidence', query='Alpha')),
                       reply(call('finish_answer', answer='unknown', requirements=[], evidence_chain=[]))])
        with patch.object(retriever, 'read', side_effect=OSError('read failed')):
            result = WikiAgent(retriever, call_llm_with_tools=lambda *a, **k: next(script),
                               adaptive_retrieval=True, t_max=3).retrieve('q')
        self.assertEqual(result.tool_calls[0]['call_cost'], 2)
        self.assertEqual(result.total_calls, 3)
        self.assertIn('read failed', result.tool_calls[0]['result'])

    def test_four_hop_bridge_can_retrieve_new_entities_and_submit(self):
        paths = [self.page('Alpha', 'Alpha has parent Beta.'), self.page('Beta', 'Beta has spouse Gamma.'),
                 self.page('Gamma', 'Gamma founded Delta.'), self.page('Delta', 'Delta is based in Paris.')]
        retriever = self.retriever()
        script = iter([reply(call('retrieve_evidence', query=name)) for name in ['Beta', 'Gamma', 'Delta']] +
                      [reply(self.finish(retriever, paths))])
        result = WikiAgent(retriever, call_llm_with_tools=lambda *a, **k: next(script), prefetch_pages=2,
                           adaptive_retrieval=True, t_max=8).retrieve(
            "Where is the company founded by the spouse of Alpha's parent based?")
        self.assertEqual(result.answer['prediction'], 'Paris')
        self.assertEqual(result.total_calls, 8)
        self.assertEqual(result.llm_calls, 4)
        self.assertEqual(len(result.answer['evidence_chain']), 4)

    def test_navigation_and_read_budgets_are_independent(self):
        path = self.page('Alpha', 'solar fact. ' * 500)
        self.summaries()
        retriever = self.retriever(summary_mode='bm25', summary_token_budget=512, read_token_budget=4000)
        navigation = retriever.summary_search('solar')
        read = retriever.read([path])
        self.assertLessEqual(retriever.tokenizer.count(dumps(navigation)), 512)
        self.assertGreater(len(read[0]['text']), len(navigation['results'][0]['text']))
        self.assertLessEqual(retriever.tokenizer.count(dumps(read)), 4000)

    def test_no_progress_expands_budget_and_preserves_history(self):
        for name in ['Alpha', 'Beta', 'Gamma']:
            self.page(name, name + ' shared navigation text.')
        self.summaries()
        retriever = self.retriever(summary_mode='bm25')
        script = iter([reply(call('summary_search', query='shared')),
                       reply(call('summary_search', query='shared')),
                       reply(call('summary_search', query='shared')),
                       reply(call('finish_answer', answer='unknown', requirements=[], evidence_chain=[]))])
        lengths = []
        def model(messages, **kwargs):
            lengths.append(len(messages))
            return next(script)
        result = WikiAgent(retriever, call_llm_with_tools=model, adaptive_retrieval=True).retrieve('shared')
        self.assertEqual(set(result.tool_calls[0]['arguments']['exclude_paths']),
                         {row['path'] for row in result.initial_navigation['results']})
        self.assertGreater(result.tool_calls[2]['arguments']['limit'], result.tool_calls[1]['arguments']['limit'])
        self.assertEqual(lengths, sorted(lengths))
        self.assertGreater(lengths[-1], lengths[0])

    def test_pagination_uses_original_exclusions(self):
        for name in ['Alpha', 'Beta', 'Gamma', 'Delta']:
            self.page(name, name + ' shared navigation text.')
        self.summaries()
        retriever = self.retriever(summary_mode='bm25', summary_limit=1)
        script = iter([reply(call('summary_search', query='shared')),
                       reply(call('summary_search', query='shared', offset=1)),
                       reply(call('finish_answer', answer='unknown', requirements=[], evidence_chain=[]))])
        result = WikiAgent(retriever, call_llm_with_tools=lambda *a, **k: next(script),
                           adaptive_retrieval=True).retrieve('shared')
        first, second = result.tool_calls[:2]
        self.assertEqual(first['arguments']['exclude_paths'], second['arguments']['exclude_paths'])
        paths = [p['path'] for search in result.summary_searches for p in search['results']]
        self.assertEqual(len(paths), len(set(paths)))


class DeadlineTest(Fixture):
    def test_deadline_stops_after_model_and_is_question_local(self):
        now = [0.0]
        def slow_model(*args, **kwargs):
            now[0] = 2.0
            return reply(call('finish_answer', answer='unknown', requirements=[], evidence_chain=[]))
        with patch('request_budget.time.monotonic', side_effect=lambda: now[0]):
            result = WikiAgent(self.retriever(), call_llm_with_tools=slow_model,
                               question_time_budget=1).retrieve('q')
        self.assertEqual(result.stop_reason, 'deadline_exceeded')
        self.assertEqual(result.error['kind'], 'deadline_exceeded')
        fast = WikiAgent(self.retriever(), call_llm_with_tools=Mock(return_value=reply(
            call('finish_answer', answer='unknown', requirements=[], evidence_chain=[])))).retrieve('q')
        self.assertEqual(fast.stop_reason, 'submitted')

    def test_retry_after_cannot_exhaust_budget_or_start_another_attempt(self):
        limited = Mock(status_code=429, text='limited', headers={'Retry-After': '3'})
        with patch.dict('os.environ', {'LLM_MAX_ATTEMPTS': '2'}), \
                patch('llm_client._session.post', return_value=limited) as post, \
                patch('request_budget.time.sleep') as sleep, request_budget(1) as stats:
            with self.assertRaises(DeadlineExceeded):
                llm_client.call_llm('s', 'q')
            self.assertEqual(post.call_count, 1)
            sleep.assert_not_called()
            self.assertEqual(stats['llm_attempts'], 1)
            self.assertLessEqual(post.call_args.kwargs['timeout'], 1)

    def test_embedding_retry_is_bounded_by_same_budget(self):
        client = EmbeddingClient(self.base / 'cache.sqlite')
        limited = Mock(status_code=429, ok=False, headers={'Retry-After': '3'})
        with patch('embedding_client._session.post', return_value=limited) as post, \
                patch('request_budget.time.sleep') as sleep, request_budget(1) as stats:
            with self.assertRaises(DeadlineExceeded):
                client.embed_queries(['q'])
            self.assertEqual(post.call_count, 1)
            sleep.assert_not_called()
            self.assertEqual(stats['embedding_attempts'], 1)

    def test_truncation_reason_is_preserved_without_becoming_an_answer(self):
        with patch('llm_client._session.post', return_value=completion('partial', reason='length')):
            result = WikiAgent(self.retriever(), call_llm_with_tools=llm_client.call_llm_with_tools).retrieve('q')
        self.assertEqual(result.stop_reason, 'model_error')
        self.assertEqual(result.answer['prediction'], 'unknown')
        self.assertEqual(result.error['kind'], 'output_truncated')
        self.assertEqual(result.error['finish_reason'], 'length')
        self.assertEqual(result.network_stats['llm_attempts'], 1)

    def test_http_timeout_has_a_structured_reason_and_count(self):
        with patch.dict('os.environ', {'LLM_MAX_ATTEMPTS': '1'}), \
                patch('llm_client._session.post', side_effect=llm_client.requests.Timeout()):
            result = WikiAgent(self.retriever(), call_llm_with_tools=llm_client.call_llm_with_tools,
                               model='test-model').retrieve('q')
        self.assertEqual(result.error['kind'], 'timeout')
        self.assertEqual(result.usage_by_model['test-model']['calls'], 1)
        self.assertEqual(result.network_stats['llm_attempts'], 1)


class DatasetRunnerTest(Fixture):
    def test_percentiles_include_slow_unknown_predictions(self):
        metrics = latency_percentiles([{'elapsed_seconds': value, 'prediction': 'unknown'} for value in [1, 2, 3, 100]])
        self.assertEqual(metrics['p50_elapsed_seconds'], 2.5)
        self.assertAlmostEqual(metrics['p95_elapsed_seconds'], 85.45)

    def test_hotpot_and_2wiki_profiles_preserve_answer_and_budget(self):
        paths = [self.page('Alpha', 'Alpha is French.'), self.page('Beta', 'Beta is French.')]
        for dataset in ['hotpotqa', '2wikimhqa']:
            qa_dir = self.base / 'data' / dataset
            qa_dir.mkdir(parents=True)
            (qa_dir / 'qa_pairs.jsonl').write_text(json.dumps({'id': 'q1',
                'question': 'Are Alpha and Beta both French?', 'answer': 'yes',
                'supporting_titles': ['Alpha', 'Beta'], 'type': 'comparison'}) + '\n')
            for profile, expected_turns in [('baseline', 2), ('latency', 1)]:
                output = self.base / f'{dataset}-{profile}.jsonl'
                callbacks = []
                retriever = self.retriever()
                finish = self.finish(retriever, paths, 'yes')
                def model(messages, **kwargs):
                    callbacks.append(len(messages))
                    self.assertNotIn('gold_answer', json.dumps(messages))
                    if len(callbacks) == 1 and profile == 'baseline':
                        return reply(call('wiki_read', paths=paths))
                    return reply(finish)
                with self.subTest(dataset=dataset, profile=profile), \
                        patch.object(run_qa.config, 'BASE_DIR', self.base), \
                        patch.object(run_qa.config, 'WIKI_DIR', self.root), \
                        patch.object(run_qa.config, 'set_dataset'), patch.object(run_qa.config, 'ensure_wiki_dirs'), \
                        patch.object(run_qa, 'WikiRetriever', return_value=retriever), \
                        patch.object(run_qa, 'call_llm_with_tools', side_effect=model), \
                        patch('sys.argv', ['run_qa', '--dataset', dataset, '--qa-profile', profile,
                                          '--output', str(output)]), contextlib.redirect_stdout(io.StringIO()):
                    run_qa.main()
                saved = json.loads(output.read_text())
                self.assertEqual(saved['prediction'], 'yes')
                self.assertEqual(saved['retrieval_steps'], 2)
                self.assertEqual(saved['retrieval_llm_calls'], expected_turns)
                self.assertEqual(saved['retrieval_config']['qa_profile'], profile)
                self.assertTrue(saved['timings']['total_seconds'] >= 0)
                self.assertTrue(output.with_suffix('.run.json').is_file())


if __name__ == '__main__':
    unittest.main()
