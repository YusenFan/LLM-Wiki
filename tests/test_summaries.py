import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_wiki_bench.build_agent import ingest_documents
from llm_wiki_bench.build_summaries import build_summaries
from llm_wiki_bench.summary_store import SummaryStore
from llm_wiki_bench.wiki_store import SourceStore, FactStore, digest, fact_state
from llm_wiki_bench.wiki_retriever import WikiRetriever
from llm_wiki_bench.wiki_agent import WikiAgent, RetrievalResult
from llm_wiki_bench.validate_wiki import audit
from llm_wiki_bench import bench_config as config


def message(name, args):
    return {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': name, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sources = SourceStore(self.root)
        self.facts = FactStore(self.root)
        self.store = SummaryStore(self.root)
        self.a = self.add_page('people/lin.md', 'Lin', 'Lin works at Bay University.', 'urn:lin')
        self.b = self.add_page('places/bay.md', 'Bay University', 'Bay University is in Singapore.', 'urn:bay')
        self.children = [{'path': p, 'revision': data['revision']} for p, data in
                         [('people/lin.md', self.a), ('places/bay.md', self.b)]]
        self.claims = [{'statement': "Lin's employer is located in Singapore.", 'kind': 'synthesis',
                        'supports': [{'path': c['path'], 'fact_id': d['fact_ids'][0]} for c, d in zip(self.children, [self.a, self.b])]}]

    def tearDown(self):
        self.tmp.cleanup()

    def add_page(self, path, title, text, identity):
        ref = self.sources.archive(identity, text, title)
        citation = {k: ref[k] for k in ('source_id', 'version_id')}
        citation.update(start=0, end=len(text), quote=text)
        fact = dict(statement=text, event_time='', conditions='', polarity='positive', certainty='asserted', kind='fact', citations=[citation])
        result = self.facts.apply(path, None, identity, [fact], title=title, description='Original description')
        return {**result, 'citation': citation, 'fact': fact}

    def save(self):
        return self.store.apply(children=self.children, title='Lin and Bay University', claims=self.claims)

    def generate(self, messages, **kwargs):
        pages = json.loads(messages[1]['content'])['pages']
        return {**message('write_summary', {'title': 'Lin and Bay University', 'claims': [
            {'statement': "Lin's employer is in Singapore.", 'kind': 'synthesis',
             'supports': [p['selected_facts'][0]['evidence_id'] for p in pages]}]}),
                '_usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}

    def test_progressive_reads_search_layers_and_reverse_navigation(self):
        saved = self.save()
        path = saved['path']
        retriever = WikiRetriever(self.root)
        hits = retriever.search('Singapore', layer='summaries')
        self.assertEqual([h.page.rel_path for h in hits], [path])
        self.assertTrue(all(not h.page.rel_path.startswith('summaries/') for h in retriever.search('Singapore', layer='details')))
        for view in ('overview', 'claims'):
            result = self.store.read(path, view=view)
            self.assertNotIn('quote', json.dumps(result))
            self.assertNotIn('citations', json.dumps(result))
        evidence = self.store.read(path, view='evidence', claim_ids=saved['claim_ids'])
        self.assertEqual(len(evidence['claims'][0]['facts']), 2)
        self.assertEqual(len(evidence['claims'][0]['citations']), 2)
        page = retriever.read(['people/lin.md'])[0]
        self.assertEqual(page['related_summaries'], [path])
        overview = retriever.read([path])[0]
        self.assertNotIn('wiki-summary:start', overview['text'])
        self.assertNotIn('quote', overview['text'])
        self.assertEqual(retriever.tree()['summary_layer']['current'], 1)

    def test_same_source_copies_do_not_count_as_cross_document(self):
        c = self.add_page('other/lin.md', 'Lin copy', self.a['citation']['quote'], 'urn:lin')
        children = [self.children[0], {'path': 'other/lin.md', 'revision': c['revision']}]
        claims = [{'statement': 'Lin works at Bay University.', 'kind': 'direct', 'supports': [
            self.claims[0]['supports'][0], {'path': 'other/lin.md', 'fact_id': c['fact_ids'][0]}]}]
        with self.assertRaisesRegex(ValueError, 'two distinct original'):
            self.store.apply(children=children, title='Copies', claims=claims)

    def test_selected_claim_disclosure_and_multiple_final_references(self):
        claims = [{'statement': data['fact']['statement'], 'kind': 'direct', 'supports': [support]}
                  for data, support in zip([self.a, self.b], self.claims[0]['supports'])]
        saved = self.store.apply(children=self.children, title='Employment and location', claims=claims)
        with self.assertRaisesRegex(ValueError, 'one claim_id'):
            self.store.read(saved['path'], view='evidence')
        result = RetrievalResult(source_ranges={(c['source_id'], c['version_id']): [(c['start'], c['end'])]
                                               for c in [self.a['citation'], self.b['citation']]})
        agent = WikiAgent(WikiRetriever(self.root), call_llm_with_tools=lambda *a, **k: None)
        for cid in saved['claim_ids']:
            evidence = self.store.read(saved['path'], view='evidence', claim_ids=[cid])
            self.assertEqual(len(evidence['claims']), 1)
            self.assertEqual(len(evidence['claims'][0]['facts']), 1)
            agent._observe('summary_read', evidence, result, set())
        agent._finish(dict(answer='Singapore', reasoning='Both facts verified', status='found', evidence_gaps=[],
                           citations=[self.a['citation'], self.b['citation']], summary_refs=[saved]), result)
        self.assertEqual(result.answer, 'Singapore')

    def test_corrupt_summary_is_audited_and_does_not_break_detail_search(self):
        saved = self.save()
        path = self.root/saved['path']
        path.write_text('---\ntitle: broken\n---\n<!-- wiki-summary:start -->\n```json\n{}\n```\n<!-- wiki-summary:end -->\n')
        retriever = WikiRetriever(self.root)
        self.assertTrue(retriever.search('Lin', layer='details'))
        self.assertFalse(retriever.search('broken', layer='summaries'))
        self.assertTrue(any(i['kind'] == 'summary_integrity' for i in audit(self.root)['issues']))

    def test_recursive_or_uncited_summary_is_rejected_without_writes(self):
        with self.assertRaisesRegex(ValueError, 'support'):
            self.store.apply(children=self.children, title='Invalid', claims=[{**self.claims[0], 'supports': []}])
        self.assertFalse((self.root/'summaries').exists())
        saved = self.save()
        children = [{'path': saved['path'], 'revision': saved['revision']}, self.children[1]]
        with self.assertRaisesRegex(ValueError, 'never summaries'):
            self.store.apply(children=children, title='Recursive', claims=self.claims)
        with self.assertRaises(ValueError):
            self.facts.apply(saved['path'], saved['revision'], 'overwrite', [self.a['fact']])

    def test_changed_child_hides_stale_summary_and_rebuilds_with_history(self):
        saved = self.save()
        path = saved['path']
        retriever = WikiRetriever(self.root)
        self.assertTrue(retriever.search('Singapore', layer='summaries'))
        changed = self.facts.apply('people/lin.md', self.a['revision'], 'urn:lin', [
            {**self.a['fact'], 'statement': 'Lin is employed at Bay University.', 'supersedes': self.a['fact_ids']}])
        stale = self.store.read(path)
        self.assertEqual(stale['status'], 'stale')
        self.assertEqual(stale['claims'], [])
        self.assertNotIn('Singapore', stale['text'])
        self.assertFalse(retriever.search('Singapore', layer='summaries'))
        self.assertEqual(audit(self.root)['stale_summaries'], 1)
        rebuilt = build_summaries(self.root, call=self.generate, limit=1)
        self.assertEqual(rebuilt['created'], 1, rebuilt)
        self.assertEqual(self.store.read(path)['status'], 'current')
        self.assertTrue(list((self.root/'.summary-history').rglob('*.md')))
        self.assertNotEqual(self.store.get(path)[1], saved['revision'])

    def test_source_version_change_and_missing_child_invalidate(self):
        saved = self.save()
        self.sources.archive('urn:lin', 'Lin formerly worked at Bay University.', 'Lin')
        self.assertEqual(self.store.read(saved['path'])['status'], 'stale')
        (self.root/'places/bay.md').unlink()
        self.assertIn('dependency unavailable', ' '.join(self.store.read(saved['path'])['stale_reasons']))

    def test_source_tampering_is_not_hidden_by_summary(self):
        saved = self.save()
        c = self.a['citation']
        (self.root/'sources/versions'/f"{c['source_id']}--{c['version_id']}.md").write_text('tampered')
        self.assertEqual(self.store.read(saved['path'])['status'], 'stale')
        self.assertFalse(WikiRetriever(self.root).search('Singapore', layer='summaries'))

    def test_unknown_support_and_generation_race_are_rejected(self):
        bad = [{**self.claims[0], 'supports': [{'path': 'people/lin.md', 'fact_id': 'invented'}]}]
        with self.assertRaisesRegex(ValueError, 'current source-backed'):
            self.store.apply(children=self.children, title='Invented', claims=bad)
        def racing(messages, **kwargs):
            result = self.generate(messages, **kwargs)
            page = self.root/'people/lin.md'
            page.write_text(page.read_text() + '\nChanged during generation\n')
            return result
        report = build_summaries(self.root, call=racing, limit=1)
        self.assertEqual(report['failed'], 1)
        self.assertEqual(report['created'], 0)
        self.assertEqual(build_summaries(self.root, call=self.generate, limit=1)['created'], 1)

    def test_generation_cache_dry_run_and_legacy_eligibility(self):
        dry = build_summaries(self.root, call=lambda *a, **k: self.fail('dry run called model'), dry_run=True)
        self.assertEqual(dry['pending'], 1)
        self.assertFalse((self.root/'summaries').exists())
        first = build_summaries(self.root, call=self.generate, limit=1)
        self.assertEqual(first['created'], 1)
        self.assertEqual(first['usage_by_model'][config.LLM_PREMIUM_MODEL]['total_tokens'], 120)
        second = build_summaries(self.root, call=lambda *a, **k: self.fail('cache called model'))
        self.assertEqual(second['cached'], 1)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'legacy.md').write_text('Old generated prose without original citations.')
            result = build_summaries(root, call=lambda *a, **k: self.fail('legacy called model'))
            self.assertEqual(result['eligible_pages'], 0)
            self.assertIn('Rebuild legacy', result['reason'])

    def test_rejected_groups_are_cached_until_input_changes(self):
        skip = lambda *a, **k: message('skip_summary', {'reason': 'No useful connection'})
        self.assertEqual(build_summaries(self.root, call=skip)['skipped'], 1)
        self.assertEqual(build_summaries(self.root, call=lambda *a, **k: self.fail('repeat skip'))['cached'], 1)
        page = self.root/'people/lin.md'
        page.write_text(page.read_text() + '\nNew page revision\n')
        self.assertEqual(build_summaries(self.root, call=self.generate)['created'], 1)

    def test_summary_fact_view_reads_new_summary_not_old_description(self):
        self.facts.apply('people/lin.md', self.a['revision'], 'urn:lin', [
            {**self.a['fact'], 'kind': 'summary', 'statement': 'Current summary of Lin.'}])
        result = WikiRetriever(self.root).read(['people/lin.md'], view='summary')[0]
        self.assertEqual(result['text'], 'Current summary of Lin.')

    def test_single_agent_expands_then_verifies_both_sources(self):
        saved = self.save()
        summary_ref = {k: saved[k] for k in ('path', 'revision', 'claim_ids')}
        final = dict(answer='Singapore', reasoning='Employer link and location both verified.',
                     citations=[self.a['citation'], self.b['citation']], summary_refs=[summary_ref],
                     evidence_gaps=[], status='found')
        calls = [message('wiki_search', {'query': 'Lin employer', 'layer': 'summaries'}),
                 message('summary_read', {'path': saved['path'], 'view': 'overview'}),
                 message('summary_read', {'path': saved['path'], 'view': 'claims'}),
                 message('summary_read', {'path': saved['path'], 'view': 'evidence', 'claim_ids': saved['claim_ids']}),
                 message('source_read', {k: self.a['citation'][k] for k in ('source_id', 'version_id')}),
                 message('finish_answer', final),
                 message('evidence_note', {'gaps': ['Need the university location original'], 'new_evidence': ['Employment verified']}),
                 message('source_read', {k: self.b['citation'][k] for k in ('source_id', 'version_id')}),
                 message('finish_answer', final)]
        seen = []
        def fake(messages, **kwargs):
            self.assertNotIn('verify_subtask', [t['function']['name'] for t in kwargs['tools']])
            seen.append(kwargs['model'])
            return calls.pop(0)
        result = WikiAgent(WikiRetriever(self.root), call_llm_with_tools=fake, t_max=7).retrieve("Where is Lin's employer?")
        self.assertEqual(result.answer, 'Singapore', result.tool_calls)
        self.assertEqual(result.summary_refs, [summary_ref])
        self.assertEqual(set(seen), {config.LLM_PREMIUM_MODEL})
        self.assertEqual(result.total_calls, 7)
        self.assertIn('error', result.tool_calls[5]['result'])

    def test_summary_references_require_expansion_complete_citations_and_freshness(self):
        saved = self.save()
        ref = {k: saved[k] for k in ('path', 'revision', 'claim_ids')}
        agent = WikiAgent(WikiRetriever(self.root), call_llm_with_tools=lambda *a, **k: None)
        result = RetrievalResult(source_ranges={(c['source_id'], c['version_id']): [(c['start'], c['end'])]
                                               for c in [self.a['citation'], self.b['citation']]})
        final = dict(answer='Singapore', reasoning='Two-hop inference', status='found', evidence_gaps=[],
                     citations=[self.a['citation'], self.b['citation']], summary_refs=[ref])
        with self.assertRaisesRegex(ValueError, 'read the current summary evidence'):
            agent._finish(final, result)
        agent._observe('summary_read', self.store.read(saved['path'], view='evidence'), result, set())
        with self.assertRaisesRegex(ValueError, 'every supporting fact'):
            agent._finish({**final, 'citations': [self.a['citation']]}, result)
        agent._finish(final, result)
        page = self.root/'people/lin.md'
        page.write_text(page.read_text() + '\nA change\n')
        with self.assertRaisesRegex(ValueError, 'stale'):
            agent._finish(final, result)

    def test_build_batch_runs_summary_compilation_and_records_report(self):
        receipts = [{'status': 'complete', 'products': {'people/lin.md': self.a['fact_ids']}},
                    {'status': 'complete', 'products': {'places/bay.md': self.b['fact_ids']}}]
        with patch.object(config, 'WIKI_DIR', self.root), patch('llm_wiki_bench.build_agent.BuildAgent.ingest', side_effect=receipts):
            result = ingest_documents(['input-a.md', 'input-b.md'], call=self.generate, summary_limit=1)
        self.assertEqual(result['summaries']['created'], 1)
        self.assertTrue((self.root/'.build/summary-last-run.json').is_file())


if __name__ == '__main__':
    unittest.main()
