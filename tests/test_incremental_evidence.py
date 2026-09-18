"""Regression contracts for retained history, singleton navigation and snapshot IDs."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import llm_wiki_bench
import bench_config as config
import bench_ingest
from build_summaries import build_summaries, current_summaries
from evidence_snapshots import decorate_page, page_units, register_page
from knowledge_updates import fact_catalog, merge_knowledge
from qa_contract import validate_answer
from token_budget import dumps
from wiki_documents import archive_article, parse_document, render_knowledge
from wiki_retriever import WikiRetriever


class IncrementalEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'wiki'
        self.root.mkdir()
        self.path = 'entities/alpha.md'
        self.old_source = self.base / 'old.md'
        self.old_source.write_text('In 2000, Alpha lived in Paris.\n')
        self.new_source = self.base / 'new.md'
        self.new_source.write_text('In 2010, Alpha moved to Berlin.\n')
        self.old_article = archive_article(self.root, self.old_source)
        self.new_article = archive_article(self.root, self.new_source)
        self.old = self.proposal('In 2000, Alpha lived in Paris.', self.old_article)
        self.old['related_pages'] = [{'path': 'entities/beta.md', 'reason': 'Shared biography'}]
        self.old_text = self.render(self.old)
        self.new = self.proposal('In 2010, Alpha moved to Berlin.', self.new_article)
        self.new['aliases'] = ['Later name']
        self.new['tags'] = ['later']
        self.new['facts'][0]['change'] = {
            'relation': 'temporal_update', 'reason': 'A move in 2010 changes the residence from 2000.',
            'related_fact_ids': [fact_catalog(self.old_text)[0]['id']], 'valid_at': '2010'}

    def proposal(self, text, article):
        return {'path': self.path, 'title': 'Alpha', 'description': 'Alpha biography',
                'aliases': ['Earlier name'], 'tags': ['history'],
                'facts': [{'text': text, 'citations': [{'article': article['article']}]}],
                'related_pages': []}

    def render(self, proposal):
        return render_knowledge(self.root, proposal, {self.path, 'entities/beta.md'})[0]

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def test_update_preserves_old_fact_sources_links_and_records_time(self):
        merged = merge_knowledge(self.old_text, self.render(self.new), self.new)
        meta, body = parse_document(merged)
        for value in ('In 2000, Alpha lived in Paris.', 'In 2010, Alpha moved to Berlin.',
                      self.old_article['article'][:-3], self.new_article['article'][:-3],
                      'entities/beta', 'Shared biography'):
            self.assertIn(value, body)
        self.assertEqual(meta['aliases'], ['Earlier name', 'Later name'])
        self.assertEqual(meta['tags'], ['history', 'later'])
        update = meta['knowledge_updates'][0]
        self.assertEqual(update['related_fact_ids'], [fact_catalog(self.old_text)[0]['id']])
        self.assertEqual((update['relation'], update['valid_at']), ('temporal_update', '2010'))
        self.assertIn(update['reason'], body)
        self.assertEqual(merge_knowledge(merged, self.render(self.new), self.new), merged)

    def test_invalid_change_rejected_before_existing_page_write(self):
        self.write(self.path, self.old_text)
        for change in (None, {'relation': 'temporal_update', 'reason': 'Changed', 'related_fact_ids': []},
                       {'relation': 'correction', 'reason': 'Changed', 'related_fact_ids': ['fact_other_page']},
                       {'relation': 'addition', 'reason': '', 'related_fact_ids': []}):
            proposal = copy.deepcopy(self.new)
            proposal['facts'][0]['change'] = change
            with self.subTest(change=change), self.assertRaises(ValueError):
                bench_ingest._validate_proposal(
                    self.root, {'pages': [proposal]}, {'entities'}, {self.path: self.old_text},
                    {self.path: self.old_text}, [self.new_article], [self.old_article])
            self.assertEqual((self.root / self.path).read_text(), self.old_text)

    def test_later_ingestion_keeps_earlier_success_receipt_valid(self):
        with patch.multiple(config, WIKI_DIR=self.root, CACHE_FILE=self.root / '.build-cache.json'), \
                patch.object(config, 'get_page_types', return_value={'entities': {}}):
            initial = copy.deepcopy(self.old)
            initial['related_pages'] = []
            with patch.object(bench_ingest, 'call_llm_json', return_value={'pages': [initial]}):
                self.assertEqual(bench_ingest.ingest_batch([self.old_source])['success'], 1)
            old_receipt = bench_ingest.load_cache()[self.old_article['version']]
            with patch.object(bench_ingest, 'call_llm_json', side_effect=[
                    {'pages_to_view': [self.path]}, {'pages': [self.new]}]):
                self.assertEqual(bench_ingest.ingest_batch([self.new_source])['success'], 1)
            self.assertTrue(bench_ingest._cache_valid(old_receipt, self.root))
            content = (self.root / self.path).read_text()
            self.assertIn(self.old['facts'][0]['text'], content)
            self.assertIn(self.new['facts'][0]['text'], content)

    def test_singletons_cover_all_pages_without_model_and_refresh_after_updates(self):
        self.write(self.path, self.render({**self.old, 'related_pages': []}))
        self.write('entities/solo.md', '# Solo\n\nA rare navigation keyword: zephyr.\n')
        generate = Mock(side_effect=AssertionError('singleton must not call model'))
        stats = build_summaries(self.root, generate=generate, singletons_only=True)
        self.assertEqual((stats['singleton_built'], stats['covered_pages'], stats['uncovered_pages']), (2, 2, []))
        self.assertEqual(build_summaries(self.root, generate=generate)['cached'], 2)
        self.write('entities/solo.md', '# Solo\n\nUpdated zephyr.\n')
        self.assertEqual(len(current_summaries(self.root)), 1)
        self.assertEqual(build_summaries(self.root, generate=generate)['singleton_built'], 1)
        self.assertTrue(any('Updated zephyr.' in text for text in current_summaries(self.root).values()))
        generate.assert_not_called()

    def test_singleton_becoming_linked_is_replaced_in_current_navigation(self):
        self.write(self.path, '# Alpha\n\nAlpha fact.\n')
        self.write('entities/beta.md', '# Beta\n\nBeta fact.\n')
        build_summaries(self.root, singletons_only=True)
        self.write(self.path, self.old_text)
        self.assertEqual(current_summaries(self.root), {})
        stats = build_summaries(self.root, generate=Mock(return_value={
            'title': 'Pair', 'description': 'Pair navigation', 'tags': [], 'summary': 'Alpha and Beta.'}))
        self.assertEqual((stats['built'], stats['covered_pages']), (1, 2))
        self.assertEqual(len(current_summaries(self.root)), 1)

    def test_singleton_read_is_navigation_without_evidence_ids(self):
        self.write(self.path, '# Alpha\n\nAlpha fact.\n')
        build_summaries(self.root, singletons_only=True)
        path = next(iter(current_summaries(self.root)))
        retriever = WikiRetriever(self.root, summary_mode='bm25')
        result = json.loads(retriever.execute_tool('summary_search', {'query': 'Alpha'}))
        self.assertIn(path, [r['path'] for r in result['results']])
        rows = json.loads(retriever.execute_tool('wiki_read', {'paths': [path]}))
        self.assertIn('entities/alpha', rows[0]['text'])
        self.assertNotIn('evidence', rows[0])

    def snapshot(self, text, version='snapshot-1'):
        row = decorate_page({'path': self.path, 'text': text, 'start_offset': 0}, page_units(text), version)
        registry = {}
        register_page(row, registry)
        passages = [{'page': self.path, 'text': text, 'start_offset': 0, 'page_version': version}]
        return row, registry, passages

    def test_noncontiguous_facts_need_only_ids_and_python_copies_quotes(self):
        row, registry, passages = self.snapshot('# Alpha\n\n## Core Facts\n- Alpha founded Beta.\n'
                                                '- An unrelated fact.\n- Beta is in Paris.\n')
        ids = [row['evidence'][i]['evidence_id'] for i in (0, 2)]
        proposal = {'answer': 'Paris', 'evidence_chain': [{'claim': 'The company founded by Alpha is in Paris.',
                                                        'evidence_ids': ids}]}
        answer = validate_answer(proposal, [], passages, registry)
        self.assertEqual([c['quote'] for c in answer['evidence_chain'][0]['citations']],
                         ['Alpha founded Beta.', 'Beta is in Paris.'])
        with self.assertRaisesRegex(ValueError, 'unread or unknown'):
            validate_answer(proposal, [], passages, {})  # New question has no registered reads.

    def test_same_path_new_version_does_not_replace_read_snapshot(self):
        row, registry, passages = self.snapshot('Alpha lived in Paris.')
        newer, newer_registry, _ = self.snapshot('Alpha lives in Berlin.', 'snapshot-2')
        self.write(self.path, 'Alpha lives in Berlin.')
        proposal = {'answer': 'Paris', 'evidence_chain': [{'claim': 'Earlier residence',
                                                        'evidence_ids': [row['evidence'][0]['evidence_id']]}]}
        self.assertEqual(validate_answer(proposal, [], passages, registry)['prediction'], 'Paris')
        proposal['evidence_chain'][0]['evidence_ids'] = [newer['evidence'][0]['evidence_id']]
        with self.assertRaisesRegex(ValueError, 'unread or unknown'):
            validate_answer(proposal, [], passages, registry)
        with self.assertRaisesRegex(ValueError, 'unread|snapshot'):
            validate_answer(proposal, [], passages, newer_registry)

    def test_live_submission_rejects_handwritten_quotes(self):
        row, registry, passages = self.snapshot('Alpha lived in Paris.')
        proposal = {'answer': 'Paris', 'evidence_chain': [{'claim': 'Residence',
                    'evidence_ids': [row['evidence'][0]['evidence_id']],
                    'citations': [{'page': self.path, 'quote': 'Alpha lived in Paris.'}]}]}
        with self.assertRaisesRegex(ValueError, 'not handwritten'):
            validate_answer(proposal, [], passages, registry)

    def test_relationships_and_update_notes_never_become_fact_evidence(self):
        row, _, _ = self.snapshot('# Alpha\n\n## Core Facts\n- Earlier fact. [[sources/articles/abc]]\n'
                                 '## Related Pages\n- [[entities/beta]] — relationship\n'
                                 '## Knowledge Updates\n- A change reason\n')
        self.assertEqual([span['text'] for span in row['evidence']], ['Earlier fact.'])

    def test_knowledge_ids_fit_read_budget_and_unicode_continuation_is_lossless(self):
        body = '# Alpha\n\n## Core Facts\n' + ''.join(
            f'- 事实 {i}：Alpha moved to a new location.\n' for i in range(50))
        self.write(self.path, body)
        retriever = WikiRetriever(self.root, summary_mode='bm25', summary_token_budget=512)
        parts, offset = [], 0
        while True:
            rows = retriever.read([self.path], offset=offset)
            self.assertLessEqual(retriever.tokenizer.count(dumps(rows)), 512)
            row = rows[0]
            parts.append(row['text'])
            for span in row['evidence']:
                self.assertEqual(body[span['start_offset']:span['end_offset']], span['text'])
                self.assertGreaterEqual(span['start_offset'], offset)
                self.assertLessEqual(span['end_offset'], offset + len(row['text']))
            if row['next_offset'] is None:
                break
            self.assertGreater(row['next_offset'], offset)
            offset = row['next_offset']
        self.assertGreater(len(parts), 1)
        self.assertEqual(''.join(parts), body)


if __name__ == '__main__':
    unittest.main()
