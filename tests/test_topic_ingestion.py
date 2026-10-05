"""Subject directories and compact pages retain the ingestion/evidence contract."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import llm_wiki_bench
import bench_config as config
import bench_ingest
from evidence_snapshots import register_page
from knowledge_updates import fact_catalog, merge_knowledge
from qa_contract import validate_answer
from wiki_documents import archive_article, parse_document, render_knowledge
from wiki_retriever import WikiRetriever


class TopicIngestionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'wiki'
        self.root.mkdir()
        self.raw = self.base / 'alpha.md'
        self.raw.write_text('# Alpha\n\nAlpha founded Beta in 2001.\n', encoding='utf-8')
        self.article = archive_article(self.root, self.raw)
        self.page = {'path': 'history/alpha.md', 'title': 'Alpha', 'description': 'Founder of Beta',
                     'facts': [{'text': 'Alpha founded Beta in 2001.',
                                'citations': [{'article': self.article['article']}]}]}
        self.patches = patch.multiple(config, WIKI_DIR=self.root, CACHE_FILE=self.root / '.cache.json',
                                      _current_dataset=None)
        self.patches.start()
        self.addCleanup(self.patches.stop)
        config.save_page_types({'history': {'description': 'Historical subjects'}})

    def validate(self, proposal):
        return bench_ingest._validate_proposal(self.root, proposal, {'history'}, {}, {}, [self.article], [])

    def test_minimal_schema_and_direct_qa_evidence(self):
        text, _ = render_knowledge(self.root, self.page, {self.page['path']})
        meta, body = parse_document(text)
        self.assertEqual(meta['type'], 'knowledge')
        self.assertNotIn('tags', meta)
        self.assertNotIn('aliases', meta)
        self.assertIn('## Facts', body)
        self.assertNotIn('## Related Sources', body)
        self.assertNotIn('## Related Pages', body)
        target = self.root / self.page['path']
        target.parent.mkdir()
        target.write_text(text, encoding='utf-8')
        retriever = WikiRetriever(self.root, summary_mode='tree')
        row = retriever.read([self.page['path']])[0]
        fact = next(span for span in row['evidence'] if span['text'] == self.page['facts'][0]['text'])
        registry = {}
        register_page(row, registry)
        result = validate_answer({'answer': '2001', 'evidence_chain': [
            {'claim': 'The founding year was 2001.', 'evidence_ids': [fact['evidence_id']]}]},
            [], [{'page': row['path'], 'text': row['text'], 'start_offset': row['start_offset'],
                  'page_version': row['page_version']}], registry)
        self.assertEqual(result['prediction'], '2001')

    def test_new_directory_registered_after_validation_and_reused_next_batch(self):
        first = copy.deepcopy(self.page)
        first['path'] = 'world-history/alpha.md'
        proposal = {'directories': {'world-history': 'World history'}, 'pages': [first]}
        with patch.object(bench_ingest, 'call_llm_json', return_value=proposal):
            stats = bench_ingest.ingest_batch([self.raw])
        self.assertEqual(stats['success'], 1)
        self.assertEqual(set(config.get_page_types()), {'history', 'world-history'})
        self.assertIn('[[world-history/_index]]', (self.root / 'index.md').read_text())
        self.assertIn('World history', config.get_page_types()['world-history']['description'])
        self.assertFalse((self.root / 'entities').exists())
        self.assertFalse((self.root / 'concepts').exists())
        second_raw = self.base / 'beta.md'
        second_raw.write_text('# Beta\n\nBeta is in Paris.\n')
        article = archive_article(self.root, second_raw)
        second = {'path': 'world-history/beta.md', 'title': 'Beta', 'description': 'Organization',
                  'facts': [{'text': 'Beta is in Paris.', 'citations': [{'article': article['article']}]}],
                  'related_pages': [{'path': first['path'], 'reason': 'Founded by Alpha'}]}
        with patch.object(bench_ingest, 'call_llm_json', side_effect=[
                {'pages_to_view': []}, {'pages': [second]}]) as llm:
            stats = bench_ingest.ingest_batch([second_raw])
        self.assertEqual(stats['success'], 1)
        self.assertIn('"world-history": "World history"', llm.call_args.args[1])
        self.assertNotIn('Existing fact IDs', llm.call_args.args[1])
        self.assertTrue((self.root / second['path']).exists())
        retriever = WikiRetriever(self.root, summary_mode='tree')
        self.assertIn(second['path'], [entry['path'] for entry in retriever.tree('world-history')['entries']])

    def test_bad_directory_or_fact_does_not_mutate_catalog_or_write_page(self):
        original_catalog = (self.root / 'page_types.yaml').read_bytes()
        for name in ('sources', 'summaries', 'syntheses', '../escape', 'two/levels', '_hidden', 'UpperCase'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate({'directories': {name: 'Invalid directory'}, 'pages': [self.page]})
        for additions in (None, [], {'science': ''}, {'history': 'Overwritten description'},
                          {'unused': 'An unused subject'}):
            with self.subTest(additions=additions), self.assertRaises(ValueError):
                self.validate({'directories': additions, 'pages': [self.page]})
        undeclared = {**self.page, 'path': 'science/alpha.md'}
        with self.assertRaises(ValueError):
            self.validate({'pages': [undeclared]})
        invalid = copy.deepcopy(undeclared)
        invalid['facts'][0]['citations'] = []
        with patch.object(bench_ingest, 'call_llm_json', return_value={
                'directories': {'science': 'Science'}, 'pages': [invalid]}):
            stats = bench_ingest.ingest_batch([self.raw])
        self.assertEqual(stats['failed'], 1)
        self.assertEqual((self.root / 'page_types.yaml').read_bytes(), original_catalog)
        self.assertFalse((self.root / 'science').exists())
        self.assertFalse(config.CACHE_FILE.exists())

    def test_directory_validation_feedback_has_no_premature_side_effects(self):
        page = {**self.page, 'path': 'science/alpha.md'}
        responses = iter([{'pages': [page]}, {'directories': {'science': 'Science'}, 'pages': [page]}])
        def generate(prompt, context, **kwargs):
            self.assertEqual(set(config.get_page_types()), {'history'})
            self.assertFalse((self.root / 'science').exists())
            return next(responses)
        with patch.object(bench_ingest, 'call_llm_json', side_effect=generate) as llm:
            stats = bench_ingest.ingest_batch([self.raw])
        self.assertEqual(stats['success'], 1)
        self.assertEqual(llm.call_count, 2)
        self.assertIn('not a knowledge-page path', llm.call_args.args[1])
        self.assertIn('science', config.get_page_types())

    def test_cold_start_has_no_purpose_or_update_contract(self):
        with patch.object(config, 'get_purpose_file') as purpose, \
                patch.object(bench_ingest, 'call_llm_json', return_value={'pages': [self.page]}) as llm:
            bench_ingest._ingest_batch_one([self.raw], {})
        purpose.assert_not_called()
        llm.assert_called_once()
        prompt, context = llm.call_args.args
        self.assertNotIn('"change"', prompt)
        self.assertNotIn('Purpose:', context)
        self.assertNotIn('Existing fact IDs', context)

    def test_catalog_aliases_update_contract_and_no_duplicate_evidence(self):
        previous, _ = render_knowledge(self.root, {**self.page, 'aliases': ['A'], 'tags': ['history']},
                                       {self.page['path']})
        path = self.root / self.page['path']
        path.parent.mkdir()
        path.write_text(previous)
        trace = {}
        with patch.object(bench_ingest, 'call_llm_json', side_effect=[
                {'pages_to_view': [self.page['path']]}, {'pages': [self.page]}]) as llm:
            bench_ingest._ingest_batch_one([self.raw], {}, trace)
        catalog = json.loads(bench_ingest._page_catalog({self.page['path']: previous}))
        self.assertEqual(catalog['aliases'], ['A'])
        self.assertEqual(catalog['title'], 'Alpha')
        self.assertNotIn('tags', catalog)
        prompt, context = llm.call_args.args
        self.assertIn('"change"', prompt)
        self.assertIn(fact_catalog(previous)[0]['id'], context)
        # This same source is already supplied as an input article; do not repeat its body.
        self.assertEqual(context.count('### Alpha\nPath:'), 1)
        self.assertEqual(parse_document(path.read_text())[0]['tags'], ['history'])

    def test_legacy_core_facts_update_preserves_ids_sources_and_history(self):
        previous, _ = render_knowledge(self.root, self.page, {self.page['path']})
        previous = previous.replace('type: knowledge', 'type: entities').replace('## Facts', '## Core Facts')
        old_fact = fact_catalog(previous)[0]
        new_raw = self.base / 'later.md'
        new_raw.write_text('# Later\n\nBeta moved to Paris in 2010.\n')
        article = archive_article(self.root, new_raw)
        change = {'relation': 'addition', 'reason': 'Adds the later location.',
                  'related_fact_ids': [], 'valid_at': '2010'}
        update = {**self.page, 'facts': [{'text': 'Beta moved to Paris in 2010.',
                  'citations': [{'article': article['article']}], 'change': change}]}
        rendered, _ = render_knowledge(self.root, update, {self.page['path']})
        merged = merge_knowledge(previous, rendered, update)
        meta, body = parse_document(merged)
        self.assertEqual(meta['type'], 'knowledge')
        self.assertNotIn('Core Facts', body)
        self.assertIn('## Facts', body)
        self.assertIn(old_fact, fact_catalog(merged))
        self.assertIn(self.article['article'][:-3], body)
        self.assertIn(article['article'][:-3], body)
        self.assertEqual(meta['knowledge_updates'][0]['valid_at'], '2010')
        self.assertEqual(merge_knowledge(merged, rendered, update), merged)

    def test_initialization_preserves_subject_catalog_and_skips_purpose_generation(self):
        with patch.object(config, '_current_dataset', 'hotpotqa'), \
                patch.object(config, 'auto_init_purpose') as purpose, \
                patch.object(config, 'auto_init_page_types') as initialize:
            config.ensure_wiki_dirs()
        purpose.assert_not_called()
        initialize.assert_not_called()
        self.assertEqual(set(config.get_page_types()), {'history'})
        self.assertTrue((self.root / 'history').is_dir())

    def test_directory_initialization_accepts_subject_slugs_and_falls_back_without_types(self):
        with patch.object(config, '_sample_articles_for_init', return_value=[{'title': 'A', 'excerpt': 'Text'}] * 5), \
                patch('llm_client.call_llm_json', return_value={'page_types': {
                    'world-history': {'description': 'World history'}, 'sources': {'description': 'Reserved'},
                    '../bad': {'description': 'Escape'}}}) as llm:
            config.auto_init_page_types()
        self.assertEqual(set(config.get_page_types()), {'world-history'})
        self.assertNotIn('5-8', llm.call_args.kwargs['system_prompt'])
        with patch.object(config, '_sample_articles_for_init', return_value=[]):
            config.auto_init_page_types()
        self.assertEqual(set(config.get_page_types()), {'topics'})


if __name__ == '__main__':
    unittest.main()
