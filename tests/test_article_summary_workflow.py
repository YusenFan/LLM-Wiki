"""Offline contracts for the article → knowledge → summary → QA workflow."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import llm_wiki_bench  # establish the project's sibling-module import path
import bench_config as config
import bench_ingest
from build_summaries import build_summaries, current_summaries, summary_groups, summary_path
from run_qa import validate_answer
from wiki_agent import WikiAgent
from wiki_documents import (
    archive_article, article_reference, citation_link, knowledge_pages, parse_document, read_article,
    related_pages, render_knowledge, validate_citation, wiki_path,
)
from wiki_retriever import WikiRetriever


def graph_page(*targets):
    return '# Page\n\n## Related Pages\n' + '\n'.join(f'- [[{p[:-3]}]] — shared topic' for p in targets)


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / 'wiki'
        self.root.mkdir()
        self.raw = self.base / 'article.md'
        self.raw.write_text('---\ntitle: Alpha\n---\n\n# Alpha\n\nAlpha founded Beta.\nBeta is in Paris.\n', encoding='utf-8')
        self.article = archive_article(self.root, self.raw)
        self.citation = read_article(self.root, self.article['article'], 7, 7)

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

    def proposal(self, path='entities/alpha.md', related=None):
        return {'path': path, 'title': 'Alpha', 'description': 'Alpha and its organization',
                'aliases': ['A'], 'tags': ['history'],
                'facts': [{'text': 'Alpha founded Beta.', 'citations': [self.citation]}],
                'related_pages': related or []}

    def graph(self):
        self.write('entities/a.md', graph_page('entities/b.md', 'entities/c.md'))
        self.write('entities/b.md', graph_page('entities/a.md', 'entities/d.md'))
        self.write('entities/c.md', graph_page('entities/a.md'))
        self.write('entities/d.md', graph_page())
        self.write('entities/solo.md', graph_page())

    def config_patches(self):
        return patch.multiple(config, WIKI_DIR=self.root, CACHE_FILE=self.base / 'cache.json')

    def test_archive_exact_bytes_and_distinct_same_title_versions(self):
        self.assertEqual((self.root / self.article['article']).read_bytes(), self.raw.read_bytes())
        self.raw.write_bytes(self.raw.read_bytes() + 'Unicode: 中文\r\n'.encode())
        newer = archive_article(self.root, self.raw)
        self.assertNotEqual(newer['article'], self.article['article'])
        self.assertEqual((self.root / newer['article']).read_bytes(), self.raw.read_bytes())
        self.assertTrue((self.root / self.article['article']).exists())
        (self.root / newer['article']).write_text('tampered')
        with self.assertRaises(ValueError):
            archive_article(self.root, self.raw)

    def test_ranges_quotes_versions_and_paths_are_checked(self):
        self.assertEqual(validate_citation(self.root, self.citation)['quote'], 'Alpha founded Beta.')
        for change in ({'start_line': 0}, {'start_line': True}, {'end_line': 99},
                       {'quote': 'Invented'}, {'version': 'old'},
                       {'article': 'entities/a.md'}, {'article': '../outside.md'}):
            with self.subTest(change=change), self.assertRaises((ValueError, FileNotFoundError)):
                validate_citation(self.root, {**self.citation, **change})
        for bad in ('/tmp/a.md', '../a.md', 'entities/../../a.md', 'entities\\a.md', 'entities//a.md'):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                wiki_path(self.root, bad)
        (self.root / 'escape').symlink_to(self.base, target_is_directory=True)
        with self.assertRaises(ValueError):
            wiki_path(self.root, 'escape/article.md')

    def test_knowledge_renders_direct_citations_and_zero_relations(self):
        proposal = self.proposal()
        text, sources = render_knowledge(self.root, proposal, {proposal['path']})
        self.assertIn(citation_link({"article": self.article["article"]}), text)
        self.assertNotIn('digests', text)
        self.assertEqual(related_pages(text), {})
        self.assertEqual(sources, {self.article['article']})
        proposal['related_pages'] = [{'path': 'entities/b.md', 'reason': 'founded by Alpha'}]
        text, _ = render_knowledge(self.root, proposal, {'entities/alpha.md', 'entities/b.md'})
        self.assertEqual(related_pages(text), {'entities/b.md': 'founded by Alpha'})
        with self.assertRaises(ValueError):
            render_knowledge(self.root, proposal, {'entities/alpha.md'})
        proposal['facts'][0]['citations'] = []
        with self.assertRaises(ValueError):
            render_knowledge(self.root, proposal, {'entities/alpha.md', 'entities/b.md'})

    def test_exact_groups_subsets_and_overlap(self):
        self.graph()
        self.assertEqual(summary_groups(knowledge_pages(self.root)), [
            ('entities/a.md', 'entities/b.md', 'entities/c.md'),
            ('entities/a.md', 'entities/b.md', 'entities/d.md')])
        pair = {'entities/a.md': graph_page('entities/b.md'), 'entities/b.md': graph_page('entities/a.md')}
        self.assertEqual(summary_groups(pair), [('entities/a.md', 'entities/b.md')])
        self.assertEqual(summary_path(tuple(pair)), summary_path(tuple(reversed(pair))))

    def test_union_coverage_does_not_remove_a_group(self):
        pages = {'x/a.md': graph_page('x/b.md', 'x/c.md'),
                 'x/b.md': graph_page('x/d.md'), 'x/c.md': graph_page('x/d.md'),
                 'x/d.md': graph_page()}
        groups = summary_groups(pages)
        self.assertIn(('x/b.md', 'x/d.md'), groups)
        self.assertIn(('x/c.md', 'x/d.md'), groups)
        self.assertIn(('x/a.md', 'x/b.md', 'x/c.md'), groups)

    def test_only_explicit_explained_knowledge_relations_group(self):
        self.write('entities/a.md', '# A\n[[entities/c]]\n## related pages\n- [[entities/b]] — relation\n'
                   '- [[entities/c]]\n- [[entities/missing]] — absent\n- [[summaries/old]] — summary\n'
                   '## Related Sources\n- [[entities/c]] — not a relation\n')
        self.write('entities/b.md', graph_page())
        self.write('entities/c.md', graph_page())
        self.write('summaries/old.md', graph_page('entities/c.md'))
        self.assertEqual(summary_groups(knowledge_pages(self.root)), [('entities/a.md', 'entities/b.md')])

    def test_summaries_cache_failures_staleness_and_no_recursive_groups(self):
        self.graph()
        response = {'title': 'Shared history', 'description': 'Alpha connections',
                    'tags': ['history, culture'], 'summary': 'An overview of the related pages.'}
        generator = Mock(side_effect=[RuntimeError('temporary failure'), response])
        stats = build_summaries(self.root, generate=generator)
        self.assertEqual((stats['built'], stats['failed']), (1, 1))
        retry = Mock(return_value=response)
        stats = build_summaries(self.root, generate=retry)
        self.assertEqual((stats['built'], stats['cached']), (1, 1))
        self.assertEqual(retry.call_count, 1)
        self.assertEqual(len(current_summaries(self.root)), 2)
        for text in current_summaries(self.root).values():
            meta, body = parse_document(text)
            self.assertEqual(meta['tags'], ['history, culture'])
            for member in meta['members']:
                self.assertIn(f'[[{member[:-3]}]]', body)
        self.write('entities/c.md', graph_page() + '\nChanged knowledge.')
        self.assertEqual(len(current_summaries(self.root)), 1)
        self.assertEqual(len(summary_groups(knowledge_pages(self.root))), 2)
        self.write('entities/a.md', graph_page())
        self.write('entities/b.md', graph_page())
        self.assertEqual(current_summaries(self.root), {})
        self.assertFalse(any(e.get('layer') == 'summaries' for e in WikiRetriever(self.root).tree()['entries']))

    def test_summary_limit_leaves_retryable_pending(self):
        self.graph()
        generate = Mock(return_value={'title': 'Group', 'description': 'Overview', 'tags': [], 'summary': 'Text'})
        stats = build_summaries(self.root, limit=1, generate=generate)
        self.assertEqual((stats['built'], stats['pending']), (1, 1))
        self.assertEqual(generate.call_count, 1)

    def test_ingest_success_cache_and_failed_output_never_cached(self):
        proposal = {'pages': [self.proposal()]}
        with self.config_patches(), patch.object(config, 'get_page_types', return_value={'entities': {}}):
            with patch.object(bench_ingest, 'call_llm_json', return_value=proposal):
                stats = bench_ingest.ingest_batch([self.raw])
            self.assertEqual(stats['success'], 1)
            cache = bench_ingest.load_cache()
            self.assertTrue(bench_ingest._cache_valid(cache[self.article['version']], self.root))
            self.assertFalse(bench_ingest._cache_valid({'status': 'ingested'}, self.root))
            with patch.object(bench_ingest, 'call_llm_json') as generate:
                stats = bench_ingest.ingest_batch([self.raw])
                generate.assert_not_called()
                self.assertEqual(stats['skipped'], 1)
            self.write('entities/alpha.md', (self.root / 'entities/alpha.md').read_text() + '\nAdditional material.\n')
            self.assertTrue(bench_ingest._cache_valid(cache[self.article['version']], self.root))
            (self.root / 'entities/alpha.md').unlink()
            self.assertFalse(bench_ingest._cache_valid(cache[self.article['version']], self.root))
            with patch.object(bench_ingest, 'call_llm_json', return_value={'pages': []}):
                stats = bench_ingest.ingest_batch([self.raw])
                self.assertEqual(stats['failed'], 1)
                self.assertFalse((self.root / 'entities/alpha.md').exists())

    def test_bad_batch_validates_before_any_page_write(self):
        good = self.proposal()
        bad = self.proposal('entities/bad.md')
        bad['facts'][0]['citations'][0] = {'article': 'sources/articles/missing.md'}
        with self.config_patches(), patch.object(config, 'get_page_types', return_value={'entities': {}}):
            with patch.object(bench_ingest, 'call_llm_json', return_value={'pages': [good, bad]}):
                stats = bench_ingest.ingest_batch([self.raw])
            self.assertEqual(stats['failed'], 1)
            self.assertFalse((self.root / 'entities/alpha.md').exists())
            self.assertFalse(config.CACHE_FILE.exists())

    def test_every_input_article_must_be_cited(self):
        second = self.base / 'second.md'
        second.write_text('# Second\n\nUnrelated evidence.\n')
        with self.config_patches(), patch.object(config, 'get_page_types', return_value={'entities': {}}):
            with patch.object(bench_ingest, 'call_llm_json', return_value={'pages': [self.proposal()]}):
                stats = bench_ingest.ingest_batch([self.raw, second])
            self.assertEqual(stats['success'], 1)
            self.assertEqual(stats['failed'], 1)
            self.assertEqual(stats['warnings'][0]['action'], 'retry_individually')
            self.assertIn('uncited input', stats['warnings'][0]['error'])
            self.assertEqual(stats['errors'][0]['articles'], ['second.md'])
            self.assertTrue(bench_ingest._cache_valid(bench_ingest.load_cache()[self.article['version']], self.root))

    def test_coverage_feedback_repairs_complete_batch_before_writes(self):
        second = self.base / 'second.md'
        second.write_text('# Second\n\nSecond is a concept.\n')
        article = archive_article(self.root, second)
        extra = self.proposal('entities/second.md')
        extra['facts'] = [{'text': 'Second is a concept.', 'citations': [{'article': article['article']}]}]
        incomplete = {'pages': [self.proposal()]}
        complete = {'pages': [self.proposal(), extra]}
        def generate(prompt, context, **kwargs):
            self.assertFalse((self.root / 'entities/alpha.md').exists())
            if 'Validation error:' in context:
                self.assertIn(article['article'], context)
                self.assertIn('Second is a concept.', context)
                self.assertIn('Previous proposal (rejected)', context)
                return complete
            return incomplete
        trace = {}
        with self.config_patches(), patch.object(config, 'get_page_types', return_value={'entities': {}}), \
                patch.object(bench_ingest, 'call_llm_json', side_effect=generate) as model:
            result = bench_ingest._ingest_batch_one([self.raw, second], {}, trace)
            self.assertEqual(result['success'], 2)
            self.assertEqual(model.call_count, 2)
            self.assertEqual(len(trace['attempts']), 2)
            self.assertIn('uncited input', trace['attempts'][0]['error'])
            for entry in bench_ingest.load_cache().values():
                self.assertTrue(bench_ingest._cache_valid(entry, self.root))

    def test_split_reloads_shared_page_and_preserves_first_article(self):
        second = self.base / 'second.md'
        second.write_text('# Beta\n\nBeta grew.\n')
        article = archive_article(self.root, second)
        initial = {'pages': [self.proposal()]}
        combined = self.proposal()
        combined['facts'].append({'text': 'Beta grew.', 'citations': [{'article': article['article']}]})
        # Three rejected batch proposals, first singleton, fresh selection, second singleton.
        responses = [initial] * 4 + [{'pages_to_view': ['entities/alpha.md']}, {'pages': [combined]}]
        with self.config_patches(), patch.object(config, 'get_page_types', return_value={'entities': {}}), \
                patch.object(bench_ingest, 'call_llm_json', side_effect=responses) as model:
            stats = bench_ingest.ingest_batch([self.raw, second])
            self.assertEqual((stats['success'], stats['failed']), (2, 0))
            self.assertEqual(stats['errors'], [])
            self.assertEqual(model.call_count, 6)
            self.assertIn('Alpha founded Beta.', model.call_args.args[1])
            for entry in bench_ingest.load_cache().values():
                self.assertTrue(bench_ingest._cache_valid(entry, self.root))

    def test_validation_retries_are_bounded_and_failed_proposals_saved(self):
        with self.config_patches(), patch.object(config, 'get_page_types', return_value={'entities': {}}), \
                patch.object(bench_ingest, 'call_llm_json', return_value={'pages': []}) as model:
            stats = bench_ingest.ingest_batch([self.raw])
            self.assertEqual(model.call_count, 3)
            self.assertEqual(stats['failed'], 1)
            self.assertFalse(config.CACHE_FILE.exists())
            saved = json.loads(Path(stats['errors'][0]['diagnostic']).read_text())
            self.assertEqual(len(saved['attempts']), 3)
            self.assertTrue(all('error' in attempt for attempt in saved['attempts']))

    def test_existing_taxonomy_gains_generic_categories(self):
        self.write('page_types.yaml', 'page_types:\n  music:\n    description: Music\n')
        with self.config_patches(), patch.object(config, '_current_dataset', None):
            config.ensure_wiki_dirs()
            self.assertTrue({'music', 'entities', 'concepts'} <= set(config.get_page_types()))
            self.assertTrue((self.root / 'concepts').is_dir())

    def test_direct_knowledge_article_access_through_tree(self):
        text, _ = render_knowledge(self.root, self.proposal(), {'entities/alpha.md'})
        self.write('entities/alpha.md', text)
        self.write('sources/digests/old.md', '# Alpha\nUnverified old digest')
        self.write('sources/old.md', '# Alpha\nLegacy source summary')
        retriever = WikiRetriever(self.root)
        entries = retriever.tree(depth=3)['entries']
        self.assertTrue(any(e['path'] == 'entities/alpha.md' for e in entries))
        self.assertTrue(any(e.get('layer') == 'articles' for e in entries))
        self.assertFalse(any('digests' in p for p in retriever.pages))
        self.assertNotIn('sources/old.md', retriever.pages)
        self.assertEqual(retriever.read(['entities/alpha'])[0]['type'], 'file')
        self.assertEqual(retriever.read([self.article['article']])[0]['type'], 'article')
        self.assertEqual(retriever.source_read(self.article['article'], 7, 7)['quote'], self.citation['quote'])
        with self.assertRaises(ValueError):
            retriever.source_read('entities/alpha.md', 1, 1)

    def test_agent_gathers_article_evidence_and_respects_budget(self):
        text, _ = render_knowledge(self.root, self.proposal(), {'entities/alpha.md'})
        self.write('entities/alpha.md', text)
        def tool(name, arguments):
            return {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': name, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}
        model = Mock(side_effect=[tool('wiki_read', {'paths': ['entities/alpha.md']}),
                                  tool('source_read', {'article': self.article['article'], 'start_line': 7, 'end_line': 8}),
                                  tool('finish_answer', {'answer': 'unknown', 'evidence_chain': [], 'requirements': []})])
        agent = WikiAgent(WikiRetriever(self.root), call_llm_with_tools=model, t_max=3)
        result = agent.retrieve('Where is the organization founded by Alpha?')
        self.assertEqual(result.total_calls, 3)
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.evidence[0]['quote'], 'Alpha founded Beta.\nBeta is in Paris.')
        model = Mock(return_value=tool('wiki_tree', {'path': '/'}))
        result = WikiAgent(WikiRetriever(self.root), call_llm_with_tools=model, t_max=1).retrieve('Alpha?')
        self.assertEqual(result.total_calls, 1)
        self.assertEqual(result.pages, [])  # no hidden auto-read beyond budget
        self.assertEqual(result.evidence, [])

    def test_source_read_resolves_links_emitted_by_knowledge_pages(self):
        retriever = WikiRetriever(self.root)
        article = self.article['article']
        for link in (article, article[:-3], citation_link({'article': article}),
                     citation_link(self.citation), f' [[{article[:-3]}#L7-L7|Alpha]] '):
            with self.subTest(link=link):
                passage = json.loads(retriever.execute_tool('source_read', {
                    'article': link, 'start_line': 7, 'end_line': 7}))
                self.assertEqual(passage, self.citation)
                self.assertEqual(retriever.read([link])[0]['path'], passage['article'])

    def test_resolved_source_links_still_enforce_article_and_range_contracts(self):
        retriever = WikiRetriever(self.root)
        self.write('entities/alpha.md', '# Alpha\nNavigation only.\n')
        for invalid in ('entities/alpha', '../secret', '/tmp/secret.md',
                        'sources/articles/missing', None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                retriever.source_read(invalid, 1, 1)
        for start, end in ((0, 1), (8, 7), (1, 201), (1, 9), (True, 7)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                retriever.source_read(self.article['article'][:-3], start, end)

    def test_answer_normalizes_read_article_links_without_relaxing_quote_checks(self):
        proposal = {'answer': 'Beta', 'evidence_chain': [
            {'claim': 'Alpha founded Beta', 'citations': [dict(self.citation)]}]}
        for link in (self.article['article'][:-3], citation_link(self.citation)):
            with self.subTest(link=link):
                proposal['evidence_chain'][0]['citations'][0]['article'] = link
                accepted = validate_answer(proposal, [self.citation])
                self.assertEqual(accepted['evidence_chain'][0]['citations'][0], self.citation)
                self.assertEqual(proposal['evidence_chain'][0]['citations'][0]['article'], link)
        proposal['evidence_chain'][0]['citations'][0]['quote'] = 'Alpha founded'
        with self.assertRaisesRegex(ValueError, 'quote mismatch.*complete cited lines'):
            validate_answer(proposal, [self.citation])
        proposal['evidence_chain'][0]['citations'][0] = {**self.citation, 'end_line': 8}
        with self.assertRaisesRegex(ValueError, 'unread line range'):
            validate_answer(proposal, [self.citation])

    def test_bad_source_tool_returns_error_and_agent_can_recover(self):
        messages = []
        from wiki_agent import RetrievalResult
        result = RetrievalResult()
        agent = WikiAgent(WikiRetriever(self.root), call_llm_with_tools=Mock())
        call = {'function': {'name': 'source_read', 'arguments': json.dumps({'article': '../secret.md'})}}
        agent._execute_one(call, messages, result)
        self.assertIn('error', json.loads(messages[-1]['content']))
        self.assertEqual(result.total_calls, 1)
        self.assertEqual(result.evidence, [])

    def test_answer_accepts_exact_read_subranges_and_requires_each_hop(self):
        passage = read_article(self.root, self.article['article'], 7, 8)
        citation2 = read_article(self.root, self.article['article'], 8, 8)
        proposal = {'answer': 'Paris', 'evidence_chain': [
            {'claim': 'Alpha founded Beta', 'citations': [self.citation]},
            {'claim': 'Beta is in Paris', 'citations': [citation2]}]}
        result = validate_answer(proposal, [passage])
        self.assertEqual(result['prediction'], 'Paris')
        self.assertEqual(result['evidence_status'], 'citations_validated')
        for change in ({'quote': 'made up'}, {'version': 'wrong'}, {'start_line': 6},
                       {'article': 'summaries/a.md'}, {'start_line': True}):
            invalid = copy.deepcopy(proposal)
            invalid['evidence_chain'][0]['citations'][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_answer(invalid, [passage])
        with self.assertRaises(ValueError):
            validate_answer(proposal, [self.citation])  # the second hop was not read
        proposal['evidence_chain'][1]['citations'] = []
        with self.assertRaises(ValueError):
            validate_answer(proposal, [passage])

    def test_complete_build_summary_navigation_and_answer(self):
        first = self.proposal(related=[{'path': 'entities/beta.md', 'reason': 'founded by Alpha'}])
        second = self.proposal('entities/beta.md', [{'path': 'entities/alpha.md', 'reason': 'founder'}])
        second['title'] = 'Beta'
        second['facts'] = [{'text': 'Beta is in Paris.', 'citations': [
            read_article(self.root, self.article['article'], 8, 8)]}]
        overview = {'title': 'Alpha and Beta', 'description': 'Founding and location',
                    'tags': ['organizations'], 'summary': 'Alpha founded Beta, an organization in Paris.'}
        with self.config_patches(), patch.object(config, 'get_page_types', return_value={'entities': {}}):
            with patch.object(bench_ingest, 'call_llm_json', return_value={'pages': [first, second]}) as generate, \
                    patch('build_summaries.call_llm_json', return_value=overview):
                stats = bench_ingest.ingest_batch([self.raw])
                self.assertIn('entities/example.md', generate.call_args.args[0])
                self.assertNotIn('concepts/topic.md', generate.call_args.args[0])
                self.assertIn('Allowed page-type directories: ["entities"]', generate.call_args.args[0])
        self.assertEqual(stats['summaries']['built'], 1)
        retriever = WikiRetriever(self.root)
        summary = next(e['path'] for e in retriever.tree()['entries'] if e.get('layer') == 'summaries')
        steps = [('wiki_read', {'paths': [summary]}),
                 ('wiki_read', {'paths': ['entities/alpha.md', 'entities/beta.md']}),
                 ('source_read', {'article': self.article['article'], 'start_line': 7, 'end_line': 8})]
        replies = [{'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': str(i), 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}
                   for i, (name, args) in enumerate(steps)]
        citations = second['facts'][0]['citations']
        proposal = {'answer': 'Paris', 'requirements': [
            {'id': 'location', 'question': 'Where is Beta?', 'status': 'supported', 'citations': citations}],
            'evidence_chain': [{'requirement_id': 'location', 'claim': 'Beta is in Paris', 'citations': citations}]}
        replies.append({'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'finish', 'type': 'function', 'function': {
                'name': 'finish_answer', 'arguments': json.dumps(proposal)}}]})
        retriever.summary_mode = 'bm25'  # This offline integration test must never call embeddings.
        result = WikiAgent(retriever, call_llm_with_tools=Mock(side_effect=replies), t_max=4).retrieve('Where is Beta?')
        self.assertEqual(result.answer['prediction'], 'Paris')
        self.assertEqual(result.stop_reason, 'submitted')
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.total_calls, 4)

    def test_unread_page_cannot_be_overwritten(self):
        before = graph_page()
        self.write('entities/alpha.md', before)
        with self.config_patches(), patch.object(config, 'get_page_types', return_value={'entities': {}}):
            with patch.object(bench_ingest, 'call_llm_json', side_effect=[{'pages_to_view': []}] + [{'pages': [self.proposal()]}] * 3):
                stats = bench_ingest.ingest_batch([self.raw])
        self.assertEqual(stats['failed'], 1)
        self.assertEqual((self.root / 'entities/alpha.md').read_text(), before)

    def test_cold_start_does_not_call_page_selection(self):
        proposal = self.proposal()
        proposal['facts'][0]['citations'] = [{'article': self.article['article']}]
        with self.config_patches(), patch.object(config, 'get_page_types', return_value={'entities': {}}):
            with patch.object(bench_ingest, 'call_llm_json', return_value={'pages': [proposal]}) as llm:
                result = bench_ingest._ingest_batch_one([self.raw], {})
        self.assertEqual(result['success'], 1)
        self.assertEqual(llm.call_count, 1)
        self.assertTrue(llm.call_args.args[0].startswith('Organize the supplied articles'))
        self.assertIn('entities/example.md', llm.call_args.args[0])
        self.assertNotIn('PAGE_TYPE/', llm.call_args.args[0])
        self.assertIn('Page types:', llm.call_args.args[1])

    def test_article_reference_uses_full_original_content_without_quote_copy(self):
        self.raw.write_text('# Woodson\n\nFirst sentence.  Second sentence with 中文.\n', encoding='utf-8')
        article = archive_article(self.root, self.raw)
        reference = article_reference(self.root, {'article': article['article']})
        self.assertEqual(reference['text'], self.raw.read_text())
        proposal = self.proposal()
        proposal['facts'][0]['citations'] = [{'article': article['article'], 'quote': 'First sentence.'}]
        text, _ = render_knowledge(self.root, proposal, {proposal['path']})
        self.assertIn(citation_link({'article': article['article']}), text)
        self.assertNotIn('#L', text)

    def test_later_batch_selects_existing_page_and_reads_article_links(self):
        text, _ = render_knowledge(self.root, self.proposal(), {'entities/alpha.md'})
        self.write('entities/alpha.md', text)
        second = self.base / 'second.md'
        second.write_text('# Beta\n\nBeta grew.\n')
        article = archive_article(self.root, second)
        proposal = self.proposal()
        proposal['facts'].append({'text': 'Beta grew.', 'citations': [{'article': article['article']}]})
        with self.config_patches(), patch.object(config, 'get_page_types', return_value={'entities': {}}):
            with patch.object(bench_ingest, 'call_llm_json', side_effect=[
                {'pages_to_view': ['entities/alpha.md']}, {'pages': [proposal]}]) as llm:
                result = bench_ingest._ingest_batch_one([second], {})
        self.assertEqual(result['success'], 1)
        self.assertEqual(llm.call_count, 2)
        self.assertIn('## Existing knowledge-page catalog', llm.call_args_list[0].args[1])
        self.assertIn('Alpha founded Beta.', llm.call_args_list[1].args[1])

    def test_failed_model_output_is_saved_for_diagnosis(self):
        self.write('entities/alpha.md', graph_page())
        selection = {'pages_to_view': 'entities/nonexistent.md'}
        with self.config_patches(), patch.object(bench_ingest, 'call_llm_json', return_value=selection):
            stats = bench_ingest.ingest_batch([self.raw])
        failure = stats['errors'][0]
        saved = json.loads(Path(failure['diagnostic']).read_text())
        self.assertEqual(saved['stage'], 'select_pages')
        self.assertEqual(saved['selection'], selection)
        self.assertIn('pages_to_view', saved['error'])

    def test_nonexistent_selection_is_reported_without_rejecting_valid_reads(self):
        self.write('entities/alpha.md', graph_page())
        selection = {'pages_to_view': ['entities/alpha.md', 'entities/new.md', 'entities/alpha.md']}
        with self.config_patches(), patch.object(config, 'get_page_types', return_value={'entities': {}}):
            with patch.object(bench_ingest, 'call_llm_json', side_effect=[selection, {'pages': [self.proposal()]}]):
                stats = bench_ingest.ingest_batch([self.raw])
        self.assertEqual(stats['success'], 1)
        self.assertEqual(stats['failed'], 0)
        self.assertEqual(stats['warnings'][0]['ignored_selections'], ['entities/new.md'])
        self.assertFalse((self.root / 'entities/new.md').exists())

    def test_isolated_build_uses_separate_cache(self):
        names = ('WIKI_DIR', 'CACHE_FILE', 'RAW_DIR', 'WIKI_INDEX', 'WIKI_OVERVIEW', 'WIKI_LOG',
                 'INGEST_LOG_DIR', '_current_dataset')
        with patch.multiple(config, **{name: getattr(config, name) for name in names}):
            config.set_dataset('hotpotqa', wiki_dir=self.root)
            self.assertEqual(config.WIKI_DIR, self.root)
            self.assertEqual(config.CACHE_FILE, self.root / '.build-cache.json')
            self.assertEqual(config.RAW_DIR, config.BASE_DIR / 'raw/hotpotqa/articles')


if __name__ == '__main__':
    unittest.main()
