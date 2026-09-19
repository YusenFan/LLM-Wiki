"""Recall, context bounds, resumability, cache isolation and QA integration, without network."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import llm_wiki_bench
from build_summaries import build_summaries
from embedding_client import EmbeddingClient
from summary_retrieval import BM25, fuse
from token_budget import TokenBudget, dumps
from wiki_agent import WikiAgent
from wiki_retriever import WikiRetriever


class FakeEmbeddings:
    model = 'test-embedding'

    def __init__(self):
        self.calls = []
        self.stats = {'requests': 0, 'input_tokens': 0, 'cache_hits': 0, 'embedded_texts': 0}

    def embed(self, texts):
        self.calls.append(texts)
        self.stats['requests'] += 1
        self.stats['embedded_texts'] += len(texts)
        # Test semantic recall when literal query terms do not occur in a summary.
        return [[1., 0.] if 'solar' in text or 'sunlight' in text or 'needle' in text else [0., 1.]
                for text in texts]


class SummaryRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fake = FakeEmbeddings()
        self.tokenizer = TokenBudget()

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def summaries(self, bodies):
        for i in range(len(bodies)):
            self.write(f'knowledge/a{i}.md', f'# Topic {i}\n\n## Related Pages\n- [[knowledge/b{i}]] — related\n')
            self.write(f'knowledge/b{i}.md', f'# Partner {i}\n\n## Related Pages\n- [[knowledge/a{i}]] — related\n')
        proposals = [{'title': f'Topic {i} overview', 'description': 'An overview', 'tags': [], 'summary': body}
                     for i, body in enumerate(bodies)]
        build_summaries(self.root, generate=Mock(side_effect=proposals))

    def retriever(self, **kwargs):
        return WikiRetriever(self.root, embedder=self.fake, **kwargs)

    def test_bm25_uses_words_not_substrings_and_rrf_ignores_score_scales(self):
        ranking = BM25(['Redwood forests', 'Ed Wood American filmmaker', '中国电影']).rank('Ed', 10)
        self.assertEqual([i for i, _ in ranking], [1])
        self.assertEqual(BM25(['美国', '中国']).rank('中国', 2)[0][0], 1)
        self.assertEqual(fuse([(0, 1000), (1, 10)], [(1, .9), (2, .8)])[0][0], 1)
        self.assertEqual(fuse([(0, 1000), (1, 10)], [(1, .9), (2, .8)]),
                         fuse([(0, .001), (1, .0001)], [(1, 900), (2, 800)]))

    def test_hybrid_finds_semantic_candidate_and_indexes_only_current_text(self):
        self.summaries(['solar power systems', 'Ocean currents'])
        self.write('summaries/stale.md', '# sunlight stale')
        retriever = self.retriever()
        found = retriever.summary_search('sunlight', limit=1)
        self.assertEqual(found['effective_mode'], 'hybrid')
        self.assertEqual(found['results'][0]['path'], 'summaries/topic-0-overview.md')
        indexed = '\n'.join(self.fake.calls[0])
        self.assertNotIn('fingerprint', indexed)
        self.assertNotIn('knowledge/a0', indexed)
        self.assertNotIn('sunlight stale', indexed)
        self.assertIn('Partner 0', indexed)
        self.assertEqual(self.retriever(summary_mode='bm25').summary_search('sunlight')['results'], [])
        self.write('knowledge/a0.md', '# Changed source\n\n## Related Pages\n- [[knowledge/b0]] — related\n')
        current = self.retriever().summary_search('sunlight')
        self.assertEqual(current['summary_count'], 1)
        self.assertNotIn('summaries/topic-0-overview.md', [row['path'] for row in current['results']])

    def test_candidate_paging_exclusion_and_requery_reach_beyond_first_batch(self):
        self.summaries([f'shared history subject{i}' for i in range(8)])
        retriever = self.retriever(summary_mode='bm25')
        first = retriever.summary_search('shared', limit=3)
        second = retriever.summary_search('shared', limit=3, offset=first['next_offset'])
        paths = {row['path'] for row in first['results']}
        self.assertFalse(paths & {row['path'] for row in second['results']})
        excluded = retriever.summary_search('shared', exclude_paths=list(paths))
        self.assertFalse(paths & {row['path'] for row in excluded['results']})
        specific = retriever.summary_search('subject7', limit=1)
        self.assertIn('subject7', specific['results'][0]['text'])
        self.assertEqual(specific['candidate_count'], 1)

    def test_summary_and_page_payloads_stay_bounded_and_unicode_paging_is_lossless(self):
        self.summaries(['开头文章。' * 700 + '\n\nsolar power target\n\n' + '结尾很长。' * 700] * 6)
        retriever = self.retriever(summary_mode='bm25', summary_token_budget=1000)
        found = retriever.summary_search('solar', limit=5)
        self.assertLessEqual(self.tokenizer.count(dumps(found)), 1000)
        self.assertTrue(found['results'])
        self.assertIsNotNone(found['next_offset'])
        self.assertTrue(any(row.get('truncated') for row in found['results']))
        self.assertIn('solar', found['results'][0]['text'])
        path = found['results'][0]['path']
        parts, offset = [], 0
        while True:
            rows = retriever.read([path], offset=offset)
            self.assertLessEqual(self.tokenizer.count(dumps(rows)), 1000)
            parts.append(rows[0]['text'])
            if rows[0]['next_offset'] is None:
                break
            self.assertGreater(rows[0]['next_offset'], offset)
            offset = rows[0]['next_offset']
        self.assertEqual(''.join(parts), retriever.pages[path].body)
        batch = retriever.read([row['path'] for row in found['results']])
        self.assertLessEqual(self.tokenizer.count(dumps(batch)), 1000)

    def test_exclusions_refill_candidates_instead_of_hiding_the_remaining_corpus(self):
        self.summaries(['shared history'] * 4)
        retriever = self.retriever(summary_mode='hybrid', summary_limit=2, summary_candidates=2)
        first = retriever.summary_search('shared')
        first_paths = [row['path'] for row in first['results']]
        next_batch = retriever.summary_search('shared', exclude_paths=first_paths)
        self.assertEqual(len(next_batch['results']), 2)
        self.assertFalse(set(first_paths) & {row['path'] for row in next_batch['results']})

    def test_dense_chunking_keeps_tail_content_with_bounded_embedding_inputs(self):
        self.summaries(['ordinary words ' * 8000 + 'needle'])
        found = self.retriever(summary_mode='dense').summary_search('needle')
        self.assertEqual(len(found['results']), 1)
        chunks = self.fake.calls[0]
        self.assertGreater(len(chunks), 1)
        self.assertIn('needle', chunks[-1])
        tokenizer = TokenBudget('text-embedding-3-large')
        self.assertTrue(all(tokenizer.count(chunk) <= 6000 for chunk in chunks))

    def test_embedding_failure_is_explicit_and_bm25_remains_usable(self):
        self.summaries(['solar research'])
        self.fake.embed = Mock(side_effect=RuntimeError('service unavailable'))
        retriever = self.retriever()
        result = retriever.summary_search('solar')
        self.assertEqual(result['effective_mode'], 'bm25')
        self.assertTrue(result['warnings'])
        self.assertTrue(result['results'])
        retriever.summary_search('research')
        self.assertEqual(self.fake.embed.call_count, 1)
        dense = self.retriever(summary_mode='dense').summary_search('solar')
        self.assertEqual(dense['effective_mode'], 'unavailable')
        self.assertEqual(dense['results'], [])

    def test_agent_gets_filtered_summaries_and_can_requery_in_same_budget(self):
        self.summaries(['solar energy', 'Ocean currents'])
        self.write('isolated/secret-path.md', '# Isolated page\n')
        retriever = self.retriever(summary_limit=1)
        snapshots = []
        calls = [('summary_search', {'query': 'Ocean currents'}),
                 ('finish_answer', {'answer': 'unknown', 'requirements': [], 'evidence_chain': []})]

        def model(messages, **kwargs):
            snapshots.append(json.loads(json.dumps(messages)))
            name, arguments = calls.pop(0)
            return {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': name, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}

        result = WikiAgent(retriever, call_llm_with_tools=model, t_max=2).retrieve('sunlight')
        self.assertNotIn('secret-path', dumps(snapshots[0]))
        self.assertIn('secret-path.md', dumps(retriever.tree('isolated')))
        self.assertEqual(result.total_calls, 2)
        self.assertEqual(result.stop_reason, 'submitted')
        self.assertEqual(len(result.summary_searches), 2)
        self.assertEqual(result.initial_navigation['results'][0]['path'], 'summaries/topic-0-overview.md')
        self.assertEqual(result.embedding_usage['requests'], 3)
        self.assertEqual(result.evidence, [])  # Navigation never fabricates source reads.

    def test_no_summaries_keeps_tree_and_original_navigation_available(self):
        self.write('isolated/solo.md', '# A standalone knowledge page\n')
        retriever = self.retriever()
        self.assertEqual(retriever.initial_navigation('q')['results'], [])
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(retriever.tree('isolated')['entries'][0]['path'], 'isolated/solo.md')
        tree = self.retriever(summary_mode='tree')
        self.assertIn('directory_tree', tree.initial_navigation('q'))
        self.assertNotIn('summary_search', [t['function']['name'] for t in tree.tool_schemas])

    def test_invalid_requests_are_recoverable(self):
        retriever = self.retriever()
        for args in ({'query': ''}, {'query': 'q', 'limit': 0}, {'query': 'q', 'offset': -1},
                     {'query': 'q', 'exclude_paths': 'x'}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                retriever.summary_search(**args)
        with self.assertRaises(ValueError):
            retriever.read(['a', 'b'], offset=1)


class EmbeddingCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'embeddings.sqlite3'

    def response(self, vectors):
        return Mock(ok=True, status_code=200, json=Mock(return_value={
            'data': [{'index': i, 'embedding': vector} for i, vector in reversed(list(enumerate(vectors)))],
            'usage': {'prompt_tokens': 7}}))

    def test_response_order_cache_reuse_content_and_model_invalidation(self):
        with patch('embedding_client.requests.post') as post:
            post.return_value = self.response([[1., 0.], [0., 2.]])
            first = EmbeddingClient(self.path, model='model-a')
            self.assertEqual(first.embed(['alpha', 'beta']), [[1., 0.], [0., 1.]])
            same = EmbeddingClient(self.path, model='model-a')
            self.assertEqual(same.embed(['beta', 'alpha']), [[0., 1.], [1., 0.]])
            self.assertEqual(post.call_count, 1)
            post.return_value = self.response([[1., 1.]])
            same.embed(['alpha changed', 'beta'])
            self.assertEqual(post.call_args.kwargs['json']['input'], ['alpha changed'])
            EmbeddingClient(self.path, model='model-b').embed(['alpha'])
            self.assertEqual(post.call_count, 3)
            self.assertEqual(first.stats['input_tokens'], 7)

    def test_provider_isolation_does_not_forward_chat_credentials(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'private-key', 'OPENAI_BASE_URL': 'https://chat.example/v1',
                                       'EMBEDDING_BASE_URL': 'https://embeddings.example/v1', 'EMBEDDING_API_KEY': ''}):
            client = EmbeddingClient(self.path)
            self.assertEqual(client.api_key, '')

    def test_bad_vectors_fail_without_being_cached_or_exposing_response(self):
        with patch('embedding_client.requests.post') as post:
            post.return_value = self.response([[float('nan'), 1.]])
            with self.assertRaisesRegex(RuntimeError, 'malformed vectors'):
                EmbeddingClient(self.path).embed(['alpha'])
            post.return_value = self.response([[0., 1.]])
            self.assertEqual(EmbeddingClient(self.path).embed(['alpha']), [[0., 1.]])
            self.assertEqual(post.call_count, 2)
            post.return_value = Mock(ok=False, status_code=401, text='private-token-in-response')
            with self.assertRaisesRegex(RuntimeError, 'HTTP 401') as failure:
                EmbeddingClient(self.path).embed(['beta'])
            self.assertNotIn('private-token', str(failure.exception))


if __name__ == '__main__':
    unittest.main()
