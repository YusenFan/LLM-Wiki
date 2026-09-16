"""Keep independent remote improvements compatible with the current article workflow."""

import json
import tempfile
import unittest
from pathlib import Path

import llm_wiki_bench
from preprocess_bench import process_hotpotqa, process_musique, process_2wikimhqa
from wiki_agent import WikiAgent
from wiki_retriever import WikiRetriever


class MergeIntegrationTest(unittest.TestCase):
    def test_preprocessing_keeps_title_variants_and_filename_collisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'data.json'
            source.write_text(json.dumps([{
                '_id': '1', 'question': 'q', 'answer': 'a',
                'context': [['A/B', ['old']], ['A_B', ['other']], ['A/B', ['new']]],
            }]))
            for process in (process_hotpotqa, process_2wikimhqa):
                stats = process(source, root / process.__name__, root / 'qa')
                paths = [Path(path) for path in stats['article_paths']]
                self.assertEqual(stats['paragraph_count'], 3)
                self.assertEqual(len(set(paths)), 3)
                self.assertTrue(any('old' in path.read_text() for path in paths))
                self.assertTrue(any('new' in path.read_text() for path in paths))
            musique = root / 'musique.jsonl'
            musique.write_text(json.dumps({'id': '1', 'question': 'q', 'answer': 'a', 'paragraphs': [
                {'title': 'A', 'paragraph_text': text} for text in ['x', 'y']]}) + '\n')
            stats = process_musique(musique, root / 'musique', root / 'qa')
            self.assertEqual(stats['paragraph_count'], 2)
            self.assertEqual(len(stats['article_paths']), 2)

    def test_usage_metadata_is_recorded_without_returning_it_to_api(self):
        with tempfile.TemporaryDirectory() as temporary:
            observed = []
            def model(messages, **kwargs):
                self.assertTrue(all('_usage' not in message for message in messages))
                observed.append(len(messages))
                return {'role': 'assistant', 'content': None, '_usage': {'total_tokens': 15},
                        'tool_calls': [{'id': str(len(observed)), 'type': 'function', 'function': {
                            'name': 'wiki_read', 'arguments': json.dumps({'paths': ['/']})}}]}
            result = WikiAgent(WikiRetriever(Path(temporary)), call_llm_with_tools=model,
                               model='test-model', t_max=2).retrieve('q')
            self.assertEqual(result.llm_calls, 2)
            self.assertEqual(result.usage_by_model['test-model']['total_tokens'], 30)
            self.assertEqual(len(observed), 2)


if __name__ == '__main__':
    unittest.main()
