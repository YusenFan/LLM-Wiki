"""The one-question wrapper must preserve initialization and report build failures."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_test_one


class TestBuildOne(unittest.TestCase):
    def test_one_question_uses_ten_articles_and_keeps_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'llm_wiki_bench/datasets/hotpotqa/hotpot_dev_distractor_v1.json'
            dataset.parent.mkdir(parents=True)
            dataset.write_text(json.dumps([
                {'_id': 'one', 'question': 'Q1?', 'answer': 'A',
                 'context': [[f'Article {i}', [f'Article {i} text.']] for i in range(10)]},
                {'_id': 'two', 'question': 'Q2?', 'answer': 'B', 'context': [['Excluded', ['Not in test.']]]},
            ]))
            stats = {'success': 10, 'failed': 0, 'summaries': {'failed': 0}}
            with patch.object(build_test_one, 'ROOT', root), \
                 patch.object(build_test_one.config, 'set_dataset') as configure, \
                 patch.object(build_test_one.config, 'ensure_wiki_dirs') as initialize, \
                 patch.object(build_test_one, 'ingest_batch', return_value=stats) as ingest:
                build_test_one.main()
            initialize.assert_called_once_with()
            configure.assert_called_once_with('hotpotqa', wiki_dir=root / 'wiki_output/hotpotqa/test-one/wiki')
            paths = ingest.call_args.args[0]
            self.assertEqual(len(paths), 10)
            self.assertTrue(all(path.parent == root / 'wiki_output/hotpotqa/test-one/raw/articles' for path in paths))
            self.assertFalse(any(path.stem == 'Excluded' for path in paths))
            report = root / 'wiki_output/hotpotqa/test-one/build-result.json'
            self.assertEqual(json.loads(report.read_text()), stats)

    def test_failure_report_is_written_before_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / 'llm_wiki_bench/datasets/hotpotqa/hotpot_dev_distractor_v1.json'
            dataset.parent.mkdir(parents=True)
            dataset.write_text(json.dumps([{'_id': 'one', 'question': 'Q?', 'answer': 'A',
                                           'context': [['Article', ['Article text.']]]}]))
            stats = {'success': 0, 'failed': 1, 'summaries': {'failed': 0}}
            with patch.object(build_test_one, 'ROOT', root), \
                 patch.object(build_test_one.config, 'set_dataset'), \
                 patch.object(build_test_one.config, 'ensure_wiki_dirs'), \
                 patch.object(build_test_one, 'ingest_batch', return_value=stats):
                with self.assertRaises(SystemExit) as stopped:
                    build_test_one.main()
            self.assertNotEqual(stopped.exception.code, 0)
            report = root / 'wiki_output/hotpotqa/test-one/build-result.json'
            self.assertEqual(json.loads(report.read_text()), stats)
