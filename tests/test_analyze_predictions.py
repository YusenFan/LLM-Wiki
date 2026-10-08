import json
import tempfile
import unittest
from pathlib import Path

from llm_wiki_bench.analyze_predictions import analyze, read_jsonl, write_outputs


class PredictionAnalysisTests(unittest.TestCase):
    def test_aliases_missing_and_unmatched_are_not_false_errors(self):
        qa = {'a': {'id': 'a', 'question': 'Who?', 'answer': 'Robert', 'answer_aliases': ['Bob']},
              'b': {'id': 'b', 'question': 'Where?', 'answer': 'London'}}
        summary, errors, missing = analyze(qa, {'a': {'prediction': 'Bob'}, 'extra': {'prediction': 'x'}})
        self.assertEqual((summary['evaluated'], summary['correct_em'], summary['missing_predictions']), (1, 1, 1))
        self.assertEqual(summary['unmatched_prediction_ids'], ['extra'])
        self.assertEqual(errors, [])
        self.assertEqual(missing, [qa['b']])

    def test_runtime_failure_and_missing_telemetry(self):
        qa = {'a': {'id': 'a', 'question': 'Who?', 'answer': 'Bob', 'supporting_titles': ['Bob']}}
        _, errors, _ = analyze(qa, {'a': {'prediction': 'unknown', 'stop_reason': 'model_error'}})
        self.assertEqual(errors[0]['category'], 'runtime_failure')
        self.assertIsNone(errors[0]['missing_supporting_titles'])
        self.assertIsNone(errors[0]['gold_span_in_read_evidence'])

    def test_overlap_is_only_a_signal_and_tool_failures_survive_export(self):
        qa = {'a': {'id': 'a', 'question': 'Which show?', 'answer': 'Teen Titans Go!'}}
        pred = {'prediction': 'Teen Titans', 'tool_calls': [{'tool': 'finish_answer', 'result': '{"error":"bad evidence"}'}]}
        summary, errors, missing = analyze(qa, {'a': pred})
        self.assertEqual(summary['wrong_em'], 1)
        self.assertEqual(errors[0]['diagnosis_status'], 'heuristic_requires_review')
        self.assertEqual(errors[0]['tool_errors'][0]['error'], 'bad evidence')
        summary['predictions_path'] = 'example.jsonl'
        with tempfile.TemporaryDirectory() as temp:
            write_outputs(Path(temp), summary, errors, missing)
            saved = json.loads((Path(temp) / 'wrong_questions.jsonl').read_text())
            self.assertEqual(saved['prediction_record'], pred)

    def test_duplicate_last_wins_and_bad_input_reports_line(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'predictions.jsonl'
            path.write_text('{"id":"a","prediction":"first"}\n\n{"id":"a","prediction":"last"}\n')
            records, duplicates = read_jsonl(path)
            self.assertEqual(records['a']['prediction'], 'last')
            self.assertEqual(duplicates, ['a'])
            path.write_text('{"id":"a"}\ninvalid\n')
            with self.assertRaisesRegex(ValueError, ':2:'):
                read_jsonl(path)

    def test_rejects_wrong_qa_file(self):
        qa = {'a': {'id': 'a', 'question': 'Who?', 'answer': 'Bob'}}
        with self.assertRaisesRegex(ValueError, 'gold_answer differs'):
            analyze(qa, {'a': {'prediction': 'Bob', 'gold_answer': 'Alice'}})


if __name__ == '__main__':
    unittest.main()
