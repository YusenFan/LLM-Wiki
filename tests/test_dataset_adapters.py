"""Offline coverage for MuSiQue/2Wiki adapters and resumable QA output."""

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import llm_wiki_bench
import download_datasets as download
import evaluate as evaluator
import preprocess_bench as preprocess
from benchmark_metrics import add_run_metrics, append_build_metrics
from qa_run_state import compact_predictions, select_pending


def archive(member, body):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as output:
        output.writestr(member, body)
    return buffer.getvalue()


def reply(body):
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_content.return_value = iter([body])
    return response


def musique_item(index):
    return {
        "id": f"m{index}", "question": f"mq{index}", "answer": f"ma{index}",
        "paragraphs": [{"title": f"mt{index}", "paragraph_text": f"mp{index}",
                        "is_supporting": True}],
    }


def wiki_item(index):
    return {
        "_id": f"w{index}", "question": f"wq{index}", "answer": f"wa{index}",
        "context": [[f"wt{index}", [f"wp{index}"]]],
        "supporting_facts": [[f"wt{index}", 0]],
    }


class DatasetAdapterTest(unittest.TestCase):
    def test_build_metrics_accumulate_across_resume_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            append_build_metrics(root, "musique", 10, 2.5, {"failed": 1}, {
                "by_model": {"glm": {"calls": 2, "prompt_tokens": 10,
                                       "completion_tokens": 4, "total_tokens": 14}}})
            report = append_build_metrics(root, "musique", 10, 1.5, {"failed": 0}, {
                "by_model": {"glm": {"calls": 1, "prompt_tokens": 5,
                                       "completion_tokens": 2, "total_tokens": 7}}})
            self.assertEqual(report["duration_seconds"], 4.0)
            self.assertEqual(report["attempt_count"], 2)
            self.assertEqual(report["llm_usage_totals"], {
                "calls": 3, "prompt_tokens": 15, "completion_tokens": 6, "total_tokens": 21})
            summary = add_run_metrics({}, root, {"q": {
                "elapsed_seconds": 3, "retrieval_llm_calls": 1,
                "retrieval_usage_by_model": {"glm": {"total_tokens": 9}},
                "embedding_usage": {"model": "text-embedding-3-large", "requests": 1,
                                    "input_tokens": 5, "cache_hits": 0, "embedded_texts": 1},
            }})
            self.assertEqual(summary["wiki_construction"]["llm_usage_totals"]["total_tokens"], 21)
            self.assertEqual(summary["qa_run"]["llm_usage_totals"]["total_tokens"], 9)
            self.assertEqual(summary["qa_run"]["embedding_usage_by_model"]
                             ["text-embedding-3-large"]["input_tokens"], 5)

    def test_summary_records_requested_missing_and_runtime_errors(self):
        qas = [
            {"id": "0", "question": "q0", "answer": "a", "supporting_titles": []},
            {"id": "1", "question": "q1", "answer": "b", "supporting_titles": []},
        ]
        summary, _ = evaluator.evaluate(qas, {
            "0": {"id": "0", "prediction": "unknown", "error": "timeout"}})
        self.assertEqual(
            (summary["requested"], summary["total"], summary["missing"], summary["errors"]),
            (2, 1, 1, 1),
        )

    def test_downloaders_extract_to_preprocessor_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            musique_body = (json.dumps(musique_item(0)) + "\n").encode()
            wiki_body = json.dumps([wiki_item(0)]).encode()
            with patch("requests.get", side_effect=[
                    reply(archive("musique_v1.0/data/musique_ans_v1.0_dev.jsonl", musique_body)),
                    reply(archive("release/data_ids/dev.json", wiki_body))]):
                musique_path = download.download_musique(root / "musique")
                wiki_path = download.download_2wikimhqa(root / "2wikimhqa")
            self.assertEqual(musique_path, root / preprocess.DATASET_PROCESSORS["musique"]["input_file"])
            self.assertEqual(wiki_path, root / preprocess.DATASET_PROCESSORS["2wikimhqa"]["input_file"])
            self.assertEqual(download._validate_musique(musique_path), 1)
            self.assertEqual(download._validate_2wikimhqa(wiki_path), 1)
            self.assertEqual(list(root.rglob(".dataset-*")), [])

    def test_first_100_are_selected_in_source_order_and_stale_articles_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            musique_source = root / "musique.jsonl"
            musique_source.write_text(
                "".join(json.dumps(musique_item(i)) + "\n" for i in range(105)),
                encoding="utf-8",
            )
            wiki_source = root / "wiki.json"
            wiki_source.write_text(json.dumps([wiki_item(i) for i in range(105)]), encoding="utf-8")
            for name, processor, source, prefix in (
                    ("musique", preprocess.process_musique, musique_source, "m"),
                    ("2wikimhqa", preprocess.process_2wikimhqa, wiki_source, "w")):
                raw = root / name / "raw"
                data = root / name / "data"
                stale = raw / "articles" / "stale.md"
                stale.parent.mkdir(parents=True)
                stale.write_text("old", encoding="utf-8")
                stats = processor(source, raw, data, limit=100, clean=True)
                rows = [json.loads(line) for line in (data / "qa_pairs.jsonl").read_text().splitlines()]
                self.assertEqual(stats["qa_count"], 100)
                self.assertEqual([row["id"] for row in rows], [f"{prefix}{i}" for i in range(100)])
                self.assertFalse(stale.exists())
                self.assertEqual(len(list((raw / "articles").glob("*.md"))), 100)

    def test_resume_reruns_only_missing_and_failed_then_compacts(self):
        qas = [{"id": str(i)} for i in range(3)]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.jsonl"
            output.write_text(
                json.dumps({"id": "0", "prediction": "ok", "error": None}) + "\n"
                + json.dumps({"id": "1", "prediction": "unknown", "error": "timeout"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual([qa["id"] for qa in select_pending(qas, output, True)], ["1", "2"])
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"id": "1", "prediction": "fixed", "error": None}) + "\n")
                stream.write(json.dumps({"id": "2", "prediction": "fixed", "error": None}) + "\n")
            state = compact_predictions(output, qas)
            self.assertEqual(state, {
                "requested": 3, "written": 3, "missing_ids": [], "failed_ids": []})
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["id"] for row in rows], ["0", "1", "2"])
            self.assertEqual(rows[1]["prediction"], "fixed")


if __name__ == "__main__":
    unittest.main()
