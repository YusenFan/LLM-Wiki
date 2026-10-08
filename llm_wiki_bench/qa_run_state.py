"""Durable prediction-file helpers for resumable benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path


def load_predictions(path: Path) -> dict[str, dict]:
    """Load JSONL records by id; a later retry replaces an earlier record."""
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not item.get("id"):
                raise ValueError(f"invalid prediction record at line {line_number}")
            records[str(item["id"])] = item
    return records


def prediction_succeeded(record: dict | None) -> bool:
    if not record or record.get("error"):
        return False
    if record.get("evidence_status") == "error":
        return False
    return record.get("stop_reason") not in {"error", "model_error"}


def select_pending(qa_pairs: list[dict], output: Path, resume: bool) -> list[dict]:
    if not resume:
        return list(qa_pairs)
    existing = load_predictions(output)
    return [qa for qa in qa_pairs if not prediction_succeeded(existing.get(str(qa["id"])))]


def compact_predictions(output: Path, qa_pairs: list[dict]) -> dict:
    """Atomically keep one latest record per requested id, in dataset order."""
    records = load_predictions(output)
    requested_ids = [str(qa["id"]) for qa in qa_pairs]
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for qid in requested_ids:
            if qid in records:
                stream.write(json.dumps(records[qid], ensure_ascii=False) + "\n")
    temporary.replace(output)
    missing = [qid for qid in requested_ids if qid not in records]
    failed = [qid for qid in requested_ids if qid in records and not prediction_succeeded(records[qid])]
    return {"requested": len(requested_ids), "written": len(requested_ids) - len(missing),
            "missing_ids": missing, "failed_ids": failed}
