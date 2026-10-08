"""Persistent timing and token metrics for benchmark build and QA runs."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def latency_percentiles(rows: list[dict]) -> dict:
    values = sorted(float(row['elapsed_seconds']) for row in rows
                    if isinstance(row.get('elapsed_seconds'), (int, float))
                    and math.isfinite(row['elapsed_seconds']))
    def percentile(fraction):
        if not values:
            return None
        position = (len(values) - 1) * fraction
        left, right = math.floor(position), math.ceil(position)
        return values[left] + (values[right] - values[left]) * (position - left)
    return {"p50_elapsed_seconds": percentile(.5), "p95_elapsed_seconds": percentile(.95)}


def _merge_usage(*groups: dict) -> dict:
    merged: dict[str, dict] = {}
    for group in groups:
        for model, usage in (group or {}).items():
            bucket = merged.setdefault(model, {"calls": 0, **{key: 0 for key in TOKEN_FIELDS}})
            for key in ("calls", *TOKEN_FIELDS):
                bucket[key] += int((usage or {}).get(key, 0) or 0)
    return merged


def usage_totals(by_model: dict) -> dict:
    return {
        key: sum(int((usage or {}).get(key, 0) or 0) for usage in (by_model or {}).values())
        for key in ("calls", *TOKEN_FIELDS)
    }


def append_build_metrics(wiki_dir: Path, dataset: str, article_count: int,
                         duration_seconds: float, stats: dict, usage: dict) -> dict:
    """Append one resumable build attempt and persist cumulative construction metrics."""
    path = Path(wiki_dir) / "build_metrics.json"
    attempts = []
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("dataset") == dataset:
            attempts = list(previous.get("attempts", []))
    attempt = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration_seconds, 3),
        "article_count": article_count,
        "stats": stats,
        "llm_usage_by_model": usage.get("by_model", {}),
        "llm_usage_totals": usage.get("totals", usage_totals(usage.get("by_model", {}))),
    }
    attempts.append(attempt)
    by_model = _merge_usage(*(item.get("llm_usage_by_model", {}) for item in attempts))
    report = {
        "schema": "llm-wiki-build-metrics-v1",
        "dataset": dataset,
        "wiki_dir": str(Path(wiki_dir).resolve()),
        "article_count": article_count,
        "attempt_count": len(attempts),
        "duration_seconds": round(sum(float(item.get("duration_seconds", 0)) for item in attempts), 3),
        "llm_usage_by_model": by_model,
        "llm_usage_totals": usage_totals(by_model),
        "latest_stats": stats,
        "attempts": attempts,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return report


def add_run_metrics(summary: dict, wiki_dir: Path, predictions: dict[str, dict],
                    run_metadata: dict | None = None) -> dict:
    """Attach construction and final compacted QA usage to an evaluation summary."""
    build_path = Path(wiki_dir) / "build_metrics.json"
    summary["wiki_construction"] = (
        json.loads(build_path.read_text(encoding="utf-8")) if build_path.exists() else None
    )
    rows = list(predictions.values())
    by_model = _merge_usage(*(row.get("retrieval_usage_by_model", {}) for row in rows))
    embedding: dict[str, dict] = {}
    for row in rows:
        usage = row.get("embedding_usage") or {}
        model = usage.get("model")
        if not model:
            continue
        bucket = embedding.setdefault(model, {
            "requests": 0, "input_tokens": 0, "cache_hits": 0, "embedded_texts": 0})
        for key in bucket:
            bucket[key] += int(usage.get(key, 0) or 0)
    summary["qa_run"] = {
        "duration_seconds": round(sum(float(row.get("elapsed_seconds", 0) or 0) for row in rows), 3),
        "llm_calls": sum(int(row.get("retrieval_llm_calls", 0) or 0) for row in rows),
        "llm_usage_by_model": by_model,
        "llm_usage_totals": usage_totals(by_model),
        "embedding_usage_by_model": embedding,
        **latency_percentiles(rows),
    }
    if run_metadata is not None:
        summary["qa_run"]["startup_seconds"] = run_metadata.get("startup_seconds")
        summary["qa_run"]["index_preparation"] = run_metadata.get("index_preparation")
        summary["qa_run"]["qa_wall_seconds"] = run_metadata.get("qa_wall_seconds")
    return summary
