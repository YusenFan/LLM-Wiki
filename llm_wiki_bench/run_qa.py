"""End-to-end QA runner over a compiled LLM-Wiki.

For each QA pair this script

  1. lets the premium Answer Agent explore and verify original evidence,
  2. validates its final answer, citations, and evidence gaps,
  3. writes a JSONL prediction file, and (optionally) runs evaluation.

Usage:

    python -m llm_wiki_bench.run_qa --dataset hotpotqa --limit 5
    python -m llm_wiki_bench.run_qa --dataset hotpotqa --limit 500 --evaluate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Make sibling modules importable when launched as `python release/.../run_qa.py`
_BENCH_DIR = Path(__file__).parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.append(str(_BENCH_DIR))

import bench_config as config                        # noqa: E402
import evaluate as _evaluate                         # noqa: E402
from llm_client import call_llm_with_tools  # noqa: E402
from wiki_agent import WikiAgent                     # noqa: E402
from wiki_retriever import WikiRetriever             # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Retrieval-as-Reasoning QA over a compiled LLM-Wiki."
    )
    parser.add_argument("--dataset", "-d", required=True,
                        choices=["hotpotqa", "musique", "2wikimhqa"])
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Process only the first N QA pairs.")
    parser.add_argument("--t-max", type=int, default=30,
                        help="Maximum tool-call budget per question (default 30).")
    parser.add_argument("--patience", type=int, default=3,
                        help="Deprecated compatibility option; exploration now stops by evidence or budget.")
    parser.add_argument("--select-pages", type=int, default=5,
                        help="Maximum pages selected per wiki_search (default 5).")
    parser.add_argument("--retrieval-model", default=None,
                        help="Deprecated compatibility option; QA uses one answer model.")
    parser.add_argument("--answer-model", default=None,
                        help="Primary exploration and answer model (default: LLM_PREMIUM_MODEL).")
    parser.add_argument("--search-mode", choices=["bm25", "exact_then_bm25"], default="bm25")
    parser.add_argument("--no-subtasks", action="store_true", help="Compatibility option; subtasks are already disabled.")
    parser.add_argument("--output", "-o", default=None,
                        help="Predictions output path (default: results/<dataset>/predictions.jsonl).")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation immediately after prediction.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # 1. Locate the compiled Wiki for this dataset.
    config.set_dataset(args.dataset)
    wiki_dir = Path(config.WIKI_DIR)
    if not wiki_dir.exists():
        sys.exit(f"❌ Compiled wiki not found: {wiki_dir}\n"
                 f"   Build it first: python -m llm_wiki_bench.run --dataset {args.dataset}")

    # 2. Load QA pairs.
    qa_path = Path(config.BASE_DIR) / "data" / args.dataset / "qa_pairs.jsonl"
    if not qa_path.exists():
        sys.exit(f"❌ QA pairs not found: {qa_path}\n"
                 f"   Run: python -m llm_wiki_bench.run --dataset {args.dataset} --only-preprocess")
    with open(qa_path, encoding="utf-8") as f:
        qa_pairs = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        qa_pairs = qa_pairs[: args.limit]

    print(f"Wiki dir : {wiki_dir}")
    print(f"QA pairs : {len(qa_pairs)}")
    print(f"T_max={args.t_max}  P={args.patience}  k={args.select_pages}")

    # 3. Initialize retriever + agent.
    retriever = WikiRetriever(wiki_dir, search_mode=args.search_mode)
    retriever.load()
    if not retriever.pages:
        sys.exit("Compiled Wiki has no readable pages; build it before running QA.")
    print(f"Loaded   : {len(retriever.pages)} pages, {len(retriever.dir_indexes)} directory indices")

    agent = WikiAgent(
        retriever,
        call_llm_with_tools=call_llm_with_tools,
        model=args.answer_model or config.LLM_PREMIUM_MODEL,
        allow_subtasks=False,
        t_max=args.t_max,
        patience=args.patience,
        select_pages=args.select_pages,
        verbose=args.verbose,
    )

    # 4. Run.
    out_path = Path(args.output) if args.output else (
        Path(config.BASE_DIR) / "results" / args.dataset / "predictions.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, qa in enumerate(qa_pairs, 1):
            question = qa["question"]
            try:
                rr = agent.retrieve(question)
                prediction = rr.answer
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}/{len(qa_pairs)}] ❌ {qa['id']}: {e}")
                prediction = "unknown"
                rr = None

            record = {
                "id": qa["id"],
                "question": question,
                "prediction": prediction,
                "gold_answer": qa.get("answer", ""),
                "retrieved_titles": [name for _, name in (rr.pages if rr else [])],
                "retrieval_trace": rr.trace if rr else [],
                "retrieval_steps": rr.total_calls if rr else 0,
                "citations": rr.citations if rr else [],
                "summary_refs": rr.summary_refs if rr else [],
                "reasoning": rr.reasoning if rr else "",
                "status": rr.status if rr else "failed",
                "evidence_gaps": rr.evidence_gaps if rr else ["QA execution failed"],
                "evidence_updates": rr.evidence_updates if rr else [],
                "tool_trace": rr.tool_calls if rr else [],
                "llm_calls": rr.llm_calls if rr else 0,
                "usage_by_model": rr.usage_by_model if rr else {},
                "elapsed_seconds": rr.elapsed_seconds if rr else 0,
                "search_mode": args.search_mode,
                "candidate_limit": args.select_pages,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if i % 10 == 0 or i == len(qa_pairs) or args.verbose:
                elapsed = time.time() - t0
                print(f"  [{i}/{len(qa_pairs)}] {qa['id']}  steps={record['retrieval_steps']}  "
                      f"pages={len(record['retrieved_titles'])}  ({elapsed:.1f}s)")

    print(f"\nPredictions saved: {out_path}")
    print(f"Total time       : {time.time() - t0:.1f}s")

    # 5. Optional evaluation.
    if args.evaluate:
        predictions = _evaluate._load_predictions(out_path)
        summary, details = _evaluate.evaluate(qa_pairs, predictions)
        if summary:
            _evaluate._print_summary(summary, args.dataset)
            results_dir = Path(config.BASE_DIR) / "results" / args.dataset
            results_dir.mkdir(parents=True, exist_ok=True)
            with open(results_dir / f"{args.dataset}_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            with open(results_dir / f"{args.dataset}_details.jsonl", "w", encoding="utf-8") as f:
                for d in details:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
