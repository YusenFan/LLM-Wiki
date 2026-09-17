"""Build a separate wiki from the first ten HotpotQA questions' full context.

Run from the project root: python build_test_ten.py
Export API settings before running; this script makes model API calls.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "llm_wiki_bench"))

import bench_config as config
from bench_ingest import ingest_batch
from preprocess_bench import process_hotpotqa


def main() -> None:
    input_path = (
        ROOT / "llm_wiki_bench/datasets/hotpotqa/hotpot_dev_distractor_v1.json"
    )
    if not input_path.is_file():
        raise SystemExit(f"Dataset not found: {input_path}")

    test_dir = ROOT / "wiki_output/hotpotqa/test-ten"
    print("Preparing context articles for the first 10 HotpotQA questions...", flush=True)
    processed = process_hotpotqa(
        input_path=input_path,
        output_dir=test_dir / "raw",
        data_dir=test_dir / "data",
        limit=10,
    )

    config.set_dataset("hotpotqa", wiki_dir=test_dir / "wiki")
    config.ensure_wiki_dirs()

    # Use this run's files, not stale names left by a different preprocessing version.
    articles = sorted(Path(path) for path in processed["article_paths"])
    if not articles:
        raise SystemExit("No articles found for the first ten test questions.")
    print(f"Building from {len(articles)} articles into {test_dir / 'wiki'}", flush=True)
    stats = ingest_batch(articles, batch_size=3)
    # Keep the build result even when the terminal output has scrolled away.
    report_path = test_dir / "build-result.json"
    report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Build report: {report_path}")
    if stats["failed"] or stats["summaries"]["failed"]:
        raise SystemExit("构建存在失败，请先查看上面的错误。")

    print(f"构建完成：{test_dir / 'wiki'}")


if __name__ == "__main__":
    main()
