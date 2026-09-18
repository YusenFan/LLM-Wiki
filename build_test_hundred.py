"""Build an isolated wiki from all context of the first 100 HotpotQA questions."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "llm_wiki_bench"))

import bench_config as config
from bench_ingest import ingest_batch
from preprocess_bench import process_hotpotqa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    input_path = ROOT / "llm_wiki_bench/datasets/hotpotqa/hotpot_dev_distractor_v1.json"
    output = ROOT / "wiki_output/hotpotqa/first-100"
    if len(json.loads(input_path.read_text(encoding="utf-8"))) < 100:
        raise SystemExit("Dataset must contain at least 100 questions.")
    if not args.prepare_only and not os.environ.get("OPENAI_API_KEY", "").strip():
        raise SystemExit("Set OPENAI_API_KEY before starting the build.")
    processed = process_hotpotqa(input_path, output / "raw", output / "data", limit=100)
    if processed["qa_count"] != 100:
        raise SystemExit("Expected exactly 100 questions.")
    articles = sorted(Path(path) for path in processed["article_paths"])
    manifest = {"qa_count": processed["qa_count"], "article_count": len(articles),
                "article_paths": [str(path.relative_to(ROOT)) for path in articles]}
    (output / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared 100 questions and {len(articles)} context articles.", flush=True)
    if args.prepare_only:
        return
    config.set_dataset("hotpotqa", wiki_dir=output / "wiki")
    config.ensure_wiki_dirs()
    stats = ingest_batch(articles, batch_size=3)
    (output / "build-result.json").write_text(json.dumps(stats, indent=2) + "\n")
    if stats["failed"] or stats["summaries"]["failed"]:
        raise SystemExit("Build has failures; inspect logs and build-result.json.")
    print(f"Build complete: {output / 'wiki'}", flush=True)


if __name__ == "__main__":
    main()
