"""LLM-Wiki Benchmark — runner.

Chains the offline wiki-construction pipeline:

    1. Download the public dev set (HotpotQA / MuSiQue / 2WikiMultiHopQA)
    2. Pre-process context paragraphs into Markdown articles
    3. Ingest articles into a structured wiki

Usage:

    # Full pipeline
    python -m llm_wiki_bench.run --dataset hotpotqa --limit 500

    # Skip stages
    python -m llm_wiki_bench.run --dataset hotpotqa --only-download
    python -m llm_wiki_bench.run --dataset hotpotqa --only-preprocess --limit 500
    python -m llm_wiki_bench.run --dataset hotpotqa --only-ingest    --limit 500

    # All three datasets
    python -m llm_wiki_bench.run --all --limit 500
"""

import argparse
import sys
import time
from pathlib import Path

# Make sibling modules importable when this file is launched directly
# (e.g. `python release/llm_wiki_bench/run.py ...`).
_BENCH_DIR = Path(__file__).parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.append(str(_BENCH_DIR))


# 【下载编排】查数据集下载器并执行，打印阶段信息；以返回路径是否存在值判断成功，不调用模型。
def step_download(dataset: str) -> bool:
    """Download the public dev set."""
    from download_datasets import DATASET_DOWNLOADERS, DATASETS_DIR

    print(f"\n{'='*60}")
    print(f"  [1/3] Download dataset: {dataset}")
    print(f"{'='*60}")

    downloader = DATASET_DOWNLOADERS.get(dataset)
    if not downloader:
        print(f"  ❌ Unknown dataset: {dataset}")
        return False

    ds_dir = DATASETS_DIR / dataset
    return downloader(ds_dir) is not None


# 【预处理编排】检查下载文件并选择数据集处理器，传 limit 限制 QA 样本数；写文章和 qa_pairs，返回是否成功。
def step_preprocess(dataset: str, limit: int | None = None) -> bool:
    """Pre-process context paragraphs into Markdown articles."""
    from preprocess_bench import DATASET_PROCESSORS, DATASETS_DIR, RAW_DIR, DATA_DIR

    print(f"\n{'='*60}")
    print(f"  [2/3] Preprocess dataset: {dataset}")
    print(f"{'='*60}")

    info = DATASET_PROCESSORS.get(dataset)
    if not info:
        print(f"  ❌ Unknown dataset: {dataset}")
        return False

    input_path = DATASETS_DIR / info["input_file"]
    if not input_path.exists():
        print(f"  ❌ Dataset file not found: {input_path}")
        return False

    output_dir = RAW_DIR / dataset
    data_dir = DATA_DIR / dataset
    stats = info["processor"](input_path, output_dir, data_dir, limit=limit)
    print(
        f"  ✅ Preprocessed: {stats['qa_count']} QA pairs, "
        f"{stats['paragraph_count']} articles"
    )
    return True


# 【构建编排／间接 LLM】配置数据集、创建目录，排序读取全部已有文章后调用当前 ingest_batch；文档或摘要失败均返回 False。
# 这里不接收 run 的 limit。
def step_ingest(dataset: str, batch_size: int = 3, force: bool = False) -> bool:
    """Build the wiki by ingesting all preprocessed articles."""
    import bench_config as config
    import bench_ingest

    print(f"\n{'='*60}")
    print(f"  [3/3] Ingest articles into wiki: {dataset}")
    print(f"{'='*60}")

    config.set_dataset(dataset)
    config.ensure_wiki_dirs()

    raw_dir = config.RAW_DIR
    if not raw_dir or not raw_dir.exists():
        print(f"  ❌ Raw articles not found: {raw_dir}")
        return False

    article_paths = sorted(raw_dir.glob("*.md"))
    if not article_paths:
        print(f"  ❌ No articles in {raw_dir}")
        return False

    print(f"  📂 Articles: {len(article_paths)}")
    print(f"  📁 Wiki output: {config.WIKI_DIR}")

    result = bench_ingest.ingest_batch(article_paths, batch_size=batch_size, force=force)
    return result["failed"] == 0 and result.get('summaries', {}).get('failed', 0) == 0


# 【阶段控制】按 only/skip 参数依次运行下载、预处理和构建，前序失败阻止后续；打印耗时并返回总状态。
def run_one(dataset: str, args) -> bool:
    t0 = time.time()
    ok = True
    if args.only_download or args.only is None or args.only == "download":
        if not args.skip_download:
            ok = ok and step_download(dataset)
    if args.only_preprocess or args.only is None or args.only == "preprocess":
        if not args.skip_preprocess and ok:
            ok = ok and step_preprocess(dataset, limit=args.limit)
    if args.only_ingest or args.only is None or args.only == "ingest":
        if ok:
            ok = ok and step_ingest(
                dataset, batch_size=args.batch_size, force=args.force
            )
    print(f"\n  ⏱  Total time for {dataset}: {time.time() - t0:.1f}s")
    return ok


# 【离线 CLI】解析数据集和阶段选项，互斥校验 only 参数后逐数据集运行；--limit 限制预处理题数，不限制已有文章的 only-ingest。
def main():
    parser = argparse.ArgumentParser(
        description="LLM-Wiki benchmark runner (offline wiki construction)."
    )
    parser.add_argument(
        "--dataset", "-d",
        choices=["hotpotqa", "musique", "2wikimhqa"],
        help="Dataset to process. Use --all to run all three.",
    )
    parser.add_argument("--all", action="store_true",
                        help="Run all supported datasets sequentially.")
    parser.add_argument("--limit", "-n", type=int, default=500,
                        help="Limit the number of QA examples processed (default: 500).")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="Compatibility option; documents now build independently.")
    parser.add_argument("--force", action="store_true",
                        help="Re-ingest articles even if cached.")

    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")

    parser.add_argument("--only-download", action="store_true")
    parser.add_argument("--only-preprocess", action="store_true")
    parser.add_argument("--only-ingest", action="store_true")

    args = parser.parse_args()

    # Cross-flag fix-up: --only-X takes precedence and disables sibling stages.
    only_flags = [args.only_download, args.only_preprocess, args.only_ingest]
    args.only = None
    if sum(only_flags) > 1:
        parser.error("Use at most one of --only-download / --only-preprocess / --only-ingest.")
    if args.only_download:
        args.only = "download"
    elif args.only_preprocess:
        args.only = "preprocess"
    elif args.only_ingest:
        args.only = "ingest"

    if not args.dataset and not args.all:
        parser.error("Specify --dataset DATASET or --all.")

    datasets = ["hotpotqa", "musique", "2wikimhqa"] if args.all else [args.dataset]
    overall_ok = True
    for ds in datasets:
        overall_ok &= run_one(ds, args)

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
