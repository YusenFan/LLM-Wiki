#!/usr/bin/env python3
"""Pre-process multi-hop QA datasets: convert context paragraphs to Markdown articles.

Supported dataset formats:
- HotpotQA: JSON array; each item has question, answer, context (list of [title, sentences]).
- MuSiQue: JSONL; each item has question, answer, paragraphs (list of {title, paragraph_text, is_supporting}).
- 2WikiMultiHopQA: JSON array; each item has question, answer, context (list of [title, sentences]).

Outputs:
- raw/{dataset}/articles/{title_slug}.md  — one Markdown file per unique paragraph.
- data/{dataset}/qa_pairs.jsonl           — QA pairs (question, answer, supporting_titles).

Usage:
    python preprocess_bench.py --dataset hotpotqa
    python preprocess_bench.py --dataset musique
    python preprocess_bench.py --dataset 2wikimhqa
    python preprocess_bench.py --all
    python preprocess_bench.py --dataset hotpotqa --limit 500
"""

import hashlib
import yaml
import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent          # repo root (same as bench_config / evaluate / run_qa)
DATASETS_DIR = Path(__file__).parent / "datasets"  # same as download_datasets.DATASETS_DIR
RAW_DIR = BASE_DIR / "raw"
DATA_DIR = BASE_DIR / "data"


# 【文件名清洗】将非法文件名字符替换为下划线并规范空白、截长；碰撞由 _write_articles 的哈希后缀处理。
def sanitize_filename(name: str) -> str:
    """Strip characters that are illegal in file names."""
    # Strip filesystem-illegal characters.
    name = re.sub(r'[<>:"/\\|?*\[\]]', '_', name)
    # Collapse runs of whitespace.
    name = re.sub(r'\s+', ' ', name).strip()
    # Strip leading/trailing dots and spaces.
    name = name.strip('. ')
    # Enforce maximum length.
    return name[:200] if name else "untitled"


# 【文章落盘】将去重后的标题／正文变体写成 Markdown，保留明确 source_identity；文件名加哈希避免清洗后的同名碰撞。
# 原文版本稍后由 SourceStore 对完整输入计算。
def _write_articles(paragraphs: dict, output_dir: Path, dataset: str) -> int:
    """Preserve all title/text variants; source identity is separate from version.

    Wikipedia title identifies a source in this dataset, never a Wiki entity.
    Hash suffixes prevent collisions from filename sanitization and title truncation.
    """
    articles_dir = output_dir / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)
    for (title, text) in paragraphs:
        identity = f"{dataset}:wikipedia:{title}"
        sid = hashlib.sha256(identity.encode()).hexdigest()
        version = hashlib.sha256(text.encode()).hexdigest()
        metadata = {"source_identity": identity, "source_id": sid,
                    "source_type": "wikipedia", "title": title, "dataset": dataset}
        content = "---\n" + yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False) + "---\n\n# " + title + "\n\n" + text + "\n"
        path = articles_dir / f"{sanitize_filename(title)[:100]}--{sid[:16]}--{version[:16]}.md"
        path.write_text(content, encoding="utf-8")
    return len(paragraphs)


# 【数据转换】读取 HotpotQA JSON，按 limit 选题，收集 context 中全部不同标题／正文组合（含干扰段落），写文章和统一 qa_pairs.jsonl，返回数量统计。
def process_hotpotqa(input_path: Path, output_dir: Path, data_dir: Path,
                     limit: int = None) -> dict:
    """Process the HotpotQA distractor dev set.

    Format::

        [{"_id": "...", "question": "...", "answer": "...",
          "type": "bridge"|"comparison", "level": "easy"|"medium"|"hard",
          "supporting_facts": [[title, sent_idx], ...],
          "context": [[title, [sent1, sent2, ...]], ...]}, ...]

    Each QA item has 10 context paragraphs (2 gold + 8 distractors).
    We extract all unique (title, paragraph) pairs as articles.
    """
    print(f"  📖 Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit:
        data = data[:limit]
        print(f"  🔢 Limiting to first {limit} QA items")

    # Collect all unique paragraphs (deduplicated by exact title and text).
    paragraphs = {}  # title -> full_text
    qa_pairs = []

    for item in data:
        question = item["question"]
        answer = item["answer"]
        supporting_titles = list(set(t for t, _ in item.get("supporting_facts", [])))

        # Extract context paragraphs.
        for title, sentences in item.get("context", []):
            full_text = " ".join(sentences)
            paragraphs[(title, full_text)] = True

        qa_pairs.append({
            "id": item.get("_id", ""),
            "question": question,
            "answer": answer,
            "type": item.get("type", ""),
            "level": item.get("level", ""),
            "supporting_titles": supporting_titles,
        })

    written = _write_articles(paragraphs, output_dir, "hotpotqa")

    # Write QA pairs.
    data_dir.mkdir(parents=True, exist_ok=True)
    qa_path = data_dir / "qa_pairs.jsonl"
    with open(qa_path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    return {
        "qa_count": len(qa_pairs),
        "paragraph_count": written,
        "unique_titles": len({title for title, _ in paragraphs}),
    }


# 【数据转换】读取 MuSiQue JSONL，整理问题、别名及支撑标题并收集不同段落，输出统一文章与 QA 文件；不调用模型。
def process_musique(input_path: Path, output_dir: Path, data_dir: Path,
                    limit: int = None) -> dict:
    """Process the MuSiQue-Ans dev set.

    Format (JSONL)::

        {
          "id": "...",
          "question": "...",
          "answer": "...",
          "answer_aliases": ["..."],
          "answerable": true/false,
          "paragraphs": [
            {"idx": 0, "title": "...",
             "paragraph_text": "...", "is_supporting": true/false},
            ...
          ],
          "question_decomposition": [...]
        }
    """
    print(f"  📖 Reading {input_path}...")
    items = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if limit:
        items = items[:limit]
        print(f"  🔢 Limiting to first {limit} QA items")

    # Collect all unique paragraphs.
    paragraphs = {}  # title -> text
    qa_pairs = []

    for item in items:
        # Only process answerable questions.
        if not item.get("answerable", True):
            continue

        question = item["question"]
        answer = item["answer"]
        supporting_titles = []

        for para in item.get("paragraphs", []):
            title = para["title"]
            text = para["paragraph_text"]
            paragraphs[(title, text)] = True
            if para.get("is_supporting", False):
                supporting_titles.append(title)

        qa_pairs.append({
            "id": item.get("id", ""),
            "question": question,
            "answer": answer,
            "answer_aliases": item.get("answer_aliases", []),
            "supporting_titles": list(set(supporting_titles)),
        })

    written = _write_articles(paragraphs, output_dir, "musique")

    # Write QA pairs.
    data_dir.mkdir(parents=True, exist_ok=True)
    qa_path = data_dir / "qa_pairs.jsonl"
    with open(qa_path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    return {
        "qa_count": len(qa_pairs),
        "paragraph_count": written,
        "unique_titles": len({title for title, _ in paragraphs}),
    }


# 【数据转换】读取 2Wiki JSON，提取题目、支撑标题与上下文文章，保留标题／正文变体并写统一文件；不调用模型。
def process_2wikimhqa(input_path: Path, output_dir: Path, data_dir: Path,
                      limit: int = None) -> dict:
    """Process the 2WikiMultiHopQA dev set.

    Format::

        [{"_id": "...", "question": "...", "answer": "...",
          "type": "bridge"|"comparison"|"inference"|...,
          "supporting_facts": [[title, sent_idx], ...],
          "context": [[title, [sent1, sent2, ...]], ...],
          "evidences": [...]}, ...]
    """
    print(f"  📖 Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit:
        data = data[:limit]
        print(f"  🔢 Limiting to first {limit} QA items")

    # Collect all unique paragraphs.
    paragraphs = {}  # title -> full_text
    qa_pairs = []

    for item in data:
        question = item["question"]
        answer = item["answer"]
        supporting_titles = list(set(t for t, _ in item.get("supporting_facts", [])))

        for title, sentences in item.get("context", []):
            full_text = " ".join(sentences)
            paragraphs[(title, full_text)] = True

        qa_pairs.append({
            "id": item.get("_id", ""),
            "question": question,
            "answer": answer,
            "type": item.get("type", ""),
            "supporting_titles": supporting_titles,
        })

    written = _write_articles(paragraphs, output_dir, "2wikimhqa")

    # Write QA pairs.
    data_dir.mkdir(parents=True, exist_ok=True)
    qa_path = data_dir / "qa_pairs.jsonl"
    with open(qa_path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    return {
        "qa_count": len(qa_pairs),
        "paragraph_count": written,
        "unique_titles": len({title for title, _ in paragraphs}),
    }


DATASET_PROCESSORS = {
    "hotpotqa": {
        "processor": process_hotpotqa,
        "input_file": "hotpotqa/hotpot_dev_distractor_v1.json",
    },
    "musique": {
        "processor": process_musique,
        "input_file": "musique/data/musique_ans_v1.0_dev.jsonl",
    },
    "2wikimhqa": {
        "processor": process_2wikimhqa,
        "input_file": "2wikimhqa/data/dev.json",
    },
}


# 【预处理 CLI】解析数据集和题数限制，调对应处理器生成 raw/articles 与 data/qa_pairs；已有同名输出可能被改写。
def main():
    parser = argparse.ArgumentParser(
        description="Pre-process multi-hop QA datasets into Markdown articles."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", "-d", choices=list(DATASET_PROCESSORS.keys()),
                       help="Dataset to process.")
    group.add_argument("--all", action="store_true",
                       help="Process all datasets.")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Maximum number of QA examples to process.")
    parser.add_argument("--datasets-dir", type=str, default=str(DATASETS_DIR),
                        help=f"Datasets directory (default: {DATASETS_DIR}).")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)

    print("=" * 60)
    print("  Multi-hop QA dataset preprocessing")
    print("=" * 60)

    to_process = list(DATASET_PROCESSORS.keys()) if args.all else [args.dataset]

    results = {}
    for ds_name in to_process:
        ds_info = DATASET_PROCESSORS[ds_name]
        input_path = datasets_dir / ds_info["input_file"]

        if not input_path.exists():
            print(f"\n  [skip] {ds_name}: input file not found: {input_path}")
            print(f"         Run: python download_datasets.py --dataset {ds_name}")
            results[ds_name] = "skip (file missing)"
            continue

        print(f"\n  Processing {ds_name}:")
        output_dir = RAW_DIR / ds_name
        data_dir = DATA_DIR / ds_name

        stats = ds_info["processor"](input_path, output_dir, data_dir, limit=args.limit)

        print(f"  Done {ds_name}:")
        print(f"     QA pairs:           {stats['qa_count']}")
        print(f"     Unique paragraphs:  {stats['unique_titles']}")
        print(f"     Articles written:   {stats['paragraph_count']}")
        print(f"     Articles dir:       {output_dir / 'articles'}")
        print(f"     QA file:            {data_dir / 'qa_pairs.jsonl'}")
        results[ds_name] = f"{stats['qa_count']} QA, {stats['paragraph_count']} articles"

    print(f"\n{'=' * 60}")
    print("  Summary")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name:25s}  {status}")


if __name__ == "__main__":
    main()
