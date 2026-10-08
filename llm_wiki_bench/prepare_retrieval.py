"""Prepare a reusable summary index after building a HotpotQA or 2Wiki Wiki."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bench_config as config
from wiki_retriever import WikiRetriever


def prepare(wiki_dir: Path, *, embedding_model: str | None = None) -> dict:
    if not Path(wiki_dir).is_dir():
        raise ValueError(f"Wiki directory does not exist: {wiki_dir}")
    return WikiRetriever(Path(wiki_dir), embedding_model=embedding_model).prepare_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "2wikimhqa"])
    parser.add_argument("--wiki-dir", type=Path)
    parser.add_argument("--embedding-model")
    args = parser.parse_args()
    config.set_dataset(args.dataset, wiki_dir=args.wiki_dir)
    try:
        report = prepare(config.WIKI_DIR, embedding_model=args.embedding_model)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Index preparation failed: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
