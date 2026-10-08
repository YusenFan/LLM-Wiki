#!/usr/bin/env python3
"""Download public multi-hop QA evaluation datasets.

Supported datasets:
- HotpotQA (distractor setting, dev set)
- MuSiQue-Ans (dev set)
- 2WikiMultiHopQA (dev set)

Usage:
    python download_datasets.py                     # download all
    python download_datasets.py --dataset hotpotqa
    python download_datasets.py --dataset musique
    python download_datasets.py --dataset 2wikimhqa
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

DATASETS_DIR = Path(__file__).parent / "datasets"


# 【数据下载】已有目标文件时复用，否则用 HTTP 下载 HotpotQA dev JSON；成功返回文件路径，失败返回 None。
def download_hotpotqa(output_dir: Path):
    """Download the HotpotQA distractor dev set.

    Source: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
    """
    import urllib.request

    url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    out_path = output_dir / "hotpot_dev_distractor_v1.json"

    if out_path.exists():
        print(f"  ✅ HotpotQA dev already exists: {out_path}")
        return out_path

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ⬇️  Downloading HotpotQA dev set...")
    print(f"     URL: {url}")

    try:
        urllib.request.urlretrieve(url, str(out_path))
        # Validate.
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  ✅ HotpotQA dev: {len(data)} samples -> {out_path}")
        return out_path
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        if out_path.exists():
            out_path.unlink()
        return None


# 【数据下载】尝试获取 MuSiQue dev 数据并保存到目标目录；已有文件复用，失败打印手动获取提示并返回 None。
MUSIQUE_URLS = (
    "https://drive.usercontent.google.com/download?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&export=download&confirm=t",
)
TWOWIKI_URLS = (
    "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1",
    "https://www.dropbox.com/s/npidmtadreo6df2/data.zip?dl=1",
)


def _validate_musique(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            required = {"id", "question", "answer", "paragraphs"}
            if not isinstance(item, dict) or not required.issubset(item):
                raise ValueError(f"invalid MuSiQue record at line {line_number}")
            if not isinstance(item["paragraphs"], list):
                raise ValueError(f"invalid MuSiQue paragraphs at line {line_number}")
            count += 1
    if not count:
        raise ValueError("MuSiQue dev file is empty")
    return count


def _validate_2wikimhqa(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list) or not data:
        raise ValueError("2WikiMultiHopQA dev file must be a non-empty JSON array")
    required = {"_id", "question", "answer", "context"}
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"invalid 2WikiMultiHopQA record at index {index}")
        if not isinstance(item["context"], list):
            raise ValueError(f"invalid 2WikiMultiHopQA context at index {index}")
    return len(data)


def _download_archive_member(label: str, urls: tuple[str, ...], member_suffixes: tuple[str, ...],
                             out_path: Path, validator) -> Path | None:
    """Download one validated member from a publisher archive without extracting the archive."""
    if out_path.exists():
        try:
            count = validator(out_path)
            print(f"  ✅ {label} dev already verified: {count} samples -> {out_path}")
            return out_path
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"  ⚠️ Existing {label} file is invalid; downloading again: {exc}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ⬇️  Downloading {label} dev set...")
    for url in urls:
        archive_path = extracted_path = None
        try:
            print(f"     URL: {url}", flush=True)
            with requests.get(url, stream=True, timeout=(20, 120), allow_redirects=True) as response:
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(dir=out_path.parent, prefix=".dataset-",
                                                 suffix=".zip", delete=False) as stream:
                    archive_path = Path(stream.name)
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            with zipfile.ZipFile(archive_path) as archive:
                names = [name for name in archive.namelist() if not name.endswith("/")]
                member = None
                for suffix in member_suffixes:
                    matches = [name for name in names if name.endswith(suffix)]
                    if matches:
                        member = min(matches, key=len)
                        break
                if member is None:
                    raise ValueError(f"archive does not contain any of: {member_suffixes}")
                with archive.open(member) as source, tempfile.NamedTemporaryFile(
                        dir=out_path.parent, prefix=".dataset-", suffix=".part", delete=False) as target:
                    extracted_path = Path(target.name)
                    shutil.copyfileobj(source, target)
            count = validator(extracted_path)
            extracted_path.replace(out_path)
            print(f"  ✅ {label} dev: {count} samples -> {out_path}")
            return out_path
        except (requests.RequestException, OSError, ValueError, json.JSONDecodeError,
                zipfile.BadZipFile) as exc:
            print(f"  ⚠️ Download source failed: {exc}")
        finally:
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)
            if extracted_path is not None:
                extracted_path.unlink(missing_ok=True)
    print(f"  ❌ All {label} download sources failed; rerun to retry.")
    return None


def download_musique(output_dir: Path):
    """Download and validate the official MuSiQue-Ans dev split."""
    return _download_archive_member(
        "MuSiQue", MUSIQUE_URLS,
        ("data/musique_ans_v1.0_dev.jsonl", "musique_ans_v1.0_dev.jsonl"),
        output_dir / "data" / "musique_ans_v1.0_dev.jsonl", _validate_musique,
    )


def download_2wikimhqa(output_dir: Path):
    """Download and validate the official 2WikiMultiHopQA dev split."""
    return _download_archive_member(
        "2WikiMultiHopQA", TWOWIKI_URLS,
        ("data_ids/dev.json", "data/dev.json", "wikimultihop/dev.json", "dev.json"),
        output_dir / "data" / "dev.json", _validate_2wikimhqa,
    )


DATASET_DOWNLOADERS = {
    "hotpotqa": download_hotpotqa,
    "musique": download_musique,
    "2wikimhqa": download_2wikimhqa,
}


# 【下载 CLI】指定 --dataset 时下载单个数据集，省略时下载全部；只获取数据，不预处理也不构建 wiki。
def main():
    parser = argparse.ArgumentParser(description="Download multi-hop QA evaluation datasets")
    parser.add_argument("--dataset", "-d", choices=list(DATASET_DOWNLOADERS.keys()),
                        help="Dataset to download (omit to download all)")
    parser.add_argument("--output-dir", "-o", type=str, default=str(DATASETS_DIR),
                        help=f"Output directory (default: {DATASETS_DIR})")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Multi-hop QA dataset download")
    print("=" * 60)

    if args.dataset:
        datasets_to_download = {args.dataset: DATASET_DOWNLOADERS[args.dataset]}
    else:
        datasets_to_download = DATASET_DOWNLOADERS

    results = {}
    for name, downloader in datasets_to_download.items():
        print(f"\n  Fetching {name}:")
        ds_dir = output_dir / name
        result = downloader(ds_dir)
        results[name] = "ok" if result else "fail"

    print(f"\n{'=' * 60}")
    print("  Download results")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name:15s}  {status}")


if __name__ == "__main__":
    main()
