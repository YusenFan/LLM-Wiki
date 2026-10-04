"""Read and validate existing navigation summaries without generating pages."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from wiki_documents import knowledge_pages, parse_document, related_pages, wiki_path


def summary_groups(pages: dict[str, str]) -> list[tuple[str, ...]]:
    """Deduplicate sets and remove strict subsets, without merging overlapping groups."""
    candidates = set()
    for path, text in pages.items():
        members = frozenset({path} | (set(related_pages(text)) & pages.keys()))
        candidates.add(members)  # Uncovered singletons survive strict-subset removal.
    return sorted(tuple(sorted(group)) for group in candidates
                  if not any(group < other for other in candidates))


def group_fingerprint(members: tuple[str, ...], pages: dict[str, str]) -> str:
    content = json.dumps([(p, pages[p]) for p in members], ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def current_summaries(root: Path, pages: dict[str, str] | None = None) -> dict[str, str]:
    """Expose only summaries for current groups and current member contents."""
    pages = knowledge_pages(root) if pages is None else pages
    result = {}
    groups = {members: group_fingerprint(members, pages) for members in summary_groups(pages)}
    seen = set()
    # Read old hash filenames too; choose a readable name if both exist.
    paths = sorted((root / "summaries").glob("*.md"),
                   key=lambda p: (bool(re.fullmatch(r"[0-9a-f]{64}", p.stem)), p.name))
    for path in paths:
        relative = path.relative_to(root).as_posix()
        wiki_path(root, relative)
        text = path.read_text(encoding="utf-8")
        metadata, _ = parse_document(text)
        raw_members = metadata.get("members")
        if not isinstance(raw_members, list) or any(not isinstance(p, str) for p in raw_members):
            continue
        members = tuple(raw_members)
        if (metadata.get("schema") == "related-summary-v1"
                and members in groups and members not in seen
                and metadata.get("fingerprint") == groups[members]):
            result[relative] = text
            seen.add(members)
    return result
