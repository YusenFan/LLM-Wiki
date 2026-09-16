"""Shared document contracts: articles are evidence; other pages are navigation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath

import yaml


ARTICLE_PREFIX = "sources/articles/"
RESERVED_DIRS = {"sources", "summaries", "syntheses"}


def wiki_path(root: Path, value: str) -> Path:
    """Reject ambiguous or escaping paths before reading or writing model output."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("expected a wiki-relative Markdown path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"invalid wiki path: {value}")
    target = root / value
    if target.suffix != ".md" or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"invalid wiki path: {value}")
    return target


def parse_document(text: str) -> tuple[dict, str]:
    """Use YAML rather than comma splitting so tags and titles survive quoting."""
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}, text
    metadata = yaml.safe_load(match.group(1)) or {}
    if not isinstance(metadata, dict):
        raise ValueError("frontmatter must be an object")
    return metadata, text[match.end():]


def render_document(metadata: dict, body: str) -> str:
    return "---\n" + yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False) + "---\n\n" + body.rstrip() + "\n"


def write_document(path: Path, text: str) -> None:
    """Replace each complete file atomically; never expose half a page to readers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def archive_article(root: Path, source: Path) -> dict:
    """Keep processed article bytes unchanged; the content hash identifies its version."""
    data = source.read_bytes()
    text = data.decode("utf-8")
    metadata, _ = parse_document(text)
    version = hashlib.sha256(data).hexdigest()
    relative = f"{ARTICLE_PREFIX}{version}.md"
    target = wiki_path(root, relative)
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError(f"article archive does not match its hash: {relative}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return {"article": relative, "version": version,
            "title": str(metadata.get("title") or source.stem), "text": text}


def article_reference(root: Path, citation: dict) -> dict:
    """Resolve an article link from disk; generation does not need to reproduce its text."""
    if not isinstance(citation, dict):
        raise ValueError("article reference must be an object")
    article = citation.get("article", "")
    path = wiki_path(root, article)
    if not article.startswith(ARTICLE_PREFIX) or path.name == "_index.md":
        raise ValueError("evidence must come from sources/articles/")
    data = path.read_bytes()
    version = hashlib.sha256(data).hexdigest()
    if re.fullmatch(r"[0-9a-f]{64}", path.stem) and path.stem != version:
        raise ValueError("article content no longer matches its archived version")
    text = data.decode("utf-8")
    if not text.strip():
        raise ValueError("article is empty")
    if citation.get("version", version) != version:
        raise ValueError("citation version does not match article")
    return {"article": article, "version": version, "text": text}


def read_article(root: Path, article: str, start_line: int, end_line: int) -> dict:
    """Return an exact, inclusive line range; no model-written source text is accepted."""
    source = article_reference(root, {"article": article})
    lines = source["text"].splitlines()
    if type(start_line) is not int or type(end_line) is not int:
        raise ValueError("article line numbers must be integers")
    if not 1 <= start_line <= end_line <= len(lines):
        raise ValueError(f"invalid article range: {start_line}-{end_line} / {len(lines)}")
    return {"article": article, "version": source["version"],
            "start_line": start_line, "end_line": end_line,
            "quote": "\n".join(lines[start_line - 1:end_line]), "total_lines": len(lines)}


def validate_citation(root: Path, citation: dict) -> dict:
    """A valid range/quote proves location, not whether it entails a generated claim."""
    if not isinstance(citation, dict):
        raise ValueError("citation must be an object")
    actual = read_article(root, citation.get("article", ""),
                          citation.get("start_line"), citation.get("end_line"))
    if not actual["quote"].strip() or citation.get("quote") != actual["quote"]:
        raise ValueError("citation quote must exactly match the article lines")
    if citation.get("version", actual["version"]) != actual["version"]:
        raise ValueError("citation version does not match article")
    return actual


def citation_link(citation: dict) -> str:
    if "start_line" not in citation:
        return f"[[{citation['article'][:-3]}]]"
    return (f"[[{citation['article'][:-3]}#L{citation['start_line']}"
            f"-L{citation['end_line']}]]")


def knowledge_pages(root: Path) -> dict[str, str]:
    """Only knowledge directories participate in grouping, never summaries or articles."""
    pages = {}
    for path in sorted(root.rglob("*.md")):
        relative = path.relative_to(root).as_posix()
        parts = PurePosixPath(relative).parts
        if (len(parts) < 2 or parts[0] in RESERVED_DIRS
                or any(p.startswith(".") for p in parts) or path.name == "_index.md"):
            continue
        wiki_path(root, relative)
        pages[relative] = path.read_text(encoding="utf-8")
    return pages


def related_pages(text: str) -> dict[str, str]:
    """Read only the explicit Related Pages section, including its relationship reasons."""
    _, body = parse_document(text)
    section = re.search(r"^## Related pages\s*\n(.*?)(?=^## |\Z)",
                        body, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if not section:
        return {}
    links = {}
    for line in section.group(1).splitlines():
        match = re.match(r"\s*-\s*\[\[([^\]#|]+)(?:\|[^\]]+)?\]\]\s*[—–-]\s*(\S.*)", line)
        if match:
            path, reason = match.groups()
            links[path if path.endswith(".md") else path + ".md"] = reason.strip()
    return links


def text_field(value: object, name: str) -> str:
    """Restrict rendered fields to one line so model text cannot create new sections."""
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value or "[[" in value:
        raise ValueError(f"{name} must be nonempty single-line text without wikilinks")
    return value.strip()


def string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return list(dict.fromkeys(text_field(item, name) for item in value))


def render_knowledge(root: Path, proposal: dict, available: set[str]) -> tuple[str, set[str]]:
    """Python owns Markdown structure and link validation; the model owns the proposed facts."""
    title = text_field(proposal.get("title"), "title")
    description = text_field(proposal.get("description"), "description")
    facts = proposal.get("facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError("knowledge pages require cited facts")
    lines = [f"# {title}", "", f"> {description}", "", "## Core Facts"]
    sources = {}
    for fact in facts:
        if not isinstance(fact, dict) or not isinstance(fact.get("citations"), list) or not fact["citations"]:
            raise ValueError("every fact needs article citations")
        claim = text_field(fact.get("text"), "fact")
        # Store source links, not model-written quotes or a forced passage selection.
        citations = [article_reference(root, c) for c in fact["citations"]]
        lines.append(f"- {claim} " + " ".join(citation_link(c) for c in citations))
        for citation in citations:
            sources[citation_link(citation)] = citation
    links = proposal.get("related_pages", [])
    if not isinstance(links, list):
        raise ValueError("related_pages must be a list")
    lines.extend(["", "## Related Pages"])
    seen = set()
    for link in links:
        if not isinstance(link, dict):
            raise ValueError("related page must be an object")
        target = link.get("path")
        if not isinstance(target, str) or target not in available or target == proposal["path"]:
            raise ValueError(f"invalid related knowledge page: {target}")
        reason = text_field(link.get("reason"), "relationship reason")
        if target not in seen:
            lines.append(f"- [[{target[:-3]}]] — {reason}")
            seen.add(target)
    lines.extend(["", "## Related Sources"])
    lines.extend(f"- {link}" for link in sources)
    metadata = {"type": proposal["path"].split("/")[0], "schema": "article-evidence-v1",
                "aliases": string_list(proposal.get("aliases", []), "aliases"),
                "tags": string_list(proposal.get("tags", []), "tags")}
    return render_document(metadata, "\n".join(lines)), {c["article"] for c in sources.values()}
