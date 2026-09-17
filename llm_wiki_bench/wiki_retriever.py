"""Retrieve current summaries, navigate files, and read exact article evidence."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from build_summaries import current_summaries
from embedding_client import EmbeddingClient
from summary_retrieval import SummaryIndex
from token_budget import TokenBudget, dumps
from wiki_documents import ARTICLE_PREFIX, parse_document, read_article, resolve_wiki_link, wiki_path

_logger = logging.getLogger("llm_wiki.retriever")


# ─── Data structures ──────────────────────────────────────────────────────

@dataclass
class WikiPage:
    """A single compiled Wiki page."""
    name: str                                        # readable title (filename is a fallback)
    dir_name: str                                    # directory under wiki/
    rel_path: str                                    # e.g. "entities/Einstein.md"
    text: str                                        # full markdown (with frontmatter)
    body: str                                        # markdown without frontmatter
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    description: str = ""                            # one-line "> ..." blockquote
    links_to: list[str] = field(default_factory=list)  # [[wikilink]] targets

    @property
    def layer(self) -> str:
        if self.rel_path.startswith(ARTICLE_PREFIX):
            return "articles"
        return "summaries" if self.dir_name == "summaries" else "knowledge"


# ─── Retriever ────────────────────────────────────────────────────────────

class WikiRetriever:
    """Expose directory traversal and explicit page/article reads."""

    def __init__(self, wiki_dir: Path, *, summary_mode: str = "hybrid", summary_limit: int = 5,
                 summary_token_budget: int = 4000, summary_candidates: int = 20,
                 embedding_model: str | None = None, tokenizer_model: str = "gpt-4o", embedder=None):
        if summary_mode not in {"tree", "bm25", "dense", "hybrid"}:
            raise ValueError("summary mode must be tree, bm25, dense or hybrid")
        if type(summary_limit) is not int or not 1 <= summary_limit <= 10:
            raise ValueError("summary limit must be 1-10")
        if type(summary_token_budget) is not int or not 512 <= summary_token_budget <= 16000:
            raise ValueError("summary token budget must be 512-16000")
        if type(summary_candidates) is not int or not summary_limit <= summary_candidates <= 200:
            raise ValueError("summary candidates must be between summary limit and 200")
        self.wiki_dir = Path(wiki_dir)
        self.summary_mode = summary_mode
        self.summary_limit = summary_limit
        self.summary_token_budget = summary_token_budget
        self.summary_candidates = summary_candidates
        self.tokenizer = TokenBudget(tokenizer_model)
        self.embedder = embedder if embedder is not None else EmbeddingClient(
            self.wiki_dir / ".build" / "retrieval-embeddings.sqlite3", model=embedding_model)
        self.summary_index = None
        self.pages: dict[str, WikiPage] = {}
        self.dir_indexes: dict[str, str] = {}        # dir_name → _index.md content
        self._loaded = False

    # ── loading ───────────────────────────────────────────────────────────

    def load(self) -> None:
        """Walk the wiki directory and build all indices (idempotent)."""
        if self._loaded:
            return
        if not self.wiki_dir.exists():
            _logger.warning("wiki dir does not exist: %s", self.wiki_dir)
            self._loaded = True
            return

        summaries = current_summaries(self.wiki_dir)
        for md in sorted(self.wiki_dir.rglob("*.md")):
            rel = md.relative_to(self.wiki_dir)
            rel_str = str(rel)

            if any(part.startswith(".") for part in rel.parts):
                continue
            if ((rel.parts[0] == "sources" and not rel_str.startswith(ARTICLE_PREFIX))
                    or rel_str.startswith("syntheses/")):
                continue
            if rel_str.startswith("summaries/"):
                if md.name == "_index.md":
                    self.dir_indexes["summaries"] = "# Summaries\n" + "\n".join(
                        f"- [[{p[:-3]}]]" for p in sorted(summaries))
                    continue
                if rel_str not in summaries:
                    continue

            if md.name in ("index.md", "overview.md", "log.md"):
                continue

            if md.name == "_index.md":
                dir_name = str(rel.parent) if str(rel.parent) != "." else ""
                if dir_name:
                    try:
                        self.dir_indexes[dir_name] = md.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        pass
                continue

            try:
                wiki_path(self.wiki_dir, rel_str)
                text = md.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            page = self._parse_page(md.stem, rel, text)
            self.pages[rel_str] = page

        self._loaded = True
        _logger.info(
            "loaded wiki: %d pages, %d directory indices",
            len(self.pages), len(self.dir_indexes),
        )

    def summary_search(self, query: str, limit: int | None = None, offset: int = 0,
                       exclude_paths: list[str] | None = None) -> dict:
        """Return a token-bounded batch; agent may reformulate, page or exclude seen summaries."""
        self.load()
        if self.summary_mode == "tree":
            raise ValueError("Summary retrieval is disabled in tree mode; use wiki_tree")
        if self.summary_index is None:
            self.summary_index = SummaryIndex(self.pages, mode=self.summary_mode,
                                              candidate_k=self.summary_candidates, embedder=self.embedder)
        return self.summary_index.search(query, tokenizer=self.tokenizer,
                                         limit=self.summary_limit if limit is None else limit,
                                         token_budget=self.summary_token_budget, offset=offset,
                                         exclude_paths=exclude_paths)

    def initial_navigation(self, question: str) -> dict:
        if self.summary_mode == "tree":
            return {"mode": "tree", "directory_tree": self.wiki_map()}
        return self.summary_search(question)

    @property
    def tool_schemas(self) -> list[dict]:
        return [tool for tool in WIKI_TOOL_SCHEMAS
                if self.summary_mode != "tree" or tool["function"]["name"] != "summary_search"]

    def _parse_page(self, stem: str, rel: Path, text: str) -> WikiPage:
        dir_name = str(rel.parent) if str(rel.parent) != "." else ""

        metadata, body = parse_document(text)
        aliases = metadata.get("aliases", [])
        tags = metadata.get("tags", [])
        aliases = [a for a in aliases if isinstance(a, str)] if isinstance(aliases, list) else []
        tags = [t for t in tags if isinstance(t, str)] if isinstance(tags, list) else []
        description = ""
        heading = next((line[2:].strip() for line in body.splitlines() if line.startswith("# ")), "")
        stem = str(metadata.get("title") or metadata.get("source_title") or heading or stem)
        if re.fullmatch(r"[0-9a-f]{64}", stem):
            stem = "Untitled page"

        for line in body.strip().split("\n"):
            line = line.strip()
            if line.startswith("> ") and not line.startswith("> Source"):
                description = line[2:].strip()
                break

        links_to = re.findall(r"\[\[(.+?)\]\]", body)

        return WikiPage(
            name=stem,
            dir_name=dir_name,
            rel_path=str(rel),
            text=text,
            body=body,
            aliases=aliases,
            tags=tags,
            description=description,
            links_to=links_to,
        )

    # ── directory tree ────────────────────────────────────────────────────

    def tree(self, path: str = "/", depth: int = 2, offset: int = 0, limit: int = 100) -> dict:
        """List QA-visible paths without ranking; directories can be expanded and paginated."""
        self.load()
        if not isinstance(path, str):
            raise ValueError("path must be a wiki-relative directory")
        path = path.strip()
        if path == "/":
            path = ""
        if path and ("\\" in path or PurePosixPath(path).is_absolute()
                     or ".." in PurePosixPath(path).parts or PurePosixPath(path).as_posix() != path):
            raise ValueError("path must be a wiki-relative directory, or /")
        if type(depth) is not int or not 1 <= depth <= 10:
            raise ValueError("depth must be between 1 and 10")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("offset must be nonnegative and limit must be between 1 and 200")
        entries = {}
        for page in self.pages.values():
            entries[page.rel_path] = {"path": page.rel_path, "type": "file", "title": page.name,
                                     "layer": page.layer}
        for directory in self.dir_indexes:
            index_path = f"{directory}/_index.md"
            entries[index_path] = {"path": index_path, "type": "file", "title": f"{directory} index",
                                   "layer": "index"}
        for file_path in list(entries):
            for parent in PurePosixPath(file_path).parents:
                if str(parent) != ".":
                    entries.setdefault(str(parent), {"path": str(parent), "type": "directory"})
        if path and (path not in entries or entries[path]["type"] != "directory"):
            raise ValueError("directory not found; use wiki_tree('/') to list available paths")
        prefix = path + "/" if path else ""
        visible = [item for name, item in sorted(entries.items())
                   if name.startswith(prefix) and name != path
                   and len(name[len(prefix):].split("/")) <= depth]
        selected = visible[offset:offset + limit]
        next_offset = offset + len(selected)
        return {"path": path or "/", "depth": depth, "entries": selected, "total": len(visible),
                "next_offset": next_offset if next_offset < len(visible) else None}

    # ── read ──────────────────────────────────────────────────────────────

    def read(self, paths: list[str], offset: int = 0) -> list[dict]:
        """Batch-read directory indices or page files.

        * `"/"`            → list of available top-level directories
        * `"entities"`     → directory `_index.md` (or page list)
        * `"entities/X.md"` → full page text + metadata
        """
        if not isinstance(paths, list) or not 1 <= len(paths) <= 10 or any(not isinstance(p, str) for p in paths):
            raise ValueError("paths must contain 1-10 Wiki paths")
        if type(offset) is not int or offset < 0 or (offset and len(paths) != 1):
            raise ValueError("offset must be nonnegative; continue only one path at a time")
        self.load()
        out: list[dict] = []
        for raw in paths:
            p = resolve_wiki_link(raw, self.pages)
            if p == "/":
                out.append({"path": "/", "type": "root", "dirs": sorted(self.dir_indexes.keys())})
                continue

            if p.endswith("/_index.md"):
                p = p[:-len("/_index.md")]
            if p.endswith(".md"):
                page = self.pages.get(p)
                if page is None:
                    out.append({"path": p, "type": "error", "error": "not found"})
                    continue
                if page.layer == "articles":
                    out.append({"path": p, "type": "article", "name": page.name,
                                "total_lines": len(page.text.splitlines()),
                                "instruction": "Use source_read(article, start_line, end_line) for evidence."})
                    continue
                out.append({
                    "path": p,
                    "type": "file",
                    "name": self.tokenizer.prefix(page.name, 80),
                    "text": page.body,
                    "meta": {
                        "layer": page.layer,
                    },
                })
                continue

            # directory
            idx = self.dir_indexes.get(p)
            if idx:
                out.append({"path": p, "type": "directory", "name": p, "text": idx})
                continue

            prefix = p + "/"
            pages_in_dir = [pg.name for rel, pg in self.pages.items() if rel.startswith(prefix)]
            if pages_in_dir:
                out.append({"path": p, "type": "directory", "name": p, "text": "\n".join(pages_in_dir)})
            else:
                out.append({"path": p, "type": "error", "error": "not found"})
        per_row = (self.summary_token_budget - 64) // len(out) - 8
        bounded = []
        for item in out:
            if "text" in item:
                text = item["text"]
                if offset > len(text):
                    bounded.append({"path": item["path"], "error": "offset exceeds page length"})
                    continue
                item = {**item, "text": text[offset:], "start_offset": offset, "total_chars": len(text)}
            bounded.append(self.tokenizer.fit_row(item, per_row))
        if self.tokenizer.count(dumps(bounded)) > self.summary_token_budget:
            raise ValueError("Batch metadata exceeds read budget; request fewer paths")
        return bounded

    # ── helpers ───────────────────────────────────────────────────────────

    def source_read(self, article: str, start_line: int = 1, end_line: int | None = None) -> dict:
        """Read bounded article passages and retain the version observed by the agent."""
        self.load()
        article = resolve_wiki_link(article, self.pages)
        if article not in self.pages or self.pages[article].layer != "articles":
            raise ValueError(f"article not found: {article!r}; use a sources/articles/ path from "
                             "a knowledge-page link or wiki_tree(path='sources/articles')")
        if type(start_line) is not int:
            raise ValueError("start_line must be an integer")
        total = len(self.pages[article].text.splitlines())
        if end_line is None:
            end_line = min(total, start_line + 79)
        if type(end_line) is not int or end_line - start_line >= 200:
            raise ValueError("read at most 200 lines per call")
        return read_article(self.wiki_dir, article, start_line, end_line)

    def wiki_map(self) -> str:
        """Seed the agent with an unranked directory tree, with explicit expansion guidance."""
        listing = self.tree(depth=2, limit=200)
        lines = ["# Wiki directory tree", "Directory entries are navigation; read knowledge pages to obtain evidence.",
                 "Use wiki_tree(path, depth, offset) to expand directories or continue a listing."]
        for entry in listing["entries"]:
            indent = "  " * (len(entry["path"].split("/")) - 1)
            if entry["type"] == "directory":
                lines.append(f"{indent}- {entry['path']}/")
            else:
                lines.append(f"{indent}- {entry['path']} — {entry['title']}")
        if listing["next_offset"] is not None:
            lines.append(f"More entries: wiki_tree(path='/', depth=2, offset={listing['next_offset']})")
        return "\n".join(lines)

    # ── tool dispatch (matches paper tool names) ──────────────────────────

    def execute_tool(self, name: str, arguments: dict) -> str:
        """Run a single tool call and return a JSON string."""
        if name == "summary_search":
            return dumps(self.summary_search(arguments.get("query", ""), arguments.get("limit"),
                                             arguments.get("offset", 0), arguments.get("exclude_paths")))
        if name == "wiki_tree":
            return json.dumps(self.tree(arguments.get("path", "/"), arguments.get("depth", 2),
                                        arguments.get("offset", 0), arguments.get("limit", 100)), ensure_ascii=False)
        if name == "wiki_read":
            return dumps(self.read(arguments.get("paths") or arguments.get("dirs") or [],
                                   arguments.get("offset", 0)))

        if name == "source_read":
            return json.dumps(self.source_read(arguments.get("article", ""),
                                               arguments.get("start_line", 1),
                                               arguments.get("end_line")), ensure_ascii=False)

        return json.dumps({"error": f"unknown tool: {name}"})


# ─── OpenAI tool schemas exposed to the agent ─────────────────────────────

WIKI_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "summary_search",
            "description": "Find current summaries for an unresolved entity or relation, using configured BM25/dense/hybrid retrieval. Returns budgeted summary text and member paths, never final evidence. Rephrase the query for another hop, page with next_offset, or exclude already-read summaries. If summaries miss evidence, use wiki_tree and wiki_read directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Specific missing fact/entity; at most 512 tokens"},
                    "limit": {"type": "integer", "description": "1-10 summaries; default configured limit (5)"},
                    "offset": {"type": "integer", "description": "Candidate offset from previous next_offset; same query and exclusions"},
                    "exclude_paths": {"type": "array", "items": {"type": "string"}, "description": "Already-read summary paths; use offset=0 when changing exclusions"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_tree",
            "description": "List the Wiki directory tree without relevance scoring. Choose files by title and path, expand a subdirectory, or use next_offset to see more entries. Lists current readable pages only; no content is read.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Wiki-relative directory such as film or sources/articles; default /"},
                    "depth": {"type": "integer", "description": "Levels below path, 1-10; default 2"},
                    "offset": {"type": "integer", "description": "Pagination offset, default 0"},
                    "limit": {"type": "integer", "description": "Max entries, 1-200; default 100"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_read",
            "description": (
                "Read 1-10 navigation pages within the configured token budget. Follow article links for evidence. "
                "Text may be truncated: continue a single path with next_offset (character offset into page body). "
                "For a directory, prefer wiki_tree pagination."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "offset": {"type": "integer", "description": "Character offset into the page body; use returned next_offset with a single path"},
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of paths. Use '/' for the root, 'entities' for a "
                            "directory index, 'entities/X.md' for a full page."
                        ),
                    }
                },
                "required": ["paths"],
            },
        },
    },
]

WIKI_TOOL_SCHEMAS.append({
    "type": "function",
    "function": {
        "name": "source_read",
        "description": "Optionally read original article lines for missing details, ambiguity, conflicts, or original-source verification. Knowledge-page evidence is sufficient when it explicitly answers the question.",
        "parameters": {
            "type": "object",
            "properties": {
                "article": {"type": "string", "description": "sources/articles/...md path"},
                "start_line": {"type": "integer", "description": "1-based inclusive start (default 1)"},
                "end_line": {"type": "integer", "description": "Inclusive end; at most 200 lines per call"},
            },
            "required": ["article"],
        },
    },
})
