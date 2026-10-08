"""Summary-only BM25/dense retrieval; rankings navigate, never validate evidence."""
from __future__ import annotations

import math
import hashlib
import json
import re
import sqlite3
import sys
import tempfile
import time
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from embedding_client import unit_vector
from token_budget import TokenBudget, dumps
from wiki_documents import parse_document, resolve_wiki_link
from request_budget import DeadlineExceeded, remaining_seconds


def terms(text: str) -> list[str]:
    # Whole word matching (never substrings in hashes); CJK uses individual characters.
    return re.findall(r"[\u3400-\u9fff]|[^\W_]+", re.sub(r"[\u3400-\u9fff]", r" \g<0> ", text.casefold()))


class BM25:
    """Okapi BM25 with positive IDF, k1=1.5, b=.75 and no field bonuses."""
    def __init__(self, texts: list[str]):
        self.lengths = []
        self.postings = defaultdict(dict)
        for i, text in enumerate(texts):
            words = terms(text)
            self.lengths.append(len(words))
            for word, count in Counter(words).items():
                self.postings[word][i] = count
        self.average = sum(self.lengths) / max(1, len(texts)) or 1

    def rank(self, query: str, limit: int) -> list[tuple[int, float]]:
        scores = defaultdict(float)
        n = len(self.lengths)
        for word in set(terms(query)):
            postings = self.postings.get(word, {})
            idf = math.log(1 + (n - len(postings) + .5) / (len(postings) + .5))
            for i, tf in postings.items():
                scores[i] += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * self.lengths[i] / self.average))
        return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]


def fuse(sparse: list[tuple[int, float]], dense: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """RRF combines ranks, not incomparable BM25/cosine score magnitudes."""
    scores = defaultdict(float)
    for ranking in (sparse, dense):
        for rank, (i, _) in enumerate(ranking, 1):
            scores[i] += 1 / (60 + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


@dataclass
class SummaryDocument:
    path: str
    title: str
    body: str
    members: list[dict]
    search_text: str


class SummaryIndex:
    def __init__(self, pages: dict, *, mode: str, embedder=None, candidate_k: int = 20,
                 cache_dir: Path | None = None):
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("summary mode must be bm25, dense or hybrid")
        if type(candidate_k) is not int or not 1 <= candidate_k <= 200:
            raise ValueError("summary candidate count must be 1-200")
        self.mode, self.embedder, self.candidate_k = mode, embedder, candidate_k
        self.docs = []
        for path, page in sorted(pages.items()):
            if page.layer != "summaries":
                continue
            meta, body = parse_document(page.text)
            members = [{"path": p, "title": pages[p].name} for p in meta.get("members", []) if p in pages]

            def link_title(match):
                target = resolve_wiki_link(match[1], pages)
                return pages[target].name if target in pages else ""

            # Members are added as titles once; frontmatter, URLs and path/hash syntax never score.
            prose = body.split("## Member Pages", 1)[0]
            prose = re.sub(r"\[\[(.*?)\]\]", link_title, prose)
            prose = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", prose)
            prose = re.sub(r"https?://\S+|\b[0-9a-fA-F]{32,}\b", "", prose)
            search_text = prose + "\n" + "\n".join(m["title"] for m in members)
            if page.name not in prose:
                search_text = page.name + "\n" + search_text
            self.docs.append(SummaryDocument(path, page.name, body, members, search_text))
        self.bm25 = BM25([doc.search_text for doc in self.docs])
        self._chunks = None
        self._dense_error = None
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.index_status = "missing" if mode != "bm25" else "disabled"
        settings = [getattr(embedder, key, None) for key in
                    ("model", "namespace", "endpoint", "dimensions", "query_instruction",
                     "chunk_tokens", "max_input_bytes", "tokenizer_model")]
        identity = ["summary-index-v1", settings, [(doc.path, doc.search_text) for doc in self.docs]]
        self.fingerprint = hashlib.sha256(dumps(identity).encode("utf-8")).hexdigest()
        if mode != "bm25":
            self.load_prepared_index()

    def load_prepared_index(self) -> bool:
        """Validate a non-executable binary snapshot against the current corpus and provider."""
        if self.cache_dir is None:
            return False
        manifest_path = self.cache_dir / "summary-index.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("invalid snapshot manifest")
            if manifest.get("schema") != "summary-index-v1" or manifest.get("fingerprint") != self.fingerprint:
                self.index_status = "stale"
                return False
            checksum = manifest.pop("manifest_sha256", None)
            if checksum != hashlib.sha256(dumps(manifest).encode("utf-8")).hexdigest():
                raise ValueError("snapshot metadata checksum mismatch")
            dimensions, owners = manifest["dimensions"], manifest["owners"]
            if (type(dimensions) is not int or dimensions < 1 or not isinstance(owners, list)
                    or not owners or any(type(i) is not int or not 0 <= i < len(self.docs) for i in owners)
                    or set(owners) != set(range(len(self.docs)))
                    or manifest["doc_paths"] != [doc.path for doc in self.docs]
                    or manifest.get("byteorder") not in {"little", "big"}):
                raise ValueError("invalid snapshot metadata")
            expected_dimensions = getattr(self.embedder, "dimensions", None)
            if expected_dimensions is not None and dimensions != expected_dimensions:
                raise ValueError("snapshot dimensions changed")
            data = (self.cache_dir / f"summary-index-{self.fingerprint}.bin").read_bytes()
            values = array("d")
            if len(data) != len(owners) * dimensions * values.itemsize:
                raise ValueError("snapshot size does not match metadata")
            if hashlib.sha256(data).hexdigest() != manifest["vector_sha256"]:
                raise ValueError("snapshot checksum mismatch")
            values.frombytes(data)
            if manifest["byteorder"] != sys.byteorder:
                values.byteswap()
            view = memoryview(values)
            chunks = [(owner, view[i * dimensions:(i + 1) * dimensions]) for i, owner in enumerate(owners)]
            for _, vector in chunks:
                if any(not math.isfinite(x) for x in vector) or abs(sum(x * x for x in vector) - 1) > 1e-6:
                    raise ValueError("snapshot contains invalid vectors")
            self._chunks = chunks
            self.index_status = "loaded"
            return True
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            self.index_status = "invalid"
            return False

    def _save_prepared_index(self) -> None:
        if self.cache_dir is None or not self._chunks:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        values = array("d", (x for _, vector in self._chunks for x in vector))
        data = values.tobytes()
        manifest = {"schema": "summary-index-v1", "fingerprint": self.fingerprint,
                    "dimensions": len(self._chunks[0][1]), "byteorder": sys.byteorder,
                    "owners": [owner for owner, _ in self._chunks],
                    "doc_paths": [doc.path for doc in self.docs],
                    "vector_sha256": hashlib.sha256(data).hexdigest()}
        manifest["manifest_sha256"] = hashlib.sha256(dumps(manifest).encode("utf-8")).hexdigest()
        for target, content in [(self.cache_dir / f"summary-index-{self.fingerprint}.bin", data),
                                (self.cache_dir / "summary-index.json", dumps(manifest).encode("utf-8"))]:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.cache_dir, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(content)
                temporary.replace(target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def prepare_index(self) -> dict:
        """Prepare document vectors without embedding a question."""
        started = time.monotonic()
        if self.mode == "bm25" or not self.docs:
            return {"status": "disabled" if self.mode == "bm25" else "empty",
                    "summary_count": len(self.docs), "chunks": 0, "elapsed_seconds": time.monotonic() - started}
        if self._dense_error:
            raise RuntimeError(self._dense_error)
        try:
            self._prepare_chunks()
        except DeadlineExceeded:
            raise
        except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
            self._dense_error = str(error)
            raise
        return {"status": self.index_status, "fingerprint": self.fingerprint,
                "summary_count": len(self.docs), "chunks": len(self._chunks),
                "elapsed_seconds": time.monotonic() - started}

    def _prepare_chunks(self) -> None:
        remaining_seconds()
        if self.embedder is None:
            raise RuntimeError("No embedding client configured")
        if self._chunks is None:
            tokenizer = TokenBudget(getattr(self.embedder, "tokenizer_model", "text-embedding-3-large"))
            chunk_tokens = getattr(self.embedder, "chunk_tokens", 6000)
            byte_limit = getattr(self.embedder, "max_input_bytes", 0)
            texts, owners = [], []
            for i, doc in enumerate(self.docs):
                remainder = doc.search_text
                while remainder:
                    part = tokenizer.prefix(remainder, chunk_tokens)
                    if byte_limit:
                        # Conservative UTF-8 byte bound in addition to the explicitly approximate tokenizer.
                        part = part.encode("utf-8")[:byte_limit].decode("utf-8", errors="ignore")
                    if not part:
                        raise ValueError("Embedding chunk budget cannot fit the next character")
                    texts.append(part)
                    owners.append(i)
                    remainder = remainder[len(part):]
            vectors = getattr(self.embedder, "embed_documents", self.embedder.embed)(texts)
            if len(vectors) != len(texts):
                raise RuntimeError("Embedding client returned the wrong document count")
            self._chunks = [(i, unit_vector(v)) for i, v in zip(owners, vectors)]
            if len({len(vector) for _, vector in self._chunks}) > 1:
                raise RuntimeError("Document embedding dimensions differ")
            self._save_prepared_index()
            self.index_status = "built"

    def _dense(self, query: str, excluded: set[str]) -> list[tuple[int, float]]:
        self.prepare_index()
        vector = unit_vector(getattr(self.embedder, "embed_queries", self.embedder.embed)([query])[0])
        scores = {}
        for i, document in self._chunks:
            if self.docs[i].path in excluded:
                continue
            if len(document) != len(vector):
                raise RuntimeError("Document and query embedding dimensions differ")
            similarity = sum(a * b for a, b in zip(document, vector))
            scores[i] = max(scores.get(i, -math.inf), similarity)
        return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:self.candidate_k]

    def search(self, query: str, *, tokenizer: TokenBudget, limit: int = 5,
               token_budget: int = 4000, offset: int = 0, exclude_paths: list[str] | None = None) -> dict:
        remaining_seconds()
        if not isinstance(query, str) or not query.strip() or tokenizer.count(query) > 512:
            raise ValueError("query must be nonempty and at most 512 tokens")
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("summary limit must be 1-10")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be nonnegative")
        if type(token_budget) is not int or not 512 <= token_budget <= 16000:
            raise ValueError("summary token budget must be 512-16000")
        if exclude_paths is not None and (not isinstance(exclude_paths, list)
                                         or any(not isinstance(p, str) for p in exclude_paths)):
            raise ValueError("exclude_paths must be a list of summary paths")
        excluded = set(exclude_paths or [])
        sparse = self.bm25.rank(query, len(self.docs)) if self.mode != "dense" else []
        sparse = [(i, score) for i, score in sparse if self.docs[i].path not in excluded][:self.candidate_k]
        dense, warnings = [], []
        effective = self.mode
        if self.docs and self.mode != "bm25":
            try:
                dense = self._dense(query, excluded)
            except DeadlineExceeded:
                raise
            except (RuntimeError, ValueError, OSError, sqlite3.Error) as error:
                self._dense_error = str(error)
                effective = "bm25" if self.mode == "hybrid" else "unavailable"
                warnings.append("Dense retrieval unavailable: " + str(error))
        ranking = fuse(sparse, dense) if self.mode == "hybrid" else (sparse if self.mode == "bm25" else dense)
        payload = {"query": query, "mode": self.mode, "effective_mode": effective,
                   "summary_count": len(self.docs), "candidate_count": len(ranking),
                   "offset": offset, "next_offset": None, "token_budget": token_budget,
                   "tokenizer": tokenizer.name, "warnings": warnings, "results": [],
                   "navigation_only": True,
                   "hint": "Read member knowledge pages with wiki_read and answer directly when sufficient. Use source_read for missing details or verification. Requery missing entities; use wiki_tree for uncovered pages."}
        available = token_budget - tokenizer.count(dumps(payload)) - 64
        selected = ranking[offset:offset + min(limit, max(1, available // 350))]
        for i, score in selected:
            doc = self.docs[i]
            # For long summaries, start at the paragraph matching the unresolved query best.
            start = 0
            if tokenizer.count(doc.body) > available // max(1, len(selected)):
                paragraphs = list(re.finditer(r"\S[^\n]*(?:\n(?!\n)[^\n]+)*", doc.body))
                query_terms = set(terms(query))
                if paragraphs:
                    best = max(paragraphs, key=lambda p: len(query_terms & set(terms(p[0]))))
                    start = best.start()
            row = {"path": doc.path, "title": tokenizer.prefix(doc.title, 80),
                   "rank": offset + len(payload["results"]) + 1,
                   "members": [{"path": m["path"], "title": tokenizer.prefix(m["title"], 40)} for m in doc.members[:10]],
                   "member_count": len(doc.members), "members_truncated": len(doc.members) > 10,
                   "text": doc.body[start:], "start_offset": start, "total_chars": len(doc.body)}
            row = tokenizer.fit_row(row, max(1, available // max(1, len(selected)) - 8))
            payload["results"].append(row)
        consumed = offset + len(payload["results"])
        payload["next_offset"] = consumed if consumed < len(ranking) else None
        if tokenizer.count(dumps(payload)) > token_budget:
            raise ValueError("Query or candidate metadata exceeds summary token budget; use a shorter query or larger budget")
        return payload
