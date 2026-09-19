"""OpenAI-compatible embeddings with incremental, provider/model/text-keyed caching."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from pathlib import Path

import requests

from model_auth import auth_headers


def unit_vector(value) -> list[float]:
    if (not isinstance(value, list) or not value
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError("Embedding must be a nonempty finite numeric vector")
    norm = math.sqrt(sum(x * x for x in value))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embedding has invalid norm")
    return [x / norm for x in value]


class EmbeddingClient:
    def __init__(self, cache_path: Path, model: str | None = None):
        self.model = model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
        chat_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.base_url = os.environ.get("EMBEDDING_BASE_URL", chat_base).rstrip("/")
        self.protocol = os.environ.get("EMBEDDING_PROTOCOL", "openai")
        if self.protocol not in {"openai", "tei"}:
            raise ValueError("EMBEDDING_PROTOCOL must be openai or tei")
        suffix = "embeddings" if self.protocol == "openai" else "embed"
        self.endpoint = os.environ.get("EMBEDDING_ENDPOINT", f"{self.base_url}/{suffix}")
        self.api_key = os.environ.get("EMBEDDING_API_KEY", "")
        if (not self.api_key and self.base_url == chat_base
                and self.endpoint == f"{chat_base}/embeddings"):
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.dimensions = int(os.environ["EMBEDDING_DIMENSIONS"]) if os.environ.get("EMBEDDING_DIMENSIONS") else None
        self.batch_size = int(os.environ.get("EMBEDDING_BATCH_SIZE", "16"))
        self.timeout = int(os.environ.get("EMBEDDING_TIMEOUT", "45"))
        self.chunk_tokens = int(os.environ.get("EMBEDDING_CHUNK_TOKENS", "6000"))
        self.tokenizer_model = os.environ.get("EMBEDDING_TOKENIZER_MODEL", "text-embedding-3-large")
        self.max_input_bytes = int(os.environ.get("EMBEDDING_MAX_INPUT_BYTES", "0"))
        self.query_instruction = os.environ.get("EMBEDDING_QUERY_INSTRUCTION", "")
        if (self.batch_size < 1 or self.timeout < 1 or self.chunk_tokens < 1
                or self.max_input_bytes < 0 or (self.dimensions is not None and self.dimensions < 1)):
            raise ValueError("Embedding sizes, dimensions and timeout must be positive")
        self.cache_path = cache_path
        self.stats = {"requests": 0, "input_tokens": 0, "cache_hits": 0, "embedded_texts": 0}
        self.namespace = json.dumps(["summary-v2", self.endpoint, self.protocol, self.model,
                                     os.environ.get("EMBEDDING_MODEL_VERSION", ""), self.dimensions,
                                     self.query_instruction])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        if self.query_instruction:
            texts = [f"Instruct: {self.query_instruction}\nQuery: {text}" for text in texts]
        return self.embed(texts)

    def _request(self, texts: list[str]) -> list[list[float]]:
        if self.max_input_bytes and any(len(text.encode("utf-8")) > self.max_input_bytes for text in texts):
            raise ValueError("Embedding input exceeds configured byte budget; split documents or shorten query")
        if self.protocol == "openai":
            body = {"model": self.model, "input": texts, "encoding_format": "float"}
        else:
            body = {"inputs": texts, "normalize": True, "truncate": False}
        if self.dimensions is not None:
            body["dimensions"] = self.dimensions
        for attempt in range(2):
            self.stats["requests"] += 1
            try:
                response = requests.post(self.endpoint,
                                         headers=auth_headers("EMBEDDING", self.api_key, self.endpoint),
                                         json=body, timeout=self.timeout)
            except requests.RequestException:
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise RuntimeError("Embedding endpoint could not be reached") from None
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 0:
                    time.sleep(1)
                    continue
            if not response.ok:
                # Do not expose headers, credentials, or vendor response bodies in QA logs.
                raise RuntimeError(f"Embedding endpoint returned HTTP {response.status_code}; check embedding configuration")
            try:
                payload = response.json()
                data = (payload["data"] if self.protocol == "openai" else
                        [{"index": i, "embedding": vector} for i, vector in enumerate(payload)])
                if (len(data) != len(texts) or any(type(row["index"]) is not int for row in data)
                        or sorted(row["index"] for row in data) != list(range(len(texts)))):
                    raise ValueError("invalid embedding response indices")
                vectors = [unit_vector(row["embedding"]) for row in sorted(data, key=lambda r: r["index"])]
                if len({len(v) for v in vectors}) != 1:
                    raise ValueError("inconsistent embedding dimensions")
                if self.dimensions is not None and any(len(v) != self.dimensions for v in vectors):
                    raise ValueError("unexpected embedding dimension")
                if self.protocol == "openai":
                    self.stats["input_tokens"] += int(payload.get("usage", {}).get("prompt_tokens", 0))
                self.stats["embedded_texts"] += len(texts)
                return vectors
            except (ValueError, TypeError, KeyError):
                raise RuntimeError("Embedding endpoint returned malformed vectors") from None
        raise RuntimeError("Embedding request failed")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = [hashlib.sha256((self.namespace + "\n" + text).encode()).hexdigest() for text in texts]
        vectors = {}
        with sqlite3.connect(self.cache_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)")
            missing = {}
            for key, text in zip(keys, texts):
                if key in vectors or key in missing:
                    continue
                row = db.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
                try:
                    vectors[key] = unit_vector(json.loads(row[0])) if row else None
                except (ValueError, TypeError):
                    vectors[key] = None
                if self.dimensions is not None and vectors[key] is not None and len(vectors[key]) != self.dimensions:
                    vectors[key] = None
                if vectors[key] is not None:
                    self.stats["cache_hits"] += 1
                else:
                    missing[key] = text
            entries = list(missing.items())
            for offset in range(0, len(entries), self.batch_size):
                batch = entries[offset:offset + self.batch_size]
                fetched = self._request([text for _, text in batch])
                for (key, _), vector in zip(batch, fetched):
                    vectors[key] = vector
                    db.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (key, json.dumps(vector)))
                db.commit()
        result = [vectors[key] for key in keys]
        if len({len(vector) for vector in result}) != 1:
            raise RuntimeError("Embedding dimensions changed; clear the embedding cache or select a versioned model")
        return result
