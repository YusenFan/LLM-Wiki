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
        self.model = model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        chat_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.base_url = os.environ.get("EMBEDDING_BASE_URL", chat_base).rstrip("/")
        self.api_key = os.environ.get("EMBEDDING_API_KEY", "")
        if not self.api_key and self.base_url == chat_base:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.cache_path = cache_path
        self.stats = {"requests": 0, "input_tokens": 0, "cache_hits": 0, "embedded_texts": 0}
        self.namespace = json.dumps(["summary-v1", self.base_url, self.model])

    def _request(self, texts: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(2):
            self.stats["requests"] += 1
            try:
                response = requests.post(f"{self.base_url}/embeddings", headers=headers,
                                         json={"model": self.model, "input": texts, "encoding_format": "float"},
                                         timeout=45)
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
                data = payload["data"]
                if (len(data) != len(texts) or any(type(row["index"]) is not int for row in data)
                        or sorted(row["index"] for row in data) != list(range(len(texts)))):
                    raise ValueError("invalid embedding response indices")
                vectors = [unit_vector(row["embedding"]) for row in sorted(data, key=lambda r: r["index"])]
                if len({len(v) for v in vectors}) != 1:
                    raise ValueError("inconsistent embedding dimensions")
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
                if vectors[key] is not None:
                    self.stats["cache_hits"] += 1
                else:
                    missing[key] = text
            entries = list(missing.items())
            # Summary index chunks are <= 6000 tokens: 16 inputs remain below request token limits.
            for offset in range(0, len(entries), 16):
                batch = entries[offset:offset + 16]
                fetched = self._request([text for _, text in batch])
                for (key, _), vector in zip(batch, fetched):
                    vectors[key] = vector
                    db.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (key, json.dumps(vector)))
                db.commit()
        result = [vectors[key] for key in keys]
        if len({len(vector) for vector in result}) != 1:
            raise RuntimeError("Embedding dimensions changed; clear the embedding cache or select a versioned model")
        return result
