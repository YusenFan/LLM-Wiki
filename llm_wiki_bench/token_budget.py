"""Bound navigation payloads with an explicit tokenizer and resumable text windows."""
from __future__ import annotations

import json

import tiktoken


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class TokenBudget:
    def __init__(self, model: str = "gpt-4o"):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Explicitly reported in payloads; custom providers may tokenize differently.
            self.encoding = tiktoken.get_encoding("cl100k_base")

    @property
    def name(self) -> str:
        return self.encoding.name

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    def prefix(self, text: str, limit: int) -> str:
        """Cut at a character boundary, so paging never corrupts Unicode."""
        if self.count(text) <= limit:
            return text
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if self.count(text[:mid]) <= limit:
                low = mid
            else:
                high = mid - 1
        return text[:low]

    def fit_row(self, row: dict, limit: int, decorate=None) -> dict:
        """Trim text only, keeping a character offset into the original page body."""
        row = dict(row)
        text = row.get("text", "")
        start = row.get("start_offset", 0)
        total = row.get("total_chars", start + len(text))
        if "text" not in row:
            if self.count(dumps(row)) <= limit:
                return row
            return {"path": row.get("path"), "error": "Read this path individually; metadata exceeds the batch budget."}
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            trial = {**row, "text": text[:mid], "truncated": start > 0 or start + mid < total,
                     "next_offset": start + mid if start + mid < total else None}
            if decorate:
                trial = decorate(trial)
            if self.count(dumps(trial)) <= limit:
                low = mid
            else:
                high = mid - 1
        row.update(text=text[:low], truncated=start > 0 or start + low < total,
                   next_offset=start + low if start + low < total else None)
        if not low and text:
            return {"path": row.get("path"), "error": "Read this path individually; metadata exceeds the batch budget."}
        return decorate(row) if decorate else row
