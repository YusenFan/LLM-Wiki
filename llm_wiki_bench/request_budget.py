"""Cooperative per-question deadlines and request telemetry for synchronous clients."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import math
import time


class DeadlineExceeded(RuntimeError):
    """The question has no time left for another request or retry."""


_deadline = ContextVar("qa_request_deadline", default=None)
_stats = ContextVar("qa_request_stats", default=None)


@contextmanager
def request_budget(seconds: float | None = None):
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("question time budget must be positive and finite")
    deadline = time.monotonic() + seconds if seconds is not None else None
    parent = _deadline.get()
    if parent is not None:
        deadline = min(parent, deadline) if deadline is not None else parent
    stats = {}
    deadline_token, stats_token = _deadline.set(deadline), _stats.set(stats)
    try:
        yield stats
    finally:
        _stats.reset(stats_token)
        _deadline.reset(deadline_token)


def remaining_seconds() -> float | None:
    deadline = _deadline.get()
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise DeadlineExceeded("Question time budget exhausted")
    return remaining


def request_timeout(configured: float) -> float:
    remaining = remaining_seconds()
    return min(configured, remaining) if remaining is not None else configured


def record_request(service: str, elapsed: float) -> None:
    stats = _stats.get()
    if stats is not None:
        stats[service + "_attempts"] = stats.get(service + "_attempts", 0) + 1
        key = service + "_request_seconds"
        stats[key] = stats.get(key, 0.0) + elapsed


def retry_sleep(service: str, delay: float) -> None:
    remaining = remaining_seconds()
    if remaining is not None and delay >= remaining:
        raise DeadlineExceeded("Retry wait would exhaust the question time budget")
    stats = _stats.get()
    if stats is not None:
        key = service + "_retry_wait_seconds"
        stats[key] = stats.get(key, 0.0) + delay
    time.sleep(delay)
    remaining_seconds()


def stats_snapshot() -> dict:
    return dict(_stats.get() or {})
