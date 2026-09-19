"""Chat Completions for OpenAI-compatible and Azure Foundry deployments.

OPENAI_BASE_URL includes /openai/v1 for Foundry; model is the deployment name.
Use LLM_AUTH_MODE=azure_cli for short-lived tokens without storing an API key.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import requests

try:
    from . import bench_config as config
    from .model_auth import auth_headers
except ImportError:
    import bench_config as config
    from model_auth import auth_headers

_llm_logger = logging.getLogger("ingest.llm")


class ModelCallError(RuntimeError):
    """Sanitized inference failure, optionally carrying an HTTP status."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class JSONModeUnsupported(ModelCallError):
    pass


def _api_base() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


def _headers() -> dict:
    return auth_headers("LLM", os.environ.get("OPENAI_API_KEY", ""), _api_base())


def _payload(messages, model, temperature, max_tokens) -> dict:
    token_field = os.environ.get("LLM_TOKEN_PARAMETER", "max_tokens")
    if token_field not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("LLM_TOKEN_PARAMETER must be max_tokens or max_completion_tokens")
    if max_tokens < 1:
        raise ValueError("LLM output token budget must be positive")
    payload = {"model": model or config.LLM_MODEL, "messages": messages,
               "temperature": temperature, token_field: max_tokens}
    effort = os.environ.get("LLM_REASONING_EFFORT")
    if effort:
        payload["reasoning_effort"] = effort
    return payload


def _post(payload: dict, timeout: int) -> tuple[dict, dict]:
    attempts = int(os.environ.get("LLM_MAX_ATTEMPTS", "5"))
    if not 1 <= attempts <= 10:
        raise ValueError("LLM_MAX_ATTEMPTS must be between 1 and 10")
    for attempt in range(attempts):
        delay = min(2 ** (attempt + 1), 30)
        try:
            response = requests.post(f"{_api_base()}/chat/completions", headers=_headers(),
                                     json=payload, timeout=timeout)
        except requests.RequestException:
            error = ModelCallError("LLM endpoint could not be reached")
        else:
            status = response.status_code
            if not 200 <= status < 300:
                # Inspect known capability errors only; never log vendor bodies or tokens.
                body = response.text.lower()
                if (status in {400, 422} and "response_format" in payload
                        and ("response_format" in body or "json_object" in body)
                        and any(word in body for word in ("unsupported", "not supported", "unknown", "unrecognized"))):
                    raise JSONModeUnsupported("Deployment does not support JSON response_format", status)
                error = ModelCallError(f"LLM endpoint returned HTTP {status}", status)
                if status not in {408, 429} and status < 500:
                    raise error
                try:
                    delay = min(60, max(0, float(response.headers.get("Retry-After", delay))))
                except (TypeError, ValueError):
                    pass
            else:
                try:
                    data = response.json()
                    choice = data["choices"][0]
                    message = choice["message"]
                    if not isinstance(message, dict):
                        raise ValueError("invalid message")
                except (ValueError, TypeError, KeyError, IndexError):
                    raise ModelCallError("LLM endpoint returned a malformed completion") from None
                if choice.get("finish_reason") == "length":
                    raise ModelCallError("LLM output was truncated; increase the output token budget")
                if choice.get("finish_reason") == "content_filter":
                    raise ModelCallError("LLM output was filtered")
                return message, data.get("usage", {})
        if attempt + 1 == attempts:
            raise error
        _llm_logger.warning("%s; retrying in %.1fs (%d/%d)", error, delay, attempt + 1, attempts)
        time.sleep(delay)
    raise ModelCallError("LLM request failed")


def call_llm(system_prompt: str, user_prompt: str, temperature: float | None = None,
             max_tokens: int | None = None, expect_json: bool = False, timeout: int = 300,
             model: str | None = None, enable_thinking: bool = False) -> str:
    payload = _payload(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model, config.LLM_TEMPERATURE if temperature is None else temperature,
        config.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
    )
    if expect_json:
        payload["response_format"] = {"type": "json_object"}
    if enable_thinking:
        if os.environ.get("LLM_PROVIDER") != "vllm":
            raise ValueError("enable_thinking requires LLM_PROVIDER=vllm; configure deployment reasoning separately")
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    started = time.monotonic()
    message, usage = _post(payload, timeout)
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ModelCallError("LLM returned no final content (reasoning is not a final answer)")
    _llm_logger.info("LLM done | model=%s | %.1fs | tokens=%s", payload["model"],
                     time.monotonic() - started, usage.get("total_tokens", "?"))
    return content.strip()


def call_llm_json(system_prompt: str, user_prompt: str, model: str | None = None,
                  temperature: float | None = None) -> dict:
    mode = os.environ.get("LLM_JSON_MODE", "auto")
    if mode not in {"auto", "json_object", "prompt"}:
        raise ValueError("LLM_JSON_MODE must be auto, json_object or prompt")
    try:
        text = call_llm(system_prompt, user_prompt, expect_json=mode != "prompt",
                        model=model, temperature=temperature)
    except JSONModeUnsupported:
        if mode != "auto":
            raise
        text = call_llm(system_prompt, user_prompt, expect_json=False, model=model, temperature=temperature)
    candidates = [text]
    fence = re.search(r"\x60\x60\x60json\s*(.*?)\s*\x60\x60\x60", text, re.DOTALL)
    if fence:
        candidates.append(fence[1])
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    raise ModelCallError("LLM output is not a JSON object")


def call_llm_with_tools(messages: list[dict], tools: list[dict], model: str | None = None,
                        temperature: float = 0.0, max_tokens: int | None = None,
                        timeout: int = 300) -> dict | None:
    budget = int(os.environ.get("LLM_TOOL_MAX_TOKENS", "2048")) if max_tokens is None else max_tokens
    payload = _payload(messages, model, temperature, budget)
    payload.update(tools=tools, tool_choice="auto")
    try:
        message, usage = _post(payload, timeout)
        if not message.get("tool_calls") and not (message.get("content") or "").strip():
            raise ModelCallError("LLM returned neither final content nor tool calls")
        # Preserve provider reasoning fields for multi-turn protocols; never use them as answers.
        return {**message, "_usage": usage}
    except ModelCallError as error:
        _llm_logger.error("Tool-call LLM failed: %s", error)
        return None
