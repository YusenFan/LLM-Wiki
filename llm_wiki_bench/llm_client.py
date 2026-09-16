"""LLM client — thin wrapper around any OpenAI-compatible Chat Completions API.

Reads endpoint and credentials from environment variables, so the same code
works against OpenAI, Azure OpenAI, vLLM, Ollama, etc. without modification:

    OPENAI_API_KEY    — bearer token (use a dummy value for local servers)
    OPENAI_BASE_URL   — base URL, default https://api.openai.com/v1

Models are picked up from `bench_config` (see LLM_MODEL / LLM_PREMIUM_MODEL).
"""

import json
import logging
import os
import re
import time
import requests

try:
    from . import bench_config as config
except ImportError:
    import bench_config as config

_llm_logger = logging.getLogger("ingest.llm")


# 【HTTP 配置】从进程环境读取 OPENAI_BASE_URL 并去尾斜杠；这里只拼端点，不自动加载 .env。
def _api_base() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")


# 【HTTP 配置】从环境读取密钥，构造 JSON 与 Bearer 请求头；缺失密钥时 Authorization 为空。
def _headers() -> dict:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key else "",
    }


# 【文本模型调用】构造 system/user 消息并 POST /chat/completions；可请求 JSON，最多五次尝试，失败最终抛 RuntimeError。
# 返回文本而非已执行动作；无 content 时可能退回提供商 reasoning 字段。
def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    expect_json: bool = False,
    timeout: int = 300,
    model: str | None = None,
    enable_thinking: bool = False,
) -> str:
    """Call an OpenAI-compatible Chat Completions endpoint.

    `enable_thinking` is forwarded to providers that support it (e.g. local vLLM
    deployments of GLM/Qwen). Standard OpenAI ignores the field.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    use_model = model or config.LLM_MODEL
    url = f"{_api_base()}/chat/completions"

    payload: dict = {
        "model": use_model,
        "messages": messages,
        "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
        "max_tokens": max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
    }
    if expect_json:
        payload["response_format"] = {"type": "json_object"}
    if enable_thinking and "api.openai.com" not in _api_base():
        # Pass-through field for backends that support reasoning / thinking mode
        # (vLLM GLM/Qwen). OpenAI rejects unknown arguments with HTTP 400, so
        # it is only sent to non-OpenAI endpoints.
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    _llm_logger.info(
        "LLM call | model=%s | sys=%d chars | user=%d chars | temp=%s | max_tokens=%s",
        use_model, len(system_prompt), len(user_prompt),
        payload["temperature"], payload["max_tokens"],
    )

    MAX_RETRIES = 5
    t_start = time.time()
    last_resp = None

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout)
            last_resp = resp
            resp.raise_for_status()

            if not resp.text or not resp.text.strip():
                raise RuntimeError(f"Empty response (status={resp.status_code})")

            data = resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            # Some providers (GLM, Qwen, DeepSeek) place the answer in
            # `reasoning_content` / `reasoning` when thinking mode is on.
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            if not content.strip() and reasoning.strip():
                content = reasoning

            t_elapsed = time.time() - t_start
            usage = data.get("usage", {})
            _llm_logger.info(
                "LLM done | %.1fs | resp=%d chars | tokens=%s",
                t_elapsed, len(content), usage.get("total_tokens", "?"),
            )
            return content.strip()

        except (requests.RequestException, KeyError, IndexError,
                json.JSONDecodeError, RuntimeError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                _llm_logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
                continue
            try:
                body = last_resp.text[:500] if last_resp is not None else "?"
                status = last_resp.status_code if last_resp is not None else "?"
                _llm_logger.error("LLM final failure: status=%s body=%s", status, body)
            except Exception:
                pass
            raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries: {e}") from e


# 【旧 JSON 调用】先请求 JSON 模式，调用失败后回退普通文本；依次尝试整体 JSON、代码围栏、首左花括号到末右花括号的切片。
# 不是平衡括号解析器，也不保证返回值一定是 dict。
def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float | None = None,
) -> dict:
    """Call the LLM and parse the result as a JSON object.

    Falls back to extracting the first balanced JSON object from the text if
    the provider does not honor `response_format=json_object`.
    """
    try:
        text = call_llm(system_prompt, user_prompt, expect_json=True, model=model, temperature=temperature)
    except RuntimeError:
        text = call_llm(system_prompt, user_prompt, expect_json=False, model=model, temperature=temperature)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise RuntimeError(f"Cannot parse LLM output as JSON:\n{text[:500]}")


# 【当前模型接口】发送完整 messages 与工具 Schema，tool_choice=auto；返回 assistant message 并附 _usage，重试耗尽返回 None。
# 工具调用仍只是提议，调用方负责解析、验证和执行；一次逻辑调用可能发生多次 HTTP 请求。
def call_llm_with_tools(
    messages: list[dict],
    tools: list[dict],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: int = 300,
) -> dict | None:
    """Call a Chat Completions endpoint with OpenAI-style tool calling.

    Returns the raw assistant message (a dict that may contain `tool_calls`),
    or `None` if all retries fail.
    """
    use_model = model or config.LLM_MODEL
    url = f"{_api_base()}/chat/completions"

    payload: dict = {
        "model": use_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": tools,
        "tool_choice": "auto",
    }

    MAX_RETRIES = 5
    last_resp = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=_headers(), json=payload, timeout=timeout)
            last_resp = resp
            resp.raise_for_status()
            if not resp.text or not resp.text.strip():
                raise RuntimeError(f"Empty response (status={resp.status_code})")
            data = resp.json()
            message = data["choices"][0]["message"]
            message["_usage"] = data.get("usage", {})
            return message
        except (requests.RequestException, KeyError, IndexError,
                json.JSONDecodeError, RuntimeError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                _llm_logger.warning(
                    "tool-call LLM failed (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, MAX_RETRIES, e, wait,
                )
                time.sleep(wait)
                continue
            try:
                body = last_resp.text[:500] if last_resp is not None else "?"
                status = last_resp.status_code if last_resp is not None else "?"
                _llm_logger.error("tool-call LLM final failure: status=%s body=%s", status, body)
            except Exception:
                pass
            return None
    return None
