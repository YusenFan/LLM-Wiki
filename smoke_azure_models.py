"""Bounded live smoke: chat, JSON, tools, Qwen vectors/cache, and one synthetic Wiki QA.

No deployments are created and no benchmark datasets are read or evaluated.
Source examples/azure-models.env.example after replacing its endpoint placeholders.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def preflight(scope="full"):
    from urllib.parse import urlparse
    required = ["OPENAI_BASE_URL", "LLM_FAST_MODEL", "LLM_PREMIUM_MODEL"]
    if scope == "full":
        required += ["EMBEDDING_ENDPOINT", "EMBEDDING_MODEL"]
    for name in required:
        value = os.environ.get(name, "")
        require(value and not any(marker in value for marker in ("YOUR-", "REPLACE-", "<", ">")),
                f"Configure {name} from a deployed Azure model before the live smoke test")
    for name in (["OPENAI_BASE_URL", "EMBEDDING_ENDPOINT"] if scope == "full" else ["OPENAI_BASE_URL"]):
        endpoint = urlparse(os.environ[name])
        require(endpoint.scheme == "https" and (endpoint.hostname or "").endswith((".azure.com", ".azure.net")),
                f"{name} must be an Azure HTTPS endpoint")
    auth = [("LLM", "OPENAI_API_KEY")]
    if scope == "full":
        auth.append(("EMBEDDING", "EMBEDDING_API_KEY"))
    for prefix, key in auth:
        mode = os.environ.get(f"{prefix}_AUTH_MODE", "api_key")
        require(mode in {"azure_cli", "api_key"}, f"Invalid {prefix}_AUTH_MODE")
        require(mode == "azure_cli" or bool(os.environ.get(key)), f"Configure {prefix} authentication")
    if scope == "full":
        require(os.environ.get("EMBEDDING_QUERY_INSTRUCTION"), "Configure Qwen's query instruction explicitly")
        require(os.environ.get("EMBEDDING_DIMENSIONS") == "4096", "Smoke expects full 4096-dimensional Qwen embeddings")


def run_checks(root: Path, record, scope="full"):
    import llm_wiki_bench  # Adds sibling modules to sys.path.
    import bench_config as config
    from llm_client import call_llm, call_llm_json, call_llm_with_tools
    from embedding_client import EmbeddingClient
    from bench_ingest import ingest_batch
    from wiki_retriever import WikiRetriever
    from wiki_agent import WikiAgent

    model = config.LLM_PREMIUM_MODEL
    started = time.monotonic()
    require(call_llm("Reply with exactly OK.", "Connectivity check.", model=model, temperature=0).strip() == "OK",
            "Plain chat did not return OK")
    record("chat", started)

    started = time.monotonic()
    require(call_llm_json('Return a JSON object with ok=true and value=7.', 'JSON compatibility test.', model=model)
            == {"ok": True, "value": 7}, "Unexpected JSON response")
    record("json", started)

    started = time.monotonic()
    tool = {"type": "function", "function": {"name": "lookup_smoke_value",
            "description": "Get the test result. Call exactly once with name=smoke.",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}}
    messages = [{"role": "user", "content": "Call lookup_smoke_value with name=smoke. After receiving its result, reply with exactly that value and no explanation."}]
    first = call_llm_with_tools(messages, [tool], model=model)
    require(first and len(first.get("tool_calls", [])) == 1, "Model did not issue exactly one tool call")
    first.pop("_usage", None)
    tc = first["tool_calls"][0]
    require(tc["function"]["name"] == "lookup_smoke_value"
            and json.loads(tc["function"]["arguments"]) == {"name": "smoke"}, "Invalid tool arguments")
    messages.extend([first, {"role": "tool", "tool_call_id": tc["id"], "content": '{"value":"SMOKE_731"}'}])
    second = call_llm_with_tools(messages, [tool], model=model)
    require(second and not second.get("tool_calls") and (second.get("content") or "").strip() == "SMOKE_731",
            "Tool result did not round-trip correctly")
    record("tool_round_trip", started)

    if scope == "llm":
        return

    started = time.monotonic()
    embedder = EmbeddingClient(root / "embedding-smoke.sqlite3")
    docs = ["Solar panels convert sunlight into electricity.", "Whales swim in the ocean."]
    vectors = embedder.embed_documents(docs)
    query = embedder.embed_queries(["How can sunlight generate electricity?"])[0]
    require(len(vectors) == 2 and all(len(v) == 4096 for v in vectors + [query]), "Embedding dimension mismatch")
    scores = [sum(a * b for a, b in zip(query, doc)) for doc in vectors]
    require(scores[0] > scores[1], "Relevant passage did not outrank unrelated passage")
    requests_before = embedder.stats["requests"]
    cached = embedder.embed_documents(docs)
    require(all(abs(a - b) < 1e-12 for first, second in zip(vectors, cached) for a, b in zip(first, second)),
            "Cached vectors changed")
    require(embedder.stats["requests"] == requests_before, "Cache reuse made another API request")
    record("embeddings_and_cache", started, scores=scores, usage=embedder.stats.copy())

    started = time.monotonic()
    raw = root / "raw"
    raw.mkdir()
    fixtures = {
        "lena.md": "# Lena River\nLena River founded Aster Research in 2011. Aster Research is a research organization.\n",
        "aster.md": "# Aster Research\nAster Research was founded by Lena River in 2011. Its headquarters are in the city of Valmere.\n",
    }
    for name, body in fixtures.items():
        (raw / name).write_text(body, encoding="utf-8")
    config.set_dataset("hotpotqa", wiki_dir=root / "wiki")
    config.ensure_wiki_dirs()
    build = ingest_batch([raw / name for name in fixtures], batch_size=2)
    (root / "build-result.json").write_text(json.dumps(build, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    require(build["success"] == 2 and not build["failed"] and not build["summaries"]["failed"], "Wiki build failed")
    require(not build["summaries"].get("uncovered_pages"), "Wiki has uncovered knowledge pages")
    record("wiki_build", started)

    started = time.monotonic()
    retriever = WikiRetriever(root / "wiki", summary_mode="hybrid", tokenizer_model=model)
    result = WikiAgent(retriever, call_llm_with_tools=call_llm_with_tools, model=model, t_max=10).retrieve(
        "In which city is Aster Research, founded by Lena River, headquartered?")
    (root / "qa-result.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    require(result.initial_navigation.get("effective_mode") == "hybrid"
            and result.initial_navigation.get("results") and not result.initial_navigation.get("warnings"),
            "Initial dense retrieval was unavailable or empty")
    require(all(s.get("effective_mode") == "hybrid" and not s.get("warnings") for s in result.summary_searches),
            "QA silently degraded to BM25")
    require(result.stop_reason == "submitted" and result.answer["prediction"].strip().casefold() == "valmere",
            "QA did not submit the expected answer")
    require(result.answer.get("evidence_status") == "citations_validated", "QA citations were not validated")
    record("wiki_qa", started, llm_calls=result.llm_calls, embedding_usage=result.embedding_usage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scope", choices=["full", "llm"], default="full",
                        help="llm tests only chat/JSON/tools; it does not validate embeddings or Wiki QA")
    args = parser.parse_args()
    root = args.output_dir or Path("wiki_output/azure-smoke") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": args.scope, "checks": [], "models": {
        "llm": os.environ.get("LLM_PREMIUM_MODEL"), "embedding": os.environ.get("EMBEDDING_MODEL")}}
    report_path = root / "smoke-result.json"

    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def record(name, started, **details):
        report["checks"].append({"name": name, "status": "passed", "seconds": round(time.monotonic() - started, 2), **details})
        save()
        print(f"PASS {name}", flush=True)

    ready = False
    try:
        preflight(args.scope)
        ready = True
        run_checks(root, record, args.scope)
        report["status"] = "passed"
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as error:
        report["status"] = "failed" if ready else "blocked"
        report["error"] = str(error)
    finally:
        save()
        print(f"{report['status'].upper()}: {report_path}", flush=True)
        if report.get("error"):
            print(report["error"], flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
