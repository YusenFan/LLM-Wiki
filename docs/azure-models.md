# Azure GLM 5.2 Fast + Qwen3 Embedding

The existing ingestion, summaries and unified QA agent use the Chat Completions
adapter. Azure's `model` value is the deployment name, not necessarily the catalog
model ID. Qwen embeddings rank summary documents only; knowledge-page reads and
evidence validation continue through the existing Wiki agent.

## Deployment readiness

Live checks on 2026-09-19 found:

- Foundry account `simonfans0928-1355-resource` and project
  `simonfans0928-1355` in `Agentic_Framework`, `westus3`.
- The existing deployment `FW-GLM-5.2-Fast`, model version `1`, reports
  `Succeeded`, with pay-per-token `DataZoneStandard` capacity `100`.
- The private `.env.azure` and example configuration now select this deployment
  for ingestion, synthesis, query, and lint models. This is **GLM 5.2 Fast**, not
  the standard GLM 5.2 variant. Source `.env.azure` before starting Python;
  the project's existing `.env` and generic defaults have not been changed.
- Qwen's embedding endpoint remains unconfigured. Managed-compute GPU quota is
  available, but a compatible deployed endpoint has not been verified. GPU quota
  alone does not establish model readiness. No new GPU deployment was created.

The earlier GLM 5.1 path was abandoned because its standard SKU was deprecated
and the subscription had no Fireworks PTU quota. The selected GLM 5.2 Fast
deployment does not require PTU. No API keys were retrieved or stored; inference
uses the signed-in Azure CLI identity.

For GLM, use the existing deployment name and its Foundry `/openai/v1` base URL.
For Qwen, use the deployment's **Consume** example: a direct Azure ML scoring URL,
a managed-deployments route, or a supported unified embeddings route may be used.
The complete embedding request URL is configurable. Do not append `/embeddings`
to an existing `/score` URL. Select `EMBEDDING_PROTOCOL=tei` only for TEI's native
`inputs`/array response, and `openai` for its `input`/`data` response.

References:

- [Fireworks deployment and available offers](https://learn.microsoft.com/en-us/azure/foundry/how-to/fireworks/enable-fireworks-models)
- [Qwen catalog and inference payloads](https://ai.azure.com/catalog/models/qwen-qwen3-embedding-8b)
- [Managed compute routes and pricing model](https://learn.microsoft.com/en-us/azure/foundry/concepts/managed-compute-overview)
- [Foundry SDK overview](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/sdk-overview?pivots=programming-language-python)
- [Qwen query instructions and dimensions](https://huggingface.co/Qwen/Qwen3-Embedding-8B)

## Configuration

Use `examples/azure-models.env.example` as the starting point for a private,
git-ignored `.env.azure`. Replace the placeholder endpoints and deployment names.
The project's existing `.env` is not modified or automatically loaded.

`LLM_AUTH_MODE=azure_cli` and `EMBEDDING_AUTH_MODE=azure_cli` fetch short-lived
tokens from the existing `az login` session. Tokens are cached in process memory
until shortly before expiry; token/key values are not written to project reports.
The identity needs inference access to each deployment. Set each
`*_AZURE_RESOURCE` to the deployment's documented token audience. Foundry and
classic Azure ML can require different audiences and role assignments.

For API keys, use `*_AUTH_MODE=api_key` and privately supply `OPENAI_API_KEY` or
`EMBEDDING_API_KEY`. `*_API_KEY_HEADER=Authorization` sends a bearer key;
`api-key` selects Azure's named header. Chat keys are not inherited by a different
embedding request URL. Endpoint authentication is independent for each client.

LLM-specific controls:

- `LLM_TOOL_MAX_TOKENS` controls the agent's tool-call output budget separately
  from `LLM_MAX_TOKENS`. The example uses 8192 and 16384 respectively; tune only
  after checking actual completions. `finish_reason=length` is a failure.
- `LLM_JSON_MODE=auto` falls back to plain JSON prompting only when the endpoint
  explicitly rejects the JSON format parameter. Auth errors and other bad
  requests do not trigger a second request without JSON mode.
- `LLM_TOKEN_PARAMETER` selects `max_tokens` or `max_completion_tokens` according
  to the endpoint contract. `LLM_REASONING_EFFORT` is sent only if explicitly set.
  vLLM's thinking parameter is never inferred merely from a non-OpenAI hostname.
- Transient network/408/429/5xx failures retry with a bounded attempt count.
  Nonretryable HTTP errors fail immediately. Vendor error bodies are not logged.
- Final content is required for text/JSON calls. Reasoning fields are preserved
  in tool conversations but never substituted for final answer content.

Qwen-specific controls:

- Queries receive `Instruct: <task>\nQuery: <query>`; documents remain unprefixed.
  Specify `EMBEDDING_QUERY_INSTRUCTION` explicitly when using an opaque deployment name.
- Set `EMBEDDING_DIMENSIONS=4096`; malformed or wrong-sized vectors are rejected.
- `EMBEDDING_BATCH_SIZE`, `EMBEDDING_TIMEOUT`, `EMBEDDING_CHUNK_TOKENS`, and
  `EMBEDDING_MAX_INPUT_BYTES` control request size and latency. The tokenizer is
  still an explicitly approximate tiktoken encoding for GLM/Qwen. The example
  adds a conservative 12000-byte UTF-8 cap, without silently dropping document
  tails or allowing TEI to truncate inputs.
- Cache v2 includes the full endpoint, protocol, model, configured version,
  dimensions, query instruction, and exact submitted text. Existing caches remain
  on disk but do not collide with the new configuration. Set the model version
  from the actual deployment; changing weights behind an unchanged deployment
  requires updating this value.

## Bounded smoke test

To validate only the configured LLM while Qwen is not yet deployed:

```bash
source .venv/bin/activate
source .env.azure
python smoke_azure_models.py --scope llm
```

This scope validates chat, JSON, and a two-turn tool call only. A passing LLM
smoke does **not** mean embeddings or end-to-end Wiki QA have passed.

Live result on 2026-09-19: all three LLM checks passed against
`FW-GLM-5.2-Fast` using Azure CLI authentication. Report:
`wiki_output/azure-smoke/20260919T043616271646Z/smoke-result.json`.
No full embedding/Wiki smoke or benchmark was run; work stopped after the LLM
smoke as requested.

Once both endpoints are deployed and the private configuration is filled:

```bash
source .venv/bin/activate
source .env.azure
python smoke_azure_models.py
```

The script writes a new timestamped directory under `wiki_output/azure-smoke/`.
It validates plain chat, JSON, a two-turn tool round trip, 4096-dimensional
embeddings, a basic semantic ordering check, and cache reuse. It then ingests two
synthetic source articles and runs **one** Wiki QA question with citation checks.
It fails if dense retrieval degrades to BM25. It does not run HotpotQA evaluation,
read the benchmark question set, or start 10-/100-question jobs.

`smoke-result.json` reports `passed`, `failed`, or `blocked`. Missing deployment
configuration is `blocked`, not a successful smoke. A model/API/QA failure after
preflight is `failed`. Build and QA artifacts are saved separately when reached.
Synthetic smoke validates integration, not benchmark quality or performance.

Offline protocol tests (no inference calls):

```bash
source .venv/bin/activate
python -m unittest discover -s tests -p test_azure_models.py -v
```

The complete test suite also needs the public cl100k_base and o200k_base tiktoken
vocabularies cached. This setup used `TIKTOKEN_CACHE_DIR=/private/tmp/llm-wiki-tiktoken`.
