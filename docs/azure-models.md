# Azure GLM 5.2 Fast + text-embedding-3-large

Ingestion, summaries, and QA use the existing GLM Chat Completions adapter.
Summary retrieval uses Azure OpenAI `text-embedding-3-large`. Knowledge-page
reads and evidence validation continue through the existing Wiki agent.

## Configuration

The private, git-ignored `.env.azure` selects `FW-GLM-5.2-Fast` for chat and
`text-embedding-3-large` for embeddings. Source it before starting Python; the
application does not automatically load environment files. The generic `.env`
is a separate configuration.

Use `examples/azure-models.env.example` as the template. Embeddings use the full
Azure deployment URL, including `/embeddings?api-version=2023-05-15`, with
`EMBEDDING_AUTH_MODE=api_key` and `EMBEDDING_API_KEY_HEADER=api-key`.
Store the embedding key only in the private environment file. Chat continues to
use the signed-in Azure CLI identity via `LLM_AUTH_MODE=azure_cli`.
Authentication is independent for each client; a different embedding URL does
not inherit chat credentials.

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

Embedding controls:

- The model and tokenizer default to `text-embedding-3-large`.
- The Azure configuration expects 3072-dimensional vectors; malformed or
  wrong-sized responses fail validation.
- Documents and queries are sent without instruction prefixes.
- The example batches 16 texts, uses a 120-second timeout, and splits summary
  documents into 6000-token chunks. The extra byte cap is disabled.
- Cache keys include endpoint, protocol, model, configured version, dimensions,
  query instruction, and exact submitted text. Existing vectors remain on disk
  but are not reused for a different model or endpoint.

## Validation

On 2026-09-19, the configured embedding endpoint passed a live check: three
texts returned 3072-dimensional vectors, the relevant passage ranked higher
(0.666 versus 0.033 cosine similarity), and two cached documents required no
additional request. This check did not run chat, Wiki ingestion, or QA.

Load the Azure configuration:

```bash
source .venv/bin/activate
source .env.azure
```

Run `python smoke_azure_models.py --scope llm` for chat, JSON, and tool calls only.
Run `python smoke_azure_models.py` for those checks plus 3072-dimensional
embeddings, semantic ordering, cache reuse, a two-article synthetic Wiki build,
and one QA question with citation checks. Full smoke fails if retrieval falls
back to BM25. It does not run a HotpotQA benchmark.

Reports are written to a new timestamped directory under `wiki_output/azure-smoke/`.
Missing configuration is reported as `blocked`; model/API/QA failures after
preflight are `failed`. Synthetic smoke validates integration, not benchmark
quality.

Offline protocol tests:

```bash
python -m unittest discover -s tests -p test_azure_models.py -v
```

Retrieval tests also need cached tiktoken vocabularies. The local cache used for
validation is `/private/tmp/llm-wiki-tiktoken`.
