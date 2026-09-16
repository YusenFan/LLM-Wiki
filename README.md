# Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki

**Haoliang Ming, Feifei Li, Xiaoqing Wu, Wenhui Que**

This repository
This repository contains the official code release for **LLM-Wiki**, an
agent-native retrieval system that operationalizes the
*Retrieval-as-Reasoning* paradigm. LLM-Wiki compiles documents into
structured Wiki pages with bidirectional links, exposes `wiki_search`,
`wiki_read`, and link-following operations through standard tool-calling
interfaces, and introduces an Error Book for persistent structural and
semantic self-correction. It supports both offline Wiki compilation and
online retrieval / question answering / evaluation.

> **Paper (arXiv):** https://arxiv.org/abs/2605.25480

---

## 中文学习入口

先读 [从原文到 Wiki，再到答案：系统学习指南](docs/system-learning-guide.md)，
按构建、摘要、检索和 QA 的实际调用链理解系统，再查
[213 个函数／方法的逐项索引](docs/function-reference.md)。
源码已补充中文函数注释，区分当前工具流程、确定性校验和保留的旧整页生成流程。

## Repository layout

```
release/
├── README.md
├── LICENSE
├── requirements.txt
├── arxiv.txt
├── configs/
│   ├── page_types.yaml       # default page-type catalog
│   ├── purpose_bench.md      # default purpose template
│   └── wiki-schema.md        # wiki schema specification
├── llm_wiki_bench/
│   ├── __init__.py
│   ├── bench_config.py       # dataset paths, LLM config, wiki directory management
│   ├── llm_client.py         # OpenAI-compatible API client (chat + tool calls)
│   ├── download_datasets.py  # download public dev sets
│   ├── preprocess_bench.py   # raw paragraphs → Markdown articles
│   ├── bench_ingest.py       # tool-driven ingestion entry point (legacy helpers retained)
│   ├── build_agent.py        # document build loop and validated receipts
│   ├── build_summaries.py    # one cross-document summary layer over cited facts
│   ├── summary_store.py      # summary dependencies, provenance and progressive reads
│   ├── wiki_store.py         # immutable sources and atomic fact increments
│   ├── validate_wiki.py      # deterministic repair, backups, integrity audit
│   ├── retrieval_experiment.py # BM25 / entity candidate comparison
│   ├── bench_error_book.py   # error book for self-correction
│   ├── run.py                # offline wiki construction runner
│   ├── wiki_retriever.py     # wiki_search + wiki_read tools
│   ├── wiki_agent.py         # Retrieval-as-Reasoning tool-calling agent
│   ├── run_qa.py             # end-to-end retrieval + answer runner
│   └── evaluate.py           # EM / F1 evaluation
└── examples/
    └── run_hotpotqa.sh       # minimal reproduction example
```

## Requirements

```bash
pip install -r requirements.txt
```

Python ≥ 3.10. The code calls any **OpenAI-compatible** chat-completion API
(OpenAI, Azure OpenAI, vLLM, Ollama, etc.) over HTTP.

## Configure the LLM backend

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"   # or your local server

# Models used by the pipeline (override as needed):
export LLM_PREMIUM_MODEL="gpt-4o"       # strong model — synthesis steps
export LLM_FAST_MODEL="gpt-4o-mini"     # fast model  — analysis steps
```

## Quickstart: build a wiki on HotpotQA

```bash
# Full pipeline (download → preprocess → ingest)
python -m llm_wiki_bench.run --dataset hotpotqa --limit 500

# Or run each stage separately:
python -m llm_wiki_bench.run --dataset hotpotqa --only-download
python -m llm_wiki_bench.run --dataset hotpotqa --only-preprocess --limit 500
python -m llm_wiki_bench.run --dataset hotpotqa --only-ingest
```

The compiled wiki is written to `wiki_output/<dataset>/wiki/`.

## Run retrieval & answer evaluation

Once a wiki has been compiled, the agent can traverse it to answer questions.
The agent composes `wiki_search` and `wiki_read` calls, follows wikilinks,
and checks evidence sufficiency before producing a final answer.

The premium Answer Agent owns both exploration and final answering in one context.
Defaults: 30 evidence/tool calls, no sub-agents, and 5 candidates per search.
For cross-document questions it can progressively reveal summary overviews, claims,
selected fact evidence, and original sources. Precise questions can search details directly.
A final synthesis call can follow exhausted evidence budget. The legacy `--patience`
flag is retained for CLI compatibility; empty searches no longer prematurely stop
exploration across other directories. This changes the original paper protocol.

```bash
# 1. Generate predictions (one JSONL line per question).
python -m llm_wiki_bench.run_qa --dataset hotpotqa --limit 500

# 2. Evaluate EM / F1 (with hop-wise and type-wise breakdowns).
python -m llm_wiki_bench.evaluate \
    --dataset hotpotqa \
    --predictions results/hotpotqa/predictions.jsonl

# Or do both in a single pass:
python -m llm_wiki_bench.run_qa --dataset hotpotqa --limit 500 --evaluate
```

Results (predictions, summary, per-question details) are written under
`results/<dataset>/`.

## Build a wiki on your own corpus

Place one Markdown file per article under `raw/<corpus_name>/articles/`, then:

```python
import sys
sys.path.insert(0, "llm_wiki_bench")

import bench_config as config
import bench_ingest

config.set_dataset("my_corpus")
config.ensure_wiki_dirs()
article_paths = sorted(config.RAW_DIR.glob("*.md"))
bench_ingest.ingest_batch(article_paths, batch_size=3)
```

## Tool-driven architecture and migration

The implementation follows [WIKI_AGENT_PLAN.md](WIKI_AGENT_PLAN.md).
See [schema and tool contracts](configs/wiki-schema.md) and
[acceptance checks](docs/wiki-agent-acceptance.md).

- Full source text is archived by explicit origin identity and immutable content version.
- The builder sees the actual filesystem tree, reads source windows, finds existing
  entities, and submits revision-checked fact increments with verified quotations.
- Old prose and unrelated facts survive updates. Conflicts and superseded claims
  remain inspectable. Directory categories grow from content; sampling is disabled.
- Each document has a validated receipt. Failed/partial builds are retried; old SHA-only
  caches are ignored. `--batch-size` is retained but documents now build independently.
- The premium Answer Agent continues requesting evidence until it can answer or report
  a gap. Final prediction JSONL retains the short answer alongside citations, reasoning,
  evidence gaps, tool trace, timing, calls and per-model token usage.

Additional configuration: `INGEST_TOOL_BUDGET=40` controls builder calls per document.
`SUMMARY_BUILD_LIMIT=20` bounds summary-generation calls after each ingestion batch;
0 disables that phase. `--answer-model` selects the single exploring/answering model
(premium by default). `--retrieval-model` and `--no-subtasks` are deprecated compatibility
flags; CLI QA never enables delegation. `--search-mode bm25|exact_then_bm25` and `--select-pages 5|10|15`
allow controlled comparisons. Optional rerankers are Python callbacks on `WikiRetriever`;
no embedding service is required. Model-provided token counts are recorded; dollar
cost depends on your provider's prices and is not inferred.

Repair existing output with backups, then audit it:

```bash
python -m llm_wiki_bench.validate_wiki \
  --wiki-dir wiki_output/hotpotqa/wiki --repair \
  --output results/hotpotqa/wiki-repair.json

python -m llm_wiki_bench.validate_wiki \
  --wiki-dir wiki_output/hotpotqa/wiki \
  --output results/hotpotqa/wiki-audit.json
```

Repair unwraps whole-page Markdown fences, quotes malformed metadata where the
value is unambiguous, resolves unique exact bare links, and snapshots legacy originals.
Backups live in `wiki/.repair-backups/`. Missing/ambiguous links are reported, never
invented. Legacy prose is still unverified until rebuilt with source-backed facts;
archiving cannot recover text that an earlier pipeline never saved. Rerun preprocessing
from the original dataset to retain previously discarded same-title variants.

Run offline protocol regressions and candidate recall comparisons:

```bash
python -m unittest discover -s tests -v
python -m llm_wiki_bench.retrieval_experiment \
  --wiki-dir wiki_output/hotpotqa/wiki --qa data/hotpotqa/qa_pairs.jsonl \
  --limit 50 --output results/hotpotqa/retrieval-baseline.json
```

Candidate supporting-title coverage is a retrieval proxy, not a source entailment or
answer-quality score. Live model runs are required to select the search algorithm,
Agent topology and budgets on measured answer quality and cost. Exact quotation
validation establishes source identity/location, not that the quotation logically
supports every generated claim; that remains a model/review evaluation requirement.

## Cross-document summaries

Ingestion now compiles one summary layer from current, source-backed facts on related
knowledge pages. Each claim links to its supporting facts and exact originals.
Changed dependencies invalidate summaries until rebuilt; stale summaries cannot be
used as final references. This borrows the iterative QA idea from Ψ-RAG, without
implementing its full embedding-based tree or adding another QA agent.

Inspect eligibility, then build or resume pending groups:

```bash
python -m llm_wiki_bench.build_summaries \
  --wiki-dir wiki_output/hotpotqa/wiki --dry-run
python -m llm_wiki_bench.build_summaries \
  --wiki-dir wiki_output/hotpotqa/wiki --limit 20 \
  --output results/hotpotqa/summary-build.json
```

Uncited legacy prose is ineligible: rebuild it through the normal ingestion entry
point first. The compiler reports pending groups, skipped unrelated pairs, failures,
and model usage. See [design, operation and limits](docs/cross-document-summaries.md).

## Citation

If you find this code useful, please cite:

```bibtex
@misc{ming2026retrievalreasoningselfevolvingagentnative,
      title={Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki},
      author={Haoliang Ming and Feifei Li and Xiaoqing Wu and Wenhui Que},
      year={2026},
      eprint={2605.25480},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.25480},
}
```

## License

Released under the MIT License. See [`LICENSE`](./LICENSE).
