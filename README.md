# Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki

**Haoliang Ming, Feifei Li, Xiaoqing Wu, Wenhui Que**

This repository is based on the official code release for **LLM-Wiki**, an
agent-native retrieval system that operationalizes the
*Retrieval-as-Reasoning* paradigm. LLM-Wiki compiles documents into
structured Wiki pages. This local refactor exposes `wiki_tree`, `wiki_read`
and `source_read` through tool-calling interfaces, with direct article citations
and related-page summaries. The original Error Book module is retained but is
not invoked by the new ingestion pipeline. Offline Wiki compilation and online
retrieval / question answering / evaluation remain separate stages.

> **Paper (arXiv):** https://arxiv.org/abs/2605.25480

---

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
│   ├── bench_ingest.py       # two-step LLM ingestion engine
│   ├── wiki_documents.py    # article archive, citation checks, page rendering
│   ├── build_summaries.py   # related-page grouping + summary generation
│   ├── bench_error_book.py   # retained legacy error-book module
│   ├── run.py                # offline wiki construction runner
│   ├── wiki_retriever.py     # wiki_tree + wiki_read + source_read tools
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
The agent starts with an unranked directory tree, chooses files with `wiki_tree`
and `wiki_read`, follows wikilinks,
and reads article passages before producing a final answer in the same conversation.
`update_evidence_state` records unresolved/supported requirements; `finish_answer`
submits an answer with exact article citations. Rejected submissions return tool
errors so the agent can correct them or retrieve more evidence within budget.
See [the unified QA loop](docs/qa-agent-loop.md) for state and stopping rules,
and [directory navigation](docs/wiki-tree-navigation.md) for tools and filename migration.

The current local refactor uses **article → cited knowledge page → high-level
summary**. It replaces digest generation and its automatic repair pipeline.
See [the code review guide](docs/wiki-agent-implementation.md) for every changed
function, the reasons behind it, and the retired behavior. The exact contracts
are in [the wiki schema](configs/wiki-schema.md).

The tool-call budget is `T_max = 15` by default. State updates, rejected calls,
and answer submission count toward it; the last slot is reserved for submission.
Each turn reports the remaining budget and evidence requirements. Search scoring,
BM25, `wiki_search`, and the old `--patience`/`--select-pages` parameters are removed.
`--retrieval-model` now selects the model for the entire QA loop. `--answer-model`
is a deprecated alias; supplying two different models is rejected. The new evidence-only answer policy differs from the original paper
runner: it does not fill gaps from model knowledge. Old benchmark results must
not be presented as results of this refactor.

For an existing digest-based wiki, build a separate output first. This is a
rebuild from processed articles, not an automatic migration of old factual
claims or citations. These build commands call your configured LLM:

```bash
python -m llm_wiki_bench.bench_ingest --dataset hotpotqa --limit 20 \
  --wiki-dir wiki_output/hotpotqa/article-evidence/wiki

# Inspect grouping without model calls or writes:
python -m llm_wiki_bench.build_summaries \
  --wiki-dir wiki_output/hotpotqa/article-evidence/wiki --dry-run

# Resume/build summaries independently, reusing unchanged successful groups:
python -m llm_wiki_bench.build_summaries \
  --wiki-dir wiki_output/hotpotqa/article-evidence/wiki --limit 10
```

Pass the same `--wiki-dir` to `run_qa` to evaluate that build. `bench_ingest
--limit` counts articles; `run_qa --limit` counts questions. A small article
build is a smoke test, not a complete question benchmark.

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
Each prediction now includes `article_evidence`, `evidence_chain`,
`evidence_status`, and `error`. `citations_validated` means the quotes and
locations match the passages read; it does not certify semantic support.

Offline checks (no LLM calls):

```bash
python -m unittest discover -s tests -v
```

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
