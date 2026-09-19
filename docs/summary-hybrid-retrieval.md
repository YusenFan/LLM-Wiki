# Summary 层混合检索

默认流程：问题 → summary BM25 + dense → RRF → 预算内摘要 → 同一个 QA agent 阅读知识页 → 信息充分时提交，否则补查或按需 source_read 原文。目录树按需展开；初始化不再把全局文件列表交给模型。

## 调用链

`run_qa.main()` 配置 `WikiRetriever` → `WikiAgent.retrieve()` → `initial_navigation(question)` → `summary_search()` → `SummaryIndex.search()`。

`WikiRetriever.load()` 使用 `current_summaries()` 排除成员或内容已变化的摘要。该 retriever 是一次 QA 进程的语料快照；修改 Wiki 后重启 QA 才会重新加载。只有 summary 进入搜索索引，knowledge/article 不会暗中加入候选。索引内容为摘要标题、正文和成员页面标题，移除 frontmatter、链接地址和长哈希；没有路径命中或字段加权奖励。

BM25 使用完整词匹配、k1=1.5、b=0.75 和正 IDF；中文按字切分。Dense 调用 OpenAI-compatible `/embeddings`，按 cosine 排序。长摘要分成不超过 6000 个 cl100k_base tokens 的片段，各片段均参与 embedding，以最大片段相似度作为摘要相似度；不静默丢掉尾部。默认每路取 20 个文档候选，RRF 使用 `sum(1 / (60 + rank))` 融合，按路径确定性打破同分。BM25-only/dense-only 模式使用本路排名。

返回排名不是置信度，更不代表事实支持关系。RRF 只用于选择阅读入口；最终答案引用 wiki_read 实际读过的知识页，或按需 source_read 读过的原文。摘要本身不作为最终证据。

`build_summaries` 为没有进入任何多页分组的孤立知识页生成 singleton summary。Python 复制该页的导航正文、移除来源链接并添加 Member Pages 入口，不调用 LLM、不制造关系。singleton 与多页摘要都按成员内容 fingerprint 判断是否有效；新关系使 singleton 成为严格子集后，旧 singleton 不再进入 current summaries。完整成功构建时所有知识页均被覆盖；结果中的 `covered_pages` 和 `uncovered_pages` 可检测失败、限额或只生成 singleton 导致的缺口。已有 Wiki 可用 `--singletons-only` 离线补齐孤立页，命令见 [增量知识与证据 ID](incremental-knowledge-evidence.md)。

## 阅读预算和补查

- 默认每批最多 5 个 summary；可配置 1–10。单批 `summary_search` 和 `wiki_read` 的完整 JSON（含元数据）最多 4000 tokens，可配置 512–16000。预算小时会减少摘要数。
- 知识页阅读结果中的证据 ID、版本和证据文本也计入完整 JSON 预算。截断后只为实际交付的文本登记证据；未读部分必须续读。
- tokenizer 优先使用 QA 模型对应的 tiktoken encoding，未知模型使用 cl100k_base，结果会明确记录 tokenizer 名称。不同供应商 tokenizer 的精确计数可能不同。首次使用 tiktoken 需要下载公开词表。
- 短摘要返回完整正文；长摘要从与查询有最多词项重叠的段落开始，返回截断窗口。这一步是确定性片段选择，不增加 LLM 调用。
- 返回 `start_offset`、`total_chars`、`truncated` 和 `next_offset`。文本 offset 是 Markdown body 的字符位置，不包含 YAML frontmatter；`wiki_read(paths=[path], offset=next_offset)` 可继续读取。`start_offset > 0` 表示开头也有省略，可从 offset=0 阅读。
- 预览显示最多 10 个成员路径；`members_truncated=true` 时通过 wiki_read 继续读摘要中的完整 Member Pages。成员路径本身不是证据。
- `summary_search(query, limit, offset, exclude_paths)` 支持新查询、候选列表分页和排除已读摘要。其外层 next_offset 是候选序号；结果项内 next_offset 是正文字符位置。改变查询或排除集合后从候选 offset=0 开始。
- 初始候选不是访问白名单。对比题应分别覆盖各实体；桥接题可用新发现的实体重新查询。没有 summary 或摘要缺少事实时，仍可通过 wiki_tree → wiki_read → source_read 寻找证据。

这是每次导航返回的预算，不是整个会话 context 上限。工具历史仍保留；wiki_tree 有条目上限，source_read 仍采用行数上限，没有实现历史压缩或原文 token 分页。

## Embedding 配置与缓存

默认 `text-embedding-3-large`，可通过 `--embedding-model` 或 `EMBEDDING_MODEL` 设置。`EMBEDDING_BASE_URL` 默认跟随 `OPENAI_BASE_URL`；在同一 endpoint 下 `EMBEDDING_API_KEY` 默认复用 `OPENAI_API_KEY`。若使用不同 endpoint，需显式设置 `EMBEDDING_API_KEY`，不会自动转发 chat 凭证。

SQLite 缓存在 Wiki 的 `.build/retrieval-embeddings.sqlite3`。内部 key 包括索引版本、endpoint、模型名和实际输入文本，hash 仅用于缓存身份，既不作为公开文件名，也不参与匹配。摘要文本变化重新 embedding，未变化片段和相同查询复用缓存；旧缓存项可以保留但不会因此使失效摘要重新进入索引。该版本本地扫描向量，不需要向量数据库。

API 请求有限重试；无效返回不写入缓存。Hybrid 中 dense 失败明确记录 warnings 和 effective_mode=bm25，本进程后续补查继续 BM25，避免反复请求故障服务。Dense-only 失败明确记录 unavailable，agent 可继续目录浏览；恢复服务后重启 QA。不会把 BM25 降级运行标作成功的 hybrid 实验。

## 运行和记录

```bash
source .venv/bin/activate
set -a
source .env
set +a
python -m llm_wiki_bench.run_qa \
  --dataset hotpotqa \
  --wiki-dir wiki_output/hotpotqa/test-one/wiki \
  --limit 1 \
  --summary-mode hybrid \
  --summary-candidates 20 \
  --summary-limit 5 \
  --summary-token-budget 4000 \
  --output wiki_output/hotpotqa/test-one/qa-hybrid.jsonl \
  --verbose
```

若本机还没有 test-one Wiki，先运行已有 `python build_test_one.py`；这是调用模型的独立构建步骤。

用 `--summary-mode bm25`、`dense` 或 `tree` 做对照。初始检索不占模型工具槽，后续检索、导航、状态更新和提交共享 `--t-max`；因此不能仅凭工具调用数比较不同入口的总成本。

JSONL 单独记录 initial_navigation、summary_searches、embedding_usage（实际请求数、输入 tokens、命中缓存条数、embedding 文本数）及检索配置。首次问题承担索引创建成本，后续问题或重跑可复用缓存。候选摘要不计入 article_evidence；该字段只来自实际 source_read。

测试：`.venv/bin/python -m unittest discover -s tests -q`。新增测试覆盖词边界、语义候选、RRF、分页补查、失效摘要、token 上限、Unicode 无损续读、embedding 分块/缓存/错误，以及单 agent 工具预算。检索效果需在固定语料、题集、模型、预算下比较候选证据覆盖、实际 source_read 覆盖、EM/F1 和费用；单题只验证流程。

参考接口：[OpenAI Embeddings](https://developers.openai.com/api/reference/resources/embeddings/methods/create)、[tiktoken](https://github.com/openai/tiktoken)；混合检索思路参考 [Ψ-RAG §3.2](https://arxiv.org/pdf/2605.00529)。
