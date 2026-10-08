# 统一 QA 循环

当前入口为 `run_qa.main()` → `WikiAgent.retrieve()`。同一个模型在同一段对话中浏览目录、阅读、维护证据状态并提交答案。旧的独立 `_answer()` 模型调用已删除。

## 执行流程

1. 2Wiki/HotpotQA 的 CLI 默认使用 `latency` profile：启动时准备或加载文档向量索引；题目中存在唯一标题/别名匹配时直接导航，否则执行 summary BM25/dense/RRF 检索。预算内最多预读 2 个知识页，再启动模型对话。`--qa-profile baseline` 保留原有摘要入口与逐步阅读策略；BM25/dense 默认各取 20 个候选，摘要上限 5 个、4000 tokens。`--summary-mode tree` 默认关闭预读和自适应摘要检索。
2. 每轮发送剩余工具调用数、当前 requirements、是否仅允许提交。
3. Agent 使用 `retrieve_evidence` 合并缺口导航与知识页阅读，也可使用 `summary_search`、`wiki_tree`、`wiki_read`、`source_read` 获取证据；使用 `update_evidence_state` 记录缺口。比较题需覆盖两方实体，桥接题按已确认的中间实体补查。没有摘要或摘要遗漏的页面仍可直接导航。
4. Agent 调用 `finish_answer`；Python 检查引用及已登记需求的覆盖。失败以工具错误返回同一对话，可继续补查或修正。
5. 成功提交后输出 `RetrievalResult.answer`；未能提交时输出 `unknown` 并保留停止原因。

普通文本回复不终止任务。没有新增 agent 或强制单独的 LLM 筛选阶段。知识页信息充分时直接提交；缺少细节、存在歧义或冲突、用户要求原文核验时，按需调用 `source_read`。

## 状态与提交协议

requirements 每项包含 `id`、`question`、`status`（`unresolved` 或 `supported`）和 `evidence_ids`。状态以 ID 合并更新；省略某一项不会删除它。更新先整体校验，坏更新不覆盖旧状态。`finish_answer` 也可携带更新，减少控制调用开销；有效更新即使答案提交失败也会保留。

支持状态必须附有本题实际读过的 `evidence_ids`；答案 evidence_chain 中的每一项包含 `requirement_id`、`claim` 和 `evidence_ids`。模型不再手写路径、引文或行号。Python 用本题的证据表生成引用：知识页为 `page`、`page_version`、字符起止位置和 `quote`；原文为 `article`、`version`、行范围和 `quote`。同一答案可混用两类 ID，非连续事实使用多个 ID。事实答案要求：

- 至少登记一个需求，并且所有登记需求均为 supported。
- 每个 ID 必须存在于本题实际返回内容的证据表中；搜索结果、目录、summary 和关系说明不产生可引用 ID。
- 知识页引用必须逐字匹配 `wiki_read` 实际返回片段中的非空文本；截断后尚未读到的文本不可引用。
- 原文引用必须来自 `source_read` 实际读过的版本、行范围和完整引文。
- evidence_chain 覆盖所有登记需求 ID。

`unknown` 不要求这些条件，输出为空 evidence_chain，保留未解决需求用于诊断。

这些是结构和出处校验。模型仍负责正确拆解问题、识别全部必要关系、判断引文是否支持结论；Python 不证明语义蕴含或问题分解完整性。摘要和目录仍仅用于导航；已读知识页可直接支撑答案，无需强制读取原文。知识页上的来源链接不代表已经核验过原文。

## 预算与停止

`--t-max` 默认为 15，统计导航、状态更新、失败工具调用和答案提交；最后一个槽仅允许 `finish_answer`。非法地占用最后一个槽也会消耗预算，返回 unknown。一个响应内的多个调用按顺序执行，超过预算或成功提交后的调用不执行，但仍返回对应 tool 消息。

初始化导航是独立的 bootstrap，不占模型工具槽；其结果和 embedding 开销单独记录。预读的一批 `wiki_read` 占一个槽，记录 `origin=bootstrap`；只有实际交付的片段登记证据 ID。`retrieve_evidence` 的导航和读取各占一个槽，记录 `call_cost=2`；无可读页面或只剩一个非提交槽时仅导航，成本为 1。失败的读取也消耗读取槽。普通工具仍占一个槽，`retrieval_steps` 为操作成本之和，不一定等于 `tool_calls` 数量。非工具文本最多允许 `t_max + 2` 个模型回合。停止原因为 `submitted`、`budget_exhausted`、`turn_limit`、`model_error` 或 `deadline_exceeded`。

自适应检索按照 unresolved requirements 调整摘要 limit 和返回预算；无新增候选时扩大下一批，最高不超过配置的摘要 token 上限。默认排除已经展示过的摘要，候选分页沿用该查询起始时的排除集合，避免 offset 跳过候选。`navigation_progress` 记录新增候选/证据数量；这些数量和排名不是语义置信度，不自动判断题目已被完整解决。已读页面的未读部分仍通过 `wiki_read(next_offset)` 获取。

`source_read` 省略结束行时最多读 80 行，显式范围仍最多 200 行；结束行超过 EOF 时返回实际结束行及 `range_clamped`，非法起始行仍被拒绝。部分读取返回 `next_start_line`。引用验证继续使用实际读取范围。

LLM 与 embedding 使用分别持久化的 HTTP Session。latency profile 默认单题 120 秒，可用 `--question-time-budget 0` 关闭。这个预算在模型回合、请求和重试之间检查，并限制每次 HTTP timeout；它是同步客户端的协作式截止时间，不强制中断持续返回数据的响应或已经执行中的本地代码。重试等待将耗尽剩余时间时不再重试。模型输出截断保留具体失败原因与输出预算，不提交截断结果。

## 索引准备与配置

构建结束后可用 `run --prepare-retrieval-index`，已有 Wiki 可单独执行：

```bash
python -m llm_wiki_bench.prepare_retrieval \
  --dataset hotpotqa --wiki-dir wiki_output/hotpotqa/first-500/wiki
python -m llm_wiki_bench.prepare_retrieval \
  --dataset 2wikimhqa --wiki-dir wiki_output/2wikimhqa/test-100/wiki
```

索引存于 Wiki 的 `.build/summary-index.json` 和带 fingerprint 的二进制向量文件。fingerprint 包括当前摘要检索文本、成员标题、embedding namespace、分块配置和索引版本；校验文件 checksum、维度及单位向量。失效或损坏时重新准备，未变文本的向量继续复用 SQLite embedding 缓存。文件不使用 pickle。首次准备缺失向量仍需要 embedding 服务；离线预建将这部分工作移出在线题目路径。

```bash
python -m llm_wiki_bench.run_qa \
  --dataset 2wikimhqa --wiki-dir wiki_output/2wikimhqa/test-100/wiki \
  --qa-profile latency --prefetch-pages 2 \
  --read-token-budget 4000 --question-time-budget 120 --evaluate
```

`--prefetch-pages 0`、`--no-adaptive-retrieval`、`--no-prepare-index` 支持单项消融；`--qa-profile baseline` 默认关闭这三项及单题时间预算。baseline 仍包含读取边界修复、连接复用和索引缓存支持。`--read-token-budget` 独立于摘要预算；baseline 省略此参数时沿用摘要预算。此次保留完整对话历史，没有加入上下文压缩。

## 输出与兼容

JSONL 保留既有字段，并增加 `timings`、`llm_turns`、`network_stats`、`navigation_progress`；`error` 保存结构化失败原因。每个 LLM 回合记录耗时、用量及请求统计，工具记录耗时和操作成本。历史名称 `retrieval_llm_calls` 和 `retrieval_usage_by_model` 覆盖整个 QA 对话；embedding 单独统计。

输出旁的 `predictions.run.json` 记录启动耗时、索引准备耗时及 embedding 开销、QA 实际运行时间和完成情况；不能只比较移除了首题索引准备的单题耗时。评估增加单题 P50/P95，包含 unknown 和失败题，固定 Wiki、模型、题集和操作预算比较 EM/F1、unknown、tokens 与工具错误率。

`knowledge_evidence` 单独保存实际返回的知识页片段、版本及字符起始位置；`article_evidence` 仍仅记录实际读取的原文。`evidence_snapshots` 保存 ID 到精确引用的映射；最终 evidence_chain 同时保存模型选择的 ID 和 Python 生成的 citations。只有知识页、没有原文存档的 Wiki 也允许执行 QA。离线校验历史预测仍可使用旧 citations 格式，在线 agent 只接受 evidence_ids。

使用 `--retrieval-model` 选择整段对话的模型；默认仍为 premium 配置，缺省回退 LLM_MODEL。`--answer-model` 作为旧参数别名保留；两参数指定不同模型会明确报错。

目录导航和摘要文件名已更新，详见 [目录树导航](wiki-tree-navigation.md)。摘要分组保留去重和严格子集删除规则，并为未被多页分组覆盖的孤立知识页保留 singleton 组。原文版本和评估分母保持原有规则。实际 benchmark 对比前需固定题集、语料、模型和总预算，并处理范围内缺失预测的计分。

## 文件与验证

- `llm_wiki_bench/wiki_retriever.py`：摘要入口、目录树、预算内页面阅读和原文工具。
- `llm_wiki_bench/summary_retrieval.py`：BM25、dense 分块召回、RRF 和候选分页。
- `llm_wiki_bench/embedding_client.py`：HTTP embeddings、按内容和模型隔离的 SQLite 缓存。
- `llm_wiki_bench/prepare_retrieval.py`：独立离线索引准备入口。
- `llm_wiki_bench/request_budget.py`：单题时间预算与请求/重试计时。
- `llm_wiki_bench/token_budget.py`：token 计数、完整字符边界截断与续读。
- `llm_wiki_bench/wiki_agent.py`：循环、预算提示、状态合并、答案提交分支。
- `llm_wiki_bench/qa_contract.py`：工具 schemas 与共享引用校验。
- `llm_wiki_bench/run_qa.py`：统一入口、模型参数和结果持久化。
- `tests/test_tree_navigation.py`：树展开、分页、可读标题、摘要重名处理与旧文件迁移。
- `tests/test_qa_loop.py`：缺口拒绝后补查、引用修复、完整需求覆盖、预算边界、多调用响应、停止行为和 runner 输出。
- `tests/test_qa_latency.py`：2Wiki/HotpotQA profile 对照、比较题预读、四跳补查、组合成本、索引失效与时间预算。
- `tests/test_article_summary_workflow.py`：构建 → 摘要 → 原文 → 同循环提交的集成验证。

离线验证：`.venv/bin/python -m unittest discover -s tests -q`。这些测试使用脚本化模型响应，不代表真实模型 QA 准确率或成本改善。
