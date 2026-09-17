# 统一 QA 循环

当前入口为 `run_qa.main()` → `WikiAgent.retrieve()`。同一个模型在同一段对话中浏览目录、阅读、维护证据状态并提交答案。旧的独立 `_answer()` 模型调用已删除。

## 执行流程

1. 按原始问题执行一次 summary 检索，默认 BM25/dense 各取 20 个候选，RRF 融合后在 4000-token 预算内提供最多 5 个摘要；不预先发送全局文件列表。`--summary-mode tree` 保留目录入口对照。
2. 每轮发送剩余工具调用数、当前 requirements、是否仅允许提交。
3. Agent 使用 `summary_search` 针对缺口补查，使用 `wiki_tree`、`wiki_read`、`source_read` 获取证据；使用 `update_evidence_state` 记录缺口。没有摘要或摘要遗漏的页面仍可直接导航。
4. Agent 调用 `finish_answer`；Python 检查引用及已登记需求的覆盖。失败以工具错误返回同一对话，可继续补查或修正。
5. 成功提交后输出 `RetrievalResult.answer`；未能提交时输出 `unknown` 并保留停止原因。

普通文本回复不终止任务。没有新增 agent 或强制单独的 LLM 筛选阶段。知识页信息充分时直接提交；缺少细节、存在歧义或冲突、用户要求原文核验时，按需调用 `source_read`。

## 状态与提交协议

requirements 每项包含 `id`、`question`、`status`（`unresolved` 或 `supported`）和 `citations`。状态以 ID 合并更新；省略某一项不会删除它。更新先整体校验，坏更新不覆盖旧状态。`finish_answer` 也可携带更新，减少控制调用开销；有效更新即使答案提交失败也会保留。

支持状态必须附有已读证据引用。知识页引用包含 `page` 和 `quote`；原文引用包含 `article`、`version`、`start_line`、`end_line`、`quote`，同一答案可混用两种引用。答案 evidence_chain 中的每一项还包含 `requirement_id` 和 `claim`。事实答案要求：

- 至少登记一个需求，并且所有登记需求均为 supported。
- 知识页引用必须逐字匹配 `wiki_read` 实际返回片段中的非空文本；截断后尚未读到的文本不可引用。
- 原文引用必须来自 `source_read` 实际读过的版本、行范围和完整引文。
- evidence_chain 覆盖所有登记需求 ID。

`unknown` 不要求这些条件，输出为空 evidence_chain，保留未解决需求用于诊断。

这些是结构和出处校验。模型仍负责正确拆解问题、识别全部必要关系、判断引文是否支持结论；Python 不证明语义蕴含或问题分解完整性。摘要和目录仍仅用于导航；已读知识页可直接支撑答案，无需强制读取原文。知识页上的来源链接不代表已经核验过原文。

## 预算与停止

`--t-max` 默认为 15，统计导航、状态更新、失败工具调用和答案提交；最后一个槽仅允许 `finish_answer`。非法地占用最后一个槽也会消耗预算，返回 unknown。一个响应内的多个调用按顺序执行，超过预算或成功提交后的调用不执行，但仍返回对应 tool 消息。

初始化检索与原来的初始目录一样，是独立的 bootstrap，不占模型工具槽；它的结果和 embedding 开销单独记录。后续 `summary_search` 与其他工具一样计入 `t_max`。旧 `--patience`、`--select-pages` 参数未恢复。非工具文本不会结束任务，但最多允许 `t_max + 2` 个模型回合，避免无限提醒。停止原因为 `submitted`、`budget_exhausted`、`turn_limit` 或 `model_error`。`submitted` 也可表示明确提交 unknown。

每轮提示剩余预算，知识页证据充分时直接提交，只对未解决事实补查。没有实现重复结果检测或自适应预算。

## 输出与兼容

JSONL 保留既有 prediction、evidence_chain、article_evidence、evidence_requirements、evidence_gaps、stop_reason、tool_calls 等字段，并记录 `initial_navigation`、`summary_searches`、`embedding_usage` 和 `retrieval_config`。历史名称 `retrieval_llm_calls` 和 `retrieval_usage_by_model` 覆盖整个统一 QA 对话；embedding 不混入这些生成模型 token 统计。

`knowledge_evidence` 单独保存实际返回的知识页片段及字符起始位置；`article_evidence` 仍仅记录实际读取的原文。只有知识页、没有原文存档的 Wiki 也允许执行 QA。

使用 `--retrieval-model` 选择整段对话的模型；默认仍为 premium 配置，缺省回退 LLM_MODEL。`--answer-model` 作为旧参数别名保留；两参数指定不同模型会明确报错。

目录导航和摘要文件名已更新，详见 [目录树导航](wiki-tree-navigation.md)。原文版本、摘要分组算法和评估分母保持原有规则。实际 benchmark 对比前需固定题集、语料、模型和总预算，并处理范围内缺失预测的计分。

## 文件与验证

- `llm_wiki_bench/wiki_retriever.py`：摘要入口、目录树、预算内页面阅读和原文工具。
- `llm_wiki_bench/summary_retrieval.py`：BM25、dense 分块召回、RRF 和候选分页。
- `llm_wiki_bench/embedding_client.py`：HTTP embeddings、按内容和模型隔离的 SQLite 缓存。
- `llm_wiki_bench/token_budget.py`：token 计数、完整字符边界截断与续读。
- `llm_wiki_bench/wiki_agent.py`：循环、预算提示、状态合并、答案提交分支。
- `llm_wiki_bench/qa_contract.py`：工具 schemas 与共享引用校验。
- `llm_wiki_bench/run_qa.py`：统一入口、模型参数和结果持久化。
- `tests/test_tree_navigation.py`：树展开、分页、可读标题、摘要重名处理与旧文件迁移。
- `tests/test_qa_loop.py`：缺口拒绝后补查、引用修复、完整需求覆盖、预算边界、多调用响应、停止行为和 runner 输出。
- `tests/test_article_summary_workflow.py`：构建 → 摘要 → 原文 → 同循环提交的集成验证。

离线验证：`.venv/bin/python -m unittest discover -s tests -q`。这些测试使用脚本化模型响应，不代表真实模型 QA 准确率或成本改善。
