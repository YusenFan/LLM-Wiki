# Yusen 分支冲突解决记录

合并输入：本地 `3bcd590`（article / Related Pages 流程及单题构建修复），远端 `e5048d7`（含 `dc0d406` 的 FactStore / SourceStore 流程和中文说明）。

## 保留规则

- 12 个冲突文件中，10 个业务/文档文件以本地已确认的 article 流程为基准；`package.json` 和 lockfile 使用远端 Prettier 配置。
- 保留当前初始化的抽样及目录生成行为，没有采用远端“禁止抽样、写空类型目录”的初始化方案。
- 保留本地全部 26 个离线测试与构建修复。
- 保留远端独立改进：`.gitignore`、`.prettierignore`、文章标题/正文变体及文件名防碰撞、评估附加字段、模型用量字段、未冲突模块的中文注释。

## 合并后兼容修复

1. 预处理返回本轮实际写入的 `article_paths`。`build_test_one.py` 只导入这份列表；保留旧 raw 文件，但不因新命名方式导致一题重复读入旧文件。
2. Agent 从消息中取出 `_usage` 后再发回模型；记录 retrieval 用量，避免把内部统计字段作为 API 消息字段发送。
3. 包导出从旧 `summary_store` 改为当前 `wiki_documents`。
4. 新增防碰撞和用量字段集成回归测试，并补上已有旧 raw 文件时仍只处理第一题文章的脚本测试。

## 没有混入默认源码的旧实现

下列文件来自远端另一套已被当前设计替代的接口。其引用使用 `source_id/version_id`、`fact_apply`、`summary_read` 等旧合同，直接并入会导致导入错误或改变已确认的初始化/分组行为。因此本次合并不将它们引入当前树，而非禁用失败测试：

- `llm_wiki_bench/build_agent.py`
- `llm_wiki_bench/wiki_store.py`
- `llm_wiki_bench/summary_store.py`
- `llm_wiki_bench/validate_wiki.py`
- `llm_wiki_bench/resume_summaries.py`
- `llm_wiki_bench/retrieval_experiment.py`
- `tests/test_wiki_tools.py`、`tests/test_summaries.py`（对应旧 API；其中独立的预处理和用量用例已迁移）
- `WIKI_AGENT_PLAN.md`、`docs/cross-document-summaries.md`、`docs/function-reference.md`、`docs/system-learning-guide.md`、`docs/wiki-agent-acceptance.md`（描述旧接口）

这些内容仍完整存在于远端提交 `e5048d7`，本机另保存到 `/tmp/llm-wiki-merge-review/retired-remote/`。冲突文件双方原文保存到该目录下 `stage-2/` 和 `stage-3/`。

## 验证与提交注意

- 28 个离线测试通过，无外部模型调用。
- 编译、冲突标记、JSON 配置及暂存差异检查。
- 本地原提交 `3bcd590` 已跟踪 `.env` 和 `node_modules/.package-lock.json`。从新树停止跟踪会保留本机文件，但不清除旧提交历史。推送前必须清理未发布历史中的凭据文件；本说明不包含任何密钥值。

## 用户确认后的历史清理

用户已授权清理未推送提交并完成合并。本地重构提交以相同父提交、作者和提交说明重新创建，仅从其树中移除 `.env` 与 `node_modules`；合并提交保留远端 `e5048d7` 为父提交。原本机文件保留，不创建可被误推送的旧历史备份分支。旧对象可能仍由本机 reflog 保留，清理目标是待推送的分支历史，不是安全擦除本机数据。
