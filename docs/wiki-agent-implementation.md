> 2026-09-17 摘要检索更新：默认入口为 summary 层 BM25 + dense + RRF，正文按 token 预算加载，可在同一 QA 循环补查；目录工具继续保留。见 [摘要混合检索](summary-hybrid-retrieval.md)。旧字段加权和哈希匹配未恢复；以下搜索相关表格属于历史实现记录。

> 2026-09-17 QA 更新：检索、证据缺口状态和答案提交已合并到同一对话；知识页信息充分时直接作答，原文读取改为补充和核验入口。当前引用协议、预算和测试见 [统一 QA 循环](qa-agent-loop.md)。下文的强制原文引用规则及独立 `_answer()` 阶段属于历史实现；`validate_answer()` 已移动到 `qa_contract.py`。构建与摘要部分不受此次更新影响。

# Article 引用与 Related Pages 摘要：逐项代码审核

> 后续构建修复见 [单题构建修复](build-test-one-fix.md)：知识页现使用 article 级链接，构建不再要求行号或逐字 quote。下文逐行引用的描述记录第一版设计；相关构建行为以修复说明为准。

本次依据引用任务「重构Wiki原文引用与摘要」最后一次确认实施。当前 checkout 是旧 digest 实现，因此没有套用记忆中的另一套 FactStore/BuildAgent 架构。

## 先看这四条边界

1. **article 是证据底座。** 原样归档处理后的文章，证据链在这里停止；知识页事实直接附 article 行号。
2. **知识页负责组织事实。** Related Pages 允许零个、一个或多个可靠关联，保留简短说明。
3. **summary 负责跨页导航。** 成员集合由 Python 算，模型只写概述和 tags。没有二次主题筛选、连通分量合并、多层摘要或额外 Agent。
4. **QA 用读过的 article 作答。** tags 只是加分；孤立页和 article 可直接搜索；摘要/知识页正文不进入最终答案的证据上下文。

“clean”的取舍是把不可兼容的 digest 生成/修复流程整体替换，而不是在旧 5,832 行 ingestion 中不断增加分支。新 ingestion 保留两步模型调用和批量入口，文档格式、引用验证与分组分别有独立职责。**这是一次 ingestion 内部替换，不是只改几行 prompt。**

## 审核顺序

- [ ] 1. `wiki_documents.py`：article 与引用合同。
- [ ] 2. `bench_ingest.py`：模型提议如何通过验证并保存。
- [ ] 3. `build_summaries.py`：集合分组与摘要生成。
- [ ] 4. `wiki_retriever.py`、`wiki_agent.py`：检索导航与实际原文阅读。
- [ ] 5. `run_qa.py`：逐跳引用与最终答案。
- [ ] 6. 配置、CLI、schema、测试及下方旧函数删除清单。

## 数据流与职责

```text
processed article → archive_article（字节不变）
  → 选页 LLM → 生成 LLM（JSON 事实/引用/关联）
  → render_knowledge（Python 验证 → Markdown）
  → summary_groups（纯集合规则）
  → summary LLM（固定成员 → 概览/tags）
  → wiki_search / wiki_read（导航）
  → source_read（article 片段）
  → answer LLM（短答案 + evidence_chain）
  → validate_answer（逐条引用校验）→ predictions.jsonl
```

LLM 仍决定哪些事实相关、事实措辞、关系含义、摘要表达与推理步骤。Python 能检查引用位置和文本是否真实、路径/成员是否有效；**不能证明引文支持 claim，也不能证明模型列全了所有必要推理跳数。** `citations_validated` 刻意只描述程序实际验证的范围。

## 每个新增或修改函数：为什么、怎么做

### wiki_documents.py

| 审核点                                                                                                | 为什么改                                            | 怎么做                                                                                                         |
| ----------------------------------------------------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| [ ] [wiki_path](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_documents.py:16)         | 模型路径必须准确指向 wiki 内的文件。                | 拒绝绝对路径、目录穿越、重复分隔符和逃出目录的符号链接。                                                       |
| [ ] [parse_document](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_documents.py:29)    | tags、别名可能带逗号；字符串拆分会误读。            | 统一用 YAML 解析 frontmatter，并要求对象结构。                                                                 |
| [ ] [render_document](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_documents.py:40)   | 页面格式不应依赖模型输出是否规范。                  | Python 序列化 YAML，并统一正文换行。                                                                           |
| [ ] [write_document](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_documents.py:44)    | 读取者不应看到写了一半的页面。                      | 先写同目录临时文件，再原子替换单个文件；不声称批次级事务。                                                     |
| [ ] [archive_article](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_documents.py:52)   | 引用需要确定的 article 内容，不能依赖会碰撞的标题。 | 原样复制处理后的 article 字节，以完整 SHA-256 命名；内容改变生成另一份 article。                               |
| [ ] [read_article](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_documents.py:70)      | QA 最终证据必须是实际文章片段。                     | 返回文件的 1-based 闭区间行号、完整引用和内容版本；检查哈希命名文件未被改写。                                  |
| [ ] [validate_citation](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_documents.py:89) | 模型提供的引用不能直接当成原文。                    | Python 重新读取相同行号，逐字比较 quote，并校验可选版本。                                                      |
| [ ] [citation_link](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_documents.py:102)    | 每条事实要能直接定位 article。                      | 统一生成 `[[sources/articles/<hash>#Lx-Ly]]`。                                                                 |
| [ ] [knowledge_pages](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_documents.py:107)  | 摘要不得递归参与自己的分组。                        | 只收集知识目录中的 Markdown；排除 article、summary、synthesis、索引与隐藏路径。                                |
| [ ] [related_pages](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_documents.py:121)    | 普通正文链接和来源链接不等同于可靠的关联。          | 只解析 Related Pages 段落中带简短说明的 wikilink；无说明旧链接不猜测。                                         |
| [ ] [text_field](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_documents.py:137)       | 内容字段不能意外插入新章节或自行伪造 wikilink。     | 限定为非空单行文本；链接由 Python 生成。                                                                       |
| [ ] [string_list](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_documents.py:144)      | aliases/tags 必须是稳定可读的列表。                 | 检查字符串列表并保持顺序去重。                                                                                 |
| [ ] [render_knowledge](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_documents.py:150) | 模型负责提议事实，程序负责保存结构和引用。          | 逐事实校验 article 引用；关联须指向已有页或本批新页且有说明；渲染 Core Facts、Related Pages、Related Sources。 |

### bench_ingest.py

| 审核点                                                                                               | 为什么改                                                 | 怎么做                                                                                                                  |
| ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| [ ] [load_cache](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/bench_ingest.py:46)         | 需要复用成功构建。                                       | 读取现有 JSON 缓存，缺失时使用空映射。                                                                                  |
| [ ] [save_cache](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/bench_ingest.py:52)         | 缓存也不应出现半写文件。                                 | 复用原子写入函数保存 JSON。                                                                                             |
| [ ] [_cache_valid](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/bench_ingest.py:57)       | 旧 digest 缓存不代表新结构完成；共享页可以继续增加事实。 | 要求新 schema、归档内容正确、输出页存在且仍含 article 引用；不要求整页哈希永久不变。                                    |
| [ ] [_article_context](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/bench_ingest.py:75)   | 模型需要可核对的引用坐标。                               | 仅在提示里添加行号，归档文件保持不变。                                                                                  |
| [ ] [_existing_evidence](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/bench_ingest.py:80) | 更新知识页时，旧引用仍需有原文可核对。                   | 收集被选知识页的 article 引用，读出这些文章并提供给生成调用。                                                           |
| [ ] [_ingest_batch_one](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/bench_ingest.py:92)  | 不能把自由 Markdown、空输出或部分覆盖直接标为成功。      | 保留“选页→生成”两次调用；提供 purpose、目录、已读页面和 article；完整检查输出、来源覆盖与关联目标后写页，再写成功缓存。 |
| [ ] [rebuild_indexes](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/bench_ingest.py:165)   | 索引只是导航，不需要另一轮模型事实生成。                 | 按实际知识页确定性地生成目录索引与根索引。                                                                              |
| [ ] [ingest_batch](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/bench_ingest.py:180)      | 单篇和多篇应使用同一个证据合同，失败必须能重试。         | 统一批次处理，失败计数并保留错误；知识页阶段结束后生成摘要，即使本轮知识页均缓存命中也继续未完成摘要。                  |
| [ ] [main](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/bench_ingest.py:213)              | 旧 CLI 应继续可用，并允许隔离重建。                      | 保留 dataset/limit/batch-size/force；新增 wiki-dir，使用 config.RAW_DIR，失败返回非零退出码。                           |

### build_summaries.py

| 审核点                                                                                                 | 为什么改                                                   | 怎么做                                                                                                         |
| ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| [ ] [summary_groups](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/build_summaries.py:16)    | 严格实现已确认的集合规则，避免语义合并或连通分量扩大主题。 | 自身加有效 Related Pages；去单例、同集合和严格子集；重叠但不包含者保留。                                       |
| [ ] [summary_path](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/build_summaries.py:27)      | 同一组从不同知识页发现应复用同一摘要。                     | 排序去重后的成员路径生成完整 SHA-256 文件名。                                                                  |
| [ ] [group_fingerprint](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/build_summaries.py:33) | 重复运行时应复用内容未变的摘要。                           | 对成员路径与全文计算指纹；只是缓存有效性检查，不构建版本历史系统。                                             |
| [ ] [current_summaries](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/build_summaries.py:38) | 旧缓存和已被大集合包含的摘要不能继续作为当前导航。         | 只接受当前候选组且成员、schema、内容指纹一致的摘要；过期文件不删除。                                           |
| [ ] [build_summaries](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/build_summaries.py:66)   | 模型只需概括固定成员，不能决定去重或偷偷改组。             | 把全部成员提供给模型，校验输出字段，Python 写 members 与链接；失败不记录成功缓存，下一次自动重试；支持 limit。 |
| [ ] [main](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/build_summaries.py:118)             | 分组可以先审核，摘要可以独立续跑。                         | 提供 wiki-dir、limit、force 和无调用/无写入的 dry-run。                                                        |

### wiki_retriever.py

| 审核点                                                                                            | 为什么改                                             | 怎么做                                                                                    |
| ------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [ ] [layer](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_retriever.py:41)         | 检索时必须区分导航层和证据层。                       | 根据目录把页面分类为 summaries、knowledge、articles。                                     |
| [ ] [load](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_retriever.py:81)          | 旧 digest 与过期摘要不应继续占据检索结果。           | 排除 digest、synthesis、隐藏文件；摘要通过当前成员校验；其他索引与 BM25 继续沿用。        |
| [ ] [_parse_page](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_retriever.py:136)  | 新 YAML 列表由序列化器生成；article 文件名是哈希。   | 共用 YAML 解析器读取 tags/aliases；article 用原文件 title/source_title 作为检索显示名称。 |
| [ ] [search](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_retriever.py:214)       | 摘要优先不能使孤立知识页或未打标签的页消失。         | 增加 layer 选择和 tags 加分；tags 不过滤；默认 all 保留全部通路，既有评分算法不变。       |
| [ ] [read](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_retriever.py:314)         | 知识页与摘要链接需要直接可跟随，读导航不等于读证据。 | 支持无扩展名 wikilink 与索引；article 返回行数和 source_read 提示，不加入最终证据。       |
| [ ] [source_read](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_retriever.py:375)  | 需要显式记录 QA 看过的文章范围。                     | 只允许已载入 article，默认 80 行，最多 200 行；调用统一原文读取函数。                     |
| [ ] [execute_tool](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_retriever.py:409) | 模型可调用参数必须与实际检索行为一致。               | 传递 layer/tags，结果标记 layer，增加 source_read 分发。                                  |

### wiki_agent.py

| 审核点                                                                                        | 为什么改                                        | 怎么做                                                                                          |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [ ] [retrieve](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_agent.py:94)      | 逐级导航和直接访问都应由同一个检索 Agent 决定。 | 提示摘要优先但允许直达；缺少 article 阅读时提醒；删除超预算自动读页和按问题措辞猜测跳数的回退。 |
| [ ] [_execute_one](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/wiki_agent.py:166) | 每次读证据都要可追踪，坏参数不应伪装成成功。    | 规范调用参数，把验证错误作为工具结果；记录 source_read 返回的实际片段与版本；所有调用计入预算。 |

### run_qa.py

| 审核点                                                                                      | 为什么改                                       | 怎么做                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ ] [validate_answer](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/run_qa.py:50) | 输出证据链不能引用没读过的原文或错版本。       | 每个提交的 hop 必须有 claim 和 citations；校验路径、版本、已读范围内的子区间和完整 quote；仅声明 citations_validated。                                                  |
| [ ] [_answer](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/run_qa.py:89)         | 旧提示允许自身知识补全，与完整证据链目标冲突。 | 只把实际 article 片段交给答案调用，返回结构化短答案及各跳引用；没有片段时直接 unknown。                                                                                 |
| [ ] [main](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/run_qa.py:104)           | 审核答案需要同时看到引用、读取轨迹和错误。     | 增加 wiki-dir；无 article 的旧 wiki 在调用模型前报错；预测 JSONL 写 article_evidence/evidence_chain/evidence_status/error；验证失败写 unknown，并保留已获得的检索轨迹。 |

### bench_config.py

| 审核点                                                                                                  | 为什么改                                 | 怎么做                                                                                              |
| ------------------------------------------------------------------------------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------- |
| [ ] [set_dataset](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/bench_config.py:128)          | 重建不应要求覆盖旧 corpus 或共享旧缓存。 | 新增可选 wiki_dir；指定时缓存位于该 wiki 的 .build-cache.json，输入 article 路径仍按 dataset 选择。 |
| [ ] [get_dir_catalog_text](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/bench_config.py:272) | 新来源目录不再包含 digest。              | 来源计数只统计 articles。                                                                           |
| [ ] [auto_init_page_types](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/bench_config.py:448) | LLM 不能把 summaries 当成普通知识类型。  | 提示及 Python 校验保留 sources/summaries/syntheses；自动类型名限小写字母。                          |

### run.py

| 审核点                                                                               | 为什么改                             | 怎么做                                                |
| ------------------------------------------------------------------------------------ | ------------------------------------ | ----------------------------------------------------- |
| [ ] [step_ingest](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/run.py:80) | 构建失败不能仍向流水线返回成功。     | 传递独立 wiki_dir，并根据知识页与摘要失败数返回状态。 |
| [ ] [run_one](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/run.py:110)    | 独立输出配置必须实际到达 ingestion。 | 向 step_ingest 传递 wiki_dir。                        |
| [ ] [main](/Users/yusen/Documents/Capstone/LLM-Wiki/llm_wiki_bench/run.py:128)       | 完整流水线也要支持隔离重建。         | 新增 wiki-dir；禁止 --all 写入同一个自定义目录。      |

## 函数之外的改动

| 审核点                                       | 为什么、怎么做                                                                                       |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| [ ] ingestion 两个 prompt                    | 选页提供标题、别名、tags、简介；生成提供 purpose 和文章行号，要求 JSON，取消 digest 和关联数量下限。 |
| [ ] summary prompt                           | 不再判断是否有共同主题；按已确定成员写概览，保留不确定性，不推测新关系。                             |
| [ ] agent prompt 与 RetrievalResult.evidence | 引导摘要 → 知识页 → article，同时保留直达；新增实际片段列表，和导航 pages 分开。                     |
| [ ] answer prompt                            | 只准使用已读 article；取消自身知识补全和原先按连字符顺序推断国籍的特殊规则。                         |
| [ ] WIKI_TOOL_SCHEMAS                        | wiki_search 增加 layer/tags；新增 source_read 的文章路径及闭区间行号。                               |
| [ ] FIXED_DIRS                               | 固定目录用 articles 和 summaries；不再创建 digests。                                                 |
| [ ] configs/wiki-schema.md                   | 改为新的文章、知识页、摘要模板及集合/QA 合同。                                                       |
| [ ] README.md                                | 给出隔离重建、独立摘要和离线测试命令；明确这不是原论文相同回答策略。                                 |

## 旧 ingestion 删除清单（重点审核）

旧流程包含自由 Markdown 输出后的大量补救：补 digest 章节、按名称猜原文、强凑关联、再让模型修知识页。新流程在写入前验证结构化提议，失败保留错误供重试，因此不保留这些互相冲突的第二条写入路径。

**同时退出默认 ingestion 的旧行为**包括 Error Book 自动改写、模型自动合并/移动目录、alias 合并、自动全局 overview、矛盾扫描与补写章节。这些不是“仍在暗中运行”。`bench_error_book.py` 文件保留，但新的 ingestion 不调用。此次没有实现替代性的自动纠错/目录治理服务。下表逐个列出原入口及去向，方便你决定是否需要单独恢复某项功能。

删除前版本是任务开始时的 `HEAD`（`d0fd9f4`）；可以用 `git show d0fd9f4:llm_wiki_bench/bench_ingest.py` 对照原代码。

| 原函数（删除前行号）                               | 替代方式 / 删除原因                                                                                |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| [ ] `_read_file_safe`（旧 :50）                    | 关键源文件读取失败必须显式失败；不使用占位符或截断正文代替成功读取。                               |
| [ ] `_apply_prompt_safety_valve`（旧 :67）         | 改为 _ingest_batch_one 的两步 JSON 提议；不拼接 digest 规则、不静默截断证据。                      |
| [ ] `_compute_index_relevance`（旧 :144）          | 改为 _ingest_batch_one 的两步 JSON 提议；不拼接 digest 规则、不静默截断证据。                      |
| [ ] `_get_all_index_content`（旧 :163）            | 改为 _ingest_batch_one 的两步 JSON 提议；不拼接 digest 规则、不静默截断证据。                      |
| [ ] `_expand_dirs_from_selected`（旧 :223）        | 改为 _ingest_batch_one 的两步 JSON 提议；不拼接 digest 规则、不静默截断证据。                      |
| [ ] `_extract_candidate_names`（旧 :246）          | 改为 _ingest_batch_one 的两步 JSON 提议；不拼接 digest 规则、不静默截断证据。                      |
| [ ] `_get_existing_page_names`（旧 :267）          | 改为 _ingest_batch_one 的两步 JSON 提议；不拼接 digest 规则、不静默截断证据。                      |
| [ ] `build_select_pages_prompt`（旧 :326）         | 改为 _ingest_batch_one 的两步 JSON 提议；不拼接 digest 规则、不静默截断证据。                      |
| [ ] `build_select_pages_batch_prompt`（旧 :403）   | 改为 _ingest_batch_one 的两步 JSON 提议；不拼接 digest 规则、不静默截断证据。                      |
| [ ] `build_ingest_prompt`（旧 :459）               | 改为 _ingest_batch_one 的两步 JSON 提议；不拼接 digest 规则、不静默截断证据。                      |
| [ ] `build_ingest_prompt_batch`（旧 :625）         | 改为 _ingest_batch_one 的两步 JSON 提议；不拼接 digest 规则、不静默截断证据。                      |
| [ ] `_predict_article_stem`（旧 :748）             | archive_article 使用原始 article 字节与内容哈希；不再按标题生成或模糊匹配来源。                    |
| [ ] `_normalize_filename`（旧 :759）               | 不再解析或修补任意 Markdown/目录操作；用 render_knowledge 和固定路径规则校验并渲染。               |
| [ ] `_sanitize_frontmatter`（旧 :776）             | 不再解析或修补任意 Markdown/目录操作；用 render_knowledge 和固定路径规则校验并渲染。               |
| [ ] `_extract_type_from_content`（旧 :836）        | 不再解析或修补任意 Markdown/目录操作；用 render_knowledge 和固定路径规则校验并渲染。               |
| [ ] `_check_frontmatter_complete`（旧 :850）       | 不再解析或修补任意 Markdown/目录操作；用 render_knowledge 和固定路径规则校验并渲染。               |
| [ ] `_check_digest_completeness`（旧 :869）        | digest 中间层取消；原文归档和逐条引用改由 wiki_documents 验证，不猜测来源或自动补出处。            |
| [ ] `_inject_dates`（旧 :943）                     | 不再解析或修补任意 Markdown/目录操作；用 render_knowledge 和固定路径规则校验并渲染。               |
| [ ] `_lcs_len`（旧 :995）                          | archive_article 使用原始 article 字节与内容哈希；不再按标题生成或模糊匹配来源。                    |
| [ ] `_fuzzy_match_article`（旧 :1010）             | archive_article 使用原始 article 字节与内容哈希；不再按标题生成或模糊匹配来源。                    |
| [ ] `_inject_article_link_to_digests`（旧 :1028）  | digest 中间层取消；原文归档和逐条引用改由 wiki_documents 验证，不猜测来源或自动补出处。            |
| [ ] `_fix_digest_article_links`（旧 :1086）        | digest 中间层取消；原文归档和逐条引用改由 wiki_documents 验证，不猜测来源或自动补出处。            |
| [ ] `_rebuild_sources_index`（旧 :1139）           | 实际文件生成确定性索引，由 rebuild_indexes 与摘要索引写入负责。                                    |
| [ ] `_rebuild_global_index`（旧 :1231）            | 实际文件生成确定性索引，由 rebuild_indexes 与摘要索引写入负责。                                    |
| [ ] `_extract_existing_overview`（旧 :1270）       | 新的跨知识页概览由 Related Pages summaries 负责；不继续生成独立全局事实概览。                      |
| [ ] `_generate_overview_text`（旧 :1280）          | 新的跨知识页概览由 Related Pages summaries 负责；不继续生成独立全局事实概览。                      |
| [ ] `_count_knowledge_pages`（旧 :1339）           | 实际文件生成确定性索引，由 rebuild_indexes 与摘要索引写入负责。                                    |
| [ ] `quick_lint_bench`（旧 :1353）                 | 前置结构/链接/quote 校验替代事后自动补写；允许零/一个关联，不因缺关联而修改事实。                  |
| [ ] `auto_fix_bench`（旧 :1715）                   | 前置结构/链接/quote 校验替代事后自动补写；允许零/一个关联，不因缺关联而修改事实。                  |
| [ ] `llm_fix_incomplete_digests`（旧 :2256）       | digest 中间层取消；原文归档和逐条引用改由 wiki_documents 验证，不猜测来源或自动补出处。            |
| [ ] `llm_fix_missing_summary`（旧 :2411）          | 前置结构/链接/quote 校验替代事后自动补写；允许零/一个关联，不因缺关联而修改事实。                  |
| [ ] `llm_fix_missing_sections`（旧 :2503）         | 前置结构/链接/quote 校验替代事后自动补写；允许零/一个关联，不因缺关联而修改事实。                  |
| [ ] `llm_fix_empty_related_pages`（旧 :2639）      | 前置结构/链接/quote 校验替代事后自动补写；允许零/一个关联，不因缺关联而修改事实。                  |
| [ ] `_write_related_pages`（旧 :2937）             | 前置结构/链接/quote 校验替代事后自动补写；允许零/一个关联，不因缺关联而修改事实。                  |
| [ ] `_inject_related_sources_for_page`（旧 :3004） | digest 中间层取消；原文归档和逐条引用改由 wiki_documents 验证，不猜测来源或自动补出处。            |
| [ ] `llm_fix_broken_links`（旧 :3082）             | 前置结构/链接/quote 校验替代事后自动补写；允许零/一个关联，不因缺关联而修改事实。                  |
| [ ] `llm_verify_source_grounding`（旧 :3262）      | 退出默认的模型自动治理/改写阶段；该行为不属于此次 article 引用与单层摘要合同，见上方明确说明。     |
| [ ] `llm_detect_contradictions`（旧 :3431）        | 退出默认的模型自动治理/改写阶段；该行为不属于此次 article 引用与单层摘要合同，见上方明确说明。     |
| [ ] `llm_fix_structural`（旧 :3589）               | 前置结构/链接/quote 校验替代事后自动补写；允许零/一个关联，不因缺关联而修改事实。                  |
| [ ] `llm_fix_content`（旧 :3623）                  | 前置结构/链接/quote 校验替代事后自动补写；允许零/一个关联，不因缺关联而修改事实。                  |
| [ ] `llm_fix_all`（旧 :3650）                      | 前置结构/链接/quote 校验替代事后自动补写；允许零/一个关联，不因缺关联而修改事实。                  |
| [ ] `merge_duplicate_pages`（旧 :3661）            | 退出默认的模型自动治理/改写阶段；该行为不属于此次 article 引用与单层摘要合同，见上方明确说明。     |
| [ ] `detect_alias_overlaps`（旧 :3904）            | 退出默认的模型自动治理/改写阶段；该行为不属于此次 article 引用与单层摘要合同，见上方明确说明。     |
| [ ] `generate_overview`（旧 :3999）                | 新的跨知识页概览由 Related Pages summaries 负责；不继续生成独立全局事实概览。                      |
| [ ] `_parse_index_sections`（旧 :4072）            | 实际文件生成确定性索引，由 rebuild_indexes 与摘要索引写入负责。                                    |
| [ ] `_assemble_sections`（旧 :4092）               | 退出默认的模型自动治理/改写阶段；该行为不属于此次 article 引用与单层摘要合同，见上方明确说明。     |
| [ ] `_extract_entry_name`（旧 :4107）              | 退出默认的模型自动治理/改写阶段；该行为不属于此次 article 引用与单层摘要合同，见上方明确说明。     |
| [ ] `relocate_pending_entries`（旧 :4116）         | 退出默认的模型自动治理/改写阶段；该行为不属于此次 article 引用与单层摘要合同，见上方明确说明。     |
| [ ] `consolidate_wiki_bench`（旧 :4420）           | 退出默认的模型自动治理/改写阶段；该行为不属于此次 article 引用与单层摘要合同，见上方明确说明。     |
| [ ] `_apply_consolidate_changes`（旧 :4559）       | 退出默认的模型自动治理/改写阶段；该行为不属于此次 article 引用与单层摘要合同，见上方明确说明。     |
| [ ] `_update_wiki_references`（旧 :4671）          | 退出默认的模型自动治理/改写阶段；该行为不属于此次 article 引用与单层摘要合同，见上方明确说明。     |
| [ ] `periodic_maintenance`（旧 :4724）             | 退出默认的模型自动治理/改写阶段；该行为不属于此次 article 引用与单层摘要合同，见上方明确说明。     |
| [ ] `finalize_wiki`（旧 :4798）                    | 退出默认的模型自动治理/改写阶段；该行为不属于此次 article 引用与单层摘要合同，见上方明确说明。     |
| [ ] `parse_file_outputs`（旧 :4911）               | 不再解析或修补任意 Markdown/目录操作；用 render_knowledge 和固定路径规则校验并渲染。               |
| [ ] `parse_dir_changes`（旧 :4934）                | 不再解析或修补任意 Markdown/目录操作；用 render_knowledge 和固定路径规则校验并渲染。               |
| [ ] `write_wiki_files`（旧 :4953）                 | 不再解析或修补任意 Markdown/目录操作；用 render_knowledge 和固定路径规则校验并渲染。               |
| [ ] `_append_to_index`（旧 :5272）                 | 实际文件生成确定性索引，由 rebuild_indexes 与摘要索引写入负责。                                    |
| [ ] `ingest_single`（旧 :5350）                    | 单篇也是 batch_size=1，统一 _ingest_batch_one；读取已选路径和 article 原文，避免两套不同引用逻辑。 |
| [ ] `_read_selected_pages`（旧 :5459）             | 单篇也是 batch_size=1，统一 _ingest_batch_one；读取已选路径和 article 原文，避免两套不同引用逻辑。 |
| [ ] `_save_article_original`（旧 :5514）           | archive_article 使用原始 article 字节与内容哈希；不再按标题生成或模糊匹配来源。                    |

QA 侧还删除 `_format_context`、`_extract_clean_answer`：不再把全部导航 Markdown 拼进答案上下文，也不靠字符串清理猜测模型最终答案。删除 Agent 的 `_add_page`、`_is_likely_multihop`：不再绕过工具预算自动取页，或根据几个英语关键词猜测证据够不够。

## 测试逐项对应

测试使用临时 wiki 与假模型返回，不调用外部 API、不修改现有 corpus。测试名可直接定位：

| 测试                                                                                                                                                       | 验证的合同                                                 |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [ ] [test_archive_exact_bytes_and_distinct_same_title_versions](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:60)        | archive exact bytes and distinct same title versions       |
| [ ] [test_ranges_quotes_versions_and_paths_are_checked](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:71)                | ranges quotes versions and paths are checked               |
| [ ] [test_knowledge_renders_direct_citations_and_zero_relations](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:85)       | knowledge renders direct citations and zero relations      |
| [ ] [test_exact_groups_subsets_and_overlap](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:101)                           | exact groups subsets and overlap                           |
| [ ] [test_union_coverage_does_not_remove_a_group](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:110)                     | union coverage does not remove a group                     |
| [ ] [test_only_explicit_explained_knowledge_relations_group](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:119)          | only explicit explained knowledge relations group          |
| [ ] [test_summaries_cache_failures_staleness_and_no_recursive_groups](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:128) | summaries cache failures staleness and no recursive groups |
| [ ] [test_summary_limit_leaves_retryable_pending](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:153)                     | summary limit leaves retryable pending                     |
| [ ] [test_ingest_success_cache_and_failed_output_never_cached](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:160)        | ingest success cache and failed output never cached        |
| [ ] [test_bad_batch_validates_before_any_page_write](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:182)                  | bad batch validates before any page write                  |
| [ ] [test_every_input_article_must_be_cited](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:193)                          | every input article must be cited                          |
| [ ] [test_direct_knowledge_article_access_and_soft_tags](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:202)              | direct knowledge article access and soft tags              |
| [ ] [test_agent_gathers_article_evidence_and_respects_budget](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:219)         | agent gathers article evidence and respects budget         |
| [ ] [test_bad_source_tool_returns_error_and_agent_can_recover](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:239)        | bad source tool returns error and agent can recover        |
| [ ] [test_answer_accepts_exact_read_subranges_and_requires_each_hop](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:250)  | answer accepts exact read subranges and requires each hop  |
| [ ] [test_no_article_evidence_means_unknown_without_answer_call](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:271)      | no article evidence means unknown without answer call      |
| [ ] [test_complete_build_summary_navigation_and_answer](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:277)               | complete build summary navigation and answer               |
| [ ] [test_unread_page_cannot_be_overwritten](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:311)                          | unread page cannot be overwritten                          |
| [ ] [test_isolated_build_uses_separate_cache](/Users/yusen/Documents/Capstone/LLM-Wiki/tests/test_article_summary_workflow.py:320)                         | isolated build uses separate cache                         |

## 如何试跑（以下模型调用没有在本次自动执行）

先用独立目录重建。现有 `wiki_output/hotpotqa/wiki`、缓存和结果文件没有在本次代码改动中被转换或重写。旧 digest 页不会自动变成带逐条 article 引用的页；旧 Related Pages 也可能包含为数量要求而制造的关联，直接给旧 wiki 生成 summary 并不能解决其来源质量。

```bash
.venv/bin/python -m unittest discover -s tests -v

# 构建小样本：调用配置中的 LLM；20 是 article 数量
.venv/bin/python -m llm_wiki_bench.bench_ingest --dataset hotpotqa --limit 20 \
  --wiki-dir wiki_output/hotpotqa/article-evidence/wiki

# 只看集合，无模型调用、无写入
.venv/bin/python -m llm_wiki_bench.build_summaries \
  --wiki-dir wiki_output/hotpotqa/article-evidence/wiki --dry-run

# 独立续跑摘要
.venv/bin/python -m llm_wiki_bench.build_summaries \
  --wiki-dir wiki_output/hotpotqa/article-evidence/wiki --limit 10

# 对独立 wiki 做 QA；输出另存，1 是问题数量
.venv/bin/python -m llm_wiki_bench.run_qa --dataset hotpotqa --limit 1 \
  --wiki-dir wiki_output/hotpotqa/article-evidence/wiki \
  --output results/hotpotqa/article-evidence-smoke.jsonl
```

小 article 样本未必覆盖前几个 QA 问题，所以这只是流程 smoke test。没有重新跑 live benchmark，不能据此报告答案质量、语义支持率、费用或耗时改善。

## 已知实现边界

- 写入前验证整个模型提议，但文件替换是单文件原子操作；中途磁盘失败可能留下部分页面，没有整批成功缓存。没有多写入进程事务。
- 模型更新知识页被要求保留旧事实；程序验证引用与来源覆盖，不证明它完整保留了旧语义。
- 摘要成员全文都进入生成上下文；没有实现大组分片、层级摘要或专门规模治理。
- 摘要只做必要的缓存指纹检查，没有引入版本历史、矛盾裁决或第二个审查 Agent。
- QA 的“每跳有证据”程序保证限于模型提交的 hops；未提交的必要 hops 和引文是否真正支持结论，需要语义评估。
- 源文件行号包含 frontmatter；quote 按换行拆分并用 `\n` 拼接，归档字节本身保持原样。

## 本次实际验证记录

- 19 个离线测试全部通过，包括完整的“构建知识页 → 生成摘要 → Agent 逐级阅读 → article 引用 → 答案验证”流程；所有模型调用均为测试替身。
- `compileall`、`git diff --check` 和各 CLI 的 `--help` 检查通过。
- 当前旧 wiki 的只读检查发现 1,594 个知识页，但没有 `sources/articles/` article；新集合规则的 dry-run 输出 `[]`。因此未把旧数据当成新流程已经建成的证据。
- 没有运行在线模型或 live benchmark；没有重写旧 wiki、原有缓存或 benchmark 结果。未创建提交。
