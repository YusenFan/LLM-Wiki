# HotpotQA 100 题代码与失败分析（2026-09-18）

本次是只读审查与离线复算，没有修改业务代码，没有重跑模型。结论：0.5564 的答案 F1 可以复现；下降同时涉及样本组成、重新构建后的知识内容与导航结构，以及已确认的代码缺陷。不能简单归结为“题目数量越多越难”。

**核对范围与可比性**

- 当前本地 commit：`e223ab8`。VM 没有 Git 历史；两端 `llm_wiki_bench/*.py` 共 20 个文件 SHA-256 完全一致。
- VM 预测：`/home/azureuser/LLM-Wiki/results/hotpotqa/qa-20260918-005513/predictions.jsonl`；运行日志同目录 `qa.log`。
- VM Wiki：`wiki_output/hotpotqa/first-100/wiki`；小 Wiki 与预测：本地 `wiki_output/hotpotqa/test-ten/`。
- 前 10 题 id/question/answer 的 SHA-256 完全一致：`ebc556a90dab0afcdc8d8d51561abc7a07d8d53d54a3b6bca97cac8d0376cb72`。
- 两份预测记录均显示 QA 模型 `gpt-4o`、hybrid summary 检索、limit=5、candidates=20、token_budget=4000，实际 hybrid 未降级。VM 日志确认 T_max=15。
- 两次 Wiki 独立构建，目录、知识事实和 Related Pages 不相同。本地 10 题历史运行没有完整模型/提供商版本和构建 manifest；当前代码哈希一致不能反推历史每次调用配置完全相同。因此以下比较不是只控制“语料大小”单一变量的因果实验。

| 运行/子集                 | 题数 | Answer F1 |   EM | 平均工具调用 |
| ------------------------- | ---: | --------: | ---: | -----------: |
| 本地小 Wiki               |   10 |  0.866667 | 0.80 |         4.80 |
| VM 大 Wiki 的相同前 10 题 |   10 |  0.690476 | 0.50 |         8.90 |
| VM 大 Wiki 全部题目       |  100 |  0.556429 | 0.45 |         9.41 |
| VM 大 Wiki 后 90 题       |   90 |  0.541534 |    — |            — |

前 10 题中，第 1、10 题由正确答案变成引文失败后的 unknown；第 5 题少答 New York City；第 8 题反而从错误的 4,000 改为正确数值 3,677，但缺少标准答案的 seated。小 Wiki 中这一页省略了原文“3,677 seated”，只保留总容量 4,000，说明 0.8667 本身也存在知识压缩缺陷。

100 题分布：45 题 F1=1；16 题部分得分；39 题 F1=0。其中 27 题 unknown，12 题给了非 unknown 但零分的答案。27 道 unknown 中 25 道用满 15 个工具调用；7 道停止原因为 budget_exhausted，20 道是成功提交 unknown。后者不能误记为没有预算问题。

复算遵循[官方答案评分脚本](https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py)的归一化及 yes/no/noanswer 规则，100 题总 F1 仍为 0.556429；本次没有漏题导致的分母偏差。这是 Answer F1，不是 supporting-fact F1 或 joint F1。

**1. [P1] 更新知识页会删除旧事实和来源；最终建库成功统计掩盖覆盖损失**

位置：[bench_ingest.py:100](../llm_wiki_bench/bench_ingest.py#L100)、[bench_ingest.py:125](../llm_wiki_bench/bench_ingest.py#L125)、[bench_ingest.py:212](../llm_wiki_bench/bench_ingest.py#L212)。

调用链：`_ingest_batch_one → _validate_proposal → render_knowledge → write_document`。验证器只强制本批输入文章出现于输出；对选中的旧页面，没有强制保留它已有的文章覆盖或事实。然后整页替换。prompt 中“preserve existing supported facts and citations”没有对应的代码约束。

已在临时本地 fixture 复现：先有 A 来源支持“1941 年出生”，后来用 B 来源新增“1986 年执教”；只输出 B 的 proposal 仍通过验证，旧事实与 A 链接均消失。这个例子没有调用 LLM。

VM 实证：

- `build-result.json`：failed=0；最终成功缓存共 991 条。
- 逐条执行当前 `_cache_valid` 后，只有 975 条有效，16 条失效。
- 其中 14 篇已归档原文，没有任何知识页再引用。另 2 条只是部分旧输出失去引用，原文仍有别处入口。
- 第 31 题 `Shut Up, Make Love`：旧缓存指向 `entities/poison_american_band.md`；该页当前没有该来源和 2000 年发行事实，QA 只读 Poison 后 unknown。
- 第 85 题 `Their Lives`：旧输出 `entities/monica_lewinsky.md` 当前核心事实只有出生日期，书籍来源及人物联系消失，QA unknown。
- 第 90 题 Aram Avakian 的旧来源也消失，但这道题仍有职业证据，只因 film director/director 表述差异扣分。不能把 14 个丢失来源直接算成 14 道错题。

建议：更新使用事实级 upsert，事实有稳定 id、来源集合和替代/删除理由；正常增量合并不得静默抹掉旧事实。先保证被覆盖页面的旧来源覆盖不无故减少，再建立事实保留验证。全量 build 结束后审计所有回执和来源入口，覆盖损失应进入失败/待修复统计。仅反复重跑失效缓存可能发生“恢复 A 时又覆盖 B”，不是根治。

**2. [P1] Summary 是唯一有排序的搜索入口，绝大多数知识页没有直接检索机会**

位置：[summary_retrieval.py:69](../llm_wiki_bench/summary_retrieval.py#L69)、[build_summaries.py:17](../llm_wiki_bench/build_summaries.py#L17)、[wiki_retriever.py:136](../llm_wiki_bench/wiki_retriever.py#L136)。

`SummaryIndex`明确跳过所有非 summary 页面；summary 组又要求至少两个 Related Pages 成员。当前 QA 工具只有 summary_search、wiki_tree、wiki_read、source_read，没有知识页标题/别名/事实搜索。

| 指标                          | 小 Wiki | VM 大 Wiki |
| ----------------------------- | ------: | ---------: |
| 归档文章                      |     100 |        991 |
| 知识页                        |      99 |        963 |
| 当前有效 summary              |      22 |        141 |
| 至少进入一个 summary 的知识页 |      35 |        229 |
| 未成为 summary 成员的知识页   |      64 |        734 |

这不代表 734 页绝对无法被找到：它们仍可能被其他页面提及、直接猜路径或通过目录发现。但它们自己的内容没有被当前搜索索引覆盖。

按 gold supporting title 与知识页标题精确归一化匹配计算，100 题里只有 16 题的两个标题都能映射到任意 summary 成员；初始 top-5 中是 15 题。51 题两个标题都没有成员入口。此统计是标题代理，会受实体合并、重命名和别名影响，不能称为事实召回率。

实际 941 次工具调用中，385 次是 wiki_tree，占 40.9%。第 84 题 Handi-Snacks：7 次 summary_search + 7 次 tree，仍读了 0 个证据页；其全部 gold 原文在 archive 内。第 1 题寻找 Scott Derrickson 用了多次 summary 重试和目录分页，虽然最后找到，留给提交修复的预算已经很少。

建议：保留 summary 导航，同时对所有知识页建立标题/别名及事实检索。已知实体先精确标题/别名定位；关系问题使用知识页 BM25/embedding 排序；summary 提供跨页路线。原文作为缺事实、歧义、冲突的补充入口。不要为让孤立页“可搜索”而虚构 Related Pages。

**3. [P1] 引文错误反馈与重试机制不能有效纠正，正确候选被丢为 unknown**

位置：[qa_contract.py:35](../llm_wiki_bench/qa_contract.py#L35)、[wiki_agent.py:154](../llm_wiki_bench/wiki_agent.py#L154)、[wiki_agent.py:246](../llm_wiki_bench/wiki_agent.py#L246)。

当前严格匹配是合理的出处约束；问题是模型必须自行重复一段完整引文，而知识页错误只返回泛化字符串，没有指出具体 page/quote/字段/可用文本。相同错误 proposal 没有去重或升级修复。每次失败仍消耗工具预算。

实证：第 1、10、29、35、68、95 题曾提交与 gold 完全一致的答案，最终却 unknown。第 29 题第 3 步已经回答 Henry J. Kaiser，随后完全相同的失败提交连续发生 13 次。第 35、95 题把不同事实行去掉中间来源链接后拼成一个 quote，文本语义对但不是连续原文。

第 1 题知识页实际为“An American director...”的 description，模型加上“Scott Derrickson is...”再引用，三次相同失败。它需要引用实际文本，不需要重新寻找国籍。

建议：read 时生成不可变 evidence_id/span_id，模型选 ID，Python 按已读快照装配原文。知识页事实可以直接作为证据，不强制再 source_read。短期先返回具体失配位置、已读片段候选及原因；相同失败两次触发专门修复分支。为导航和最终提交/修复分配明确预算。保留 proposed_answer 与 validation_failure，区分“不会答”与“答了但没通过出处校验”。

这里的 6 题是可识别的损失案例，不是已测得修复收益。即使六题都修好，其他检索和语义错误仍存在；不能把它宣传为重跑必然增加 6 分。

**4. [P1] 导航关系未经来源验证，却可以当成正式知识证据**

位置：[wiki_documents.py:196](../llm_wiki_bench/wiki_documents.py#L196)、[wiki_agent.py:272](../llm_wiki_bench/wiki_agent.py#L272)、[qa_contract.py:39](../llm_wiki_bench/qa_contract.py#L39)。

Related Pages 的 reason 仅验证非空文本及目标页存在，不需要来源引用；description 同样没有事实级来源约束。QA 把整个知识页 body 收为 page_evidence，只要 quote 是子串就接受，无法区分 Core Facts、description、关系理由。

第 71 题回答 Raven 的引文来自 Tara Strong 的 Related Pages 说明，而非 Core Facts；当前提供的该人物原文没有这句 Raven 事实。第 62 题 Innerspace 的“与 Alien 都由 Jerry Goldsmith 配乐”关系理由也没有独立来源，模型据此改选电影。在本地 fixture 中，一条没有文章引用的 Related Pages reason 可直接通过 validate_answer。

建议：可引用的知识事实与纯导航关系在数据上分离。description/关系若要当证据，必须映射到有出处的事实/span；否则只参与导航。页面仍可作为 QA 主要证据，不必回退成“所有答案都读原文”。

**5. [P2] 工具参数容易造成可避免的失败并消耗预算**

位置：[wiki_retriever.py:187](../llm_wiki_bench/wiki_retriever.py#L187)、[wiki_retriever.py:299](../llm_wiki_bench/wiki_retriever.py#L299)、[wiki_documents.py:102](../llm_wiki_bench/wiki_documents.py#L102)。

- 40 次虚拟 Wiki 目录格式错误；其中 `/entities` 36 次。根路径用 `/`，子路径却必须没有开头斜线。另 4 次目录确实不存在。共有 35 道题发生 tree 错误。
- 19 次 source_read 行范围错误，涉及 17 题；典型是请求 1–20 行，而文章只有 11 行（16 次）。API 返回错误而不是已有段落。
- 预处理把全文所有句子 join 成一行，原始句子编号不保留在 QA 数据里；文章还有 YAML 与标题。模型常把内容摘要当作包含元数据的 1–11 行完整 quote，再次失败。

建议：在虚拟 Wiki 路径边界统一规范化开头/末尾斜线，同时继续禁止 traversal；source_read 的读取请求可把过大的 end 截到 EOF 并返回实际范围，引用验证仍严格针对实际返回文本。给模型稳定段落/句子 id，减少手工行号与引文复制。不要取消出处校验来“提高”分数。

**6. [P2] 评估字段过时、输出目录共享、缺失预测分母错误**

位置：[evaluate.py:116](../llm_wiki_bench/evaluate.py#L116)、[evaluate.py:137](../llm_wiki_bench/evaluate.py#L137)、[run_qa.py:151](../llm_wiki_bench/run_qa.py#L151)、[run_qa.py:184](../llm_wiki_bench/run_qa.py#L184)。

- runner 写 retrieval_llm_calls/retrieval_usage_by_model、evidence_chain[*].citations；evaluator 却读 llm_calls/usage_by_model、顶层 citations，造成全 0。
- 实际 100 题有 866 次成功记录的 QA 模型调用、114 个最终答案引用条目；模型记录总 tokens=13,112,578（不含 embedding，不代表唯一内容量，也未换算费用）。
- evidence_gap_rate=0.14 只统计模型记录了哪些 unresolved requirements，不是 27% unknown 的同义指标。
- 自定义 `--output` 仍把 summary/details 写回共享 `results/hotpotqa/`，多次实验会覆盖评估文件。当前这份 0.5564 已重新对预测复算匹配。
- 缺失预测被排除分母。最小复现：100 题仅 1 道正确预测，得到 total=1、missing=99、F1=1。本次 missing=0，因此不是本次分数下降原因。
- 当前归一化顺序和 yes/no/noanswer 特例与官方实现有差异。例如 yes indeed 对 yes 当前得到 2/3，官方为 0。本次 100 题按官方规则复算没有有意义的逐题分数差异（仅浮点舍入）。
- 预处理未保留 supporting_facts 的 sentence index，当前只报告标题召回代理，不是官方 supporting facts/joint 指标。

建议：所有结果与 manifest 放在独立 run 目录，记录问题集合、原始数据/语料/Wiki/代码哈希、模型与 endpoint 标识（不含密钥）、prompt、工具及 embedding 配置。缺失按零分计并单独报告；保留官方 Answer/Sup/Joint 指标和项目特有覆盖、出处验证、unknown 指标，避免混用。

**架构上的另外两个限制**

1. 知识编译保证“每篇输入至少被引用一次”，不保证所有可回答事实被保留。第 58 题 Jerry Glanville 的生日、第 61 题 Muggsy 的 Charlotte Sting 教练经历、第 62 题 Alien 的执行制片人、第 70 题赛季 EFL Cup 参赛信息均在原文内，却不在已读知识页。用可追溯事实单元保留日期、数值、职位、关系、限定条件；对高风险压缩建立抽样完整性检查。QA 发现具体缺口后直接读取已知来源，而不是继续宽泛 summary_search。日期/数量同页有冲突时显式保留适用条件。
2. requirements 由同一个 LLM 自己声明、自己标 supported。Python 只检查引用存在和 id 覆盖，不检查问题是否真的分解完整。第 42 题把“在 Kochi 工作”当成“总部在 Kochi”；第 86 题把 pollster 当成题目要求的 lawyer/lobbyist；第 32 题引用了 Fujioka, Gunma 却回答 Japan。将要求表达为 subject/relation/object/限定条件/答案类型，提交前核查实体、第二跳和所问粒度。可让同一 agent 检查，也可实验一个受限 verifier，但不能声称普通 Python 引文校验已经证明语义正确，更不应强制每题固定读两篇：一个知识页可能已融合全部证据。

当前实现顺序是：原始 context → article archive → LLM 生成/更新知识页 → Related Pages 分组 → summaries → summary-only hybrid 检索 → agent 翻页/读证据 → 引文检查 → 答案评分。

建议演进为：不可变原文及句子编号 → 可追溯、可增量合并的知识事实 → 全量知识页检索与 summary 导航并行 → 按问题缺口多跳读取 → 必要时原文补充 → 按 evidence_id 提交与语义约束检查 → 按 run 独立评估。summary 继续发挥跨页导航作用，不能决定某个事实是否拥有搜索入口。

**实验设计：先修正确性，再隔离测量**

- 第一批修复：旧事实覆盖保护和全库审计；评估字段/目录/分母；工具参数容错；引文诊断、重复失败控制和证据 ID。它们可通过离线 fixture 验证。
- 冻结当前 100 题与 Wiki 快照，分别测：全知识页检索；引文 ID 提交；二者组合。保持模型、问题、预算不变。先不要同时更换大模型、重建 Wiki、改变检索与增加预算，否则难以归因。
- 独立评估建库：核对 gold 支撑句在原文、知识页中的保留情况；gold 只用于离线审计。不要让评测答案/支撑标注进入线上检索与建库。
- 比较“给 gold 原文的 answer-only”“给目标知识页的 answer-only”“端到端检索”，分别识别回答能力、编译损失与导航损失。实际跑分收益本次未测量。
- 扩容实验固定同一组 100 题，在冻结的基础 Wiki 上增量加入干扰语料；另做独立重建对照。记录事实保留率、候选/已读事实召回、所有需求覆盖率、失败提交率、unknown、EM/F1、tokens、延迟。
- 当前把所有题的 distractor context 合并进一个公共 Wiki，题数从 10 增至 100 同时将来源从 100 增至 991。它是项目自定义的 pooled-corpus 评测。[官方 distractor 设置](https://hotpotqa.github.io/)是每题 10 个段落，官方 fullwiki 又有指定全 Wikipedia 语料。报告时不能把本结果直接当官方 distractor 或 fullwiki 排行榜成绩。若测 distractor，就按每题的原始 10 段限制检索；若测跨问题全局知识库，就固定语料与命名该设置。

**标注与词级评分的边界**

16 道部分得分里，多道是别名或限定语差异：Bill Clinton/William Jefferson Clinton；Virginia Woolf/Adeline Virginia Woolf；director/film director；3,677/3,677 seated。不能把它们全部算成事实错误，也不能为了抬分偷偷更改 gold 或官方归一化。

有些零分涉及标注/问题疑点：第 22 题原文支持 Dr. Robotnik，而 gold 为 Sonic；第 78 题外交对象原文为 Ais，而 gold 为 Apalachees；第 36 题同段落有 1922 和 1923 两种结束口径。第 16 题 country/county、第 89 题单篇故事与全部作品销量也有措辞问题。应保留官方分数，附人工审计标签；不能简单断言所有这些都是模型幻觉或把它们全部改判正确。

**验证和产物**

- 现有 68 个离线单元测试通过。这不覆盖此次发现的所有边界；最小复现另外确认了旧事实丢失、导航关系可引用、目录斜线错误、原文超界全失败、缺失分母和官方评分差异。
- 审计脚本与派生统计在 [results/review-20260918](../results/review-20260918/)；该目录被 Git 忽略。重点文件：[reproduce.py](../results/review-20260918/reproduce.py)、[failure-analysis.json](../results/review-20260918/failure-analysis.json)、[summary.json](../results/review-20260918/summary.json)、[build-integrity.txt](../results/review-20260918/build-integrity.txt)。
- 对全部 100 条预测做了统计，对所有 55 条未满分记录给出下面的归类；重点例子核对了 tool_calls、已读知识页及原始支撑句。标为“标题/轨迹诊断”的条目尚未逐句做完整语义证明；分类可以重叠，不能相加解释成独立因果贡献。
- VM 数据在 VM 内只读分析；整包下载被自动审批拒绝后没有执行归档传输。只保留此次审查所需的派生指标和失败题摘要。

**全部未满分记录（题号为本次 100 题顺序）**

| 题号 | id                         | Gold                                                                | Prediction                                   |     F1 | 诊断与依据                                                                                                                                                        |
| ---: | -------------------------- | ------------------------------------------------------------------- | -------------------------------------------- | -----: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|    1 | `5a8b57f25542995d1e6f1371` | yes                                                                 | unknown                                      | 0.0000 | 引文修复失败：已读两人国籍；三次提交 yes，引文改写而非逐字复制，耗尽预算。                                                                                        |
|    2 | `5a8c7595554299585d9e36b6` | Chief of Protocol                                                   | Chief of Protocol of the United States       | 0.6667 | 答案表述：职位后多输出 of the United States；词级 F1 降低。                                                                                                       |
|    5 | `5a8e3ea95542995a26add48d` | Greenwich Village, New York City                                    | Greenwich Village                            | 0.5714 | 答案粒度：Greenwich Village 少了标准答案中的 New York City。                                                                                                      |
|    8 | `5a87ab905542996e4f3088c1` | 3,677 seated                                                        | 3,677                                        | 0.6667 | 答案表述：数值正确；缺少 seated，F1=2/3。小 Wiki 此题反而回答 4,000。                                                                                             |
|   10 | `5a8db19d5542994ba4e3dd00` | yes                                                                 | unknown                                      | 0.0000 | 引文修复失败：两支乐队均已读，四次提交 yes，被改写引文卡住。                                                                                                      |
|   12 | `5a877e5d5542993e715abf7d` | David Weissman                                                      | unknown                                      | 0.0000 | 导航未覆盖：未读到 David Weissman/The Family Man，转入其他电影后用尽 15 步。此分类依据轨迹和标题。                                                                |
|   14 | `5ab56e32554299637185c594` | no                                                                  | yes                                          | 0.0000 | 语义/题目歧义：两建筑都读到；把混合用途大楼和地产公司总部统称 real estate。问题措辞本身含糊，标准为 no。                                                          |
|   15 | `5ab6d09255429954757d337d` | from 1986 to 2013                                                   | 1986 to 2013                                 | 0.8571 | 答案表述：1986 to 2013 少了 from，实体与年份正确。                                                                                                                |
|   16 | `5a75e05c55429976ec32bc5f` | 9,984                                                               | unknown                                      | 0.0000 | 问题歧义与偏航：已读到 Brown County 人口 9,984；问题把 county 写成 country，之后读 Trump/Kansas 等无关页面。                                                      |
|   19 | `5ae0d4c9554299603e418468` | 1969 until 1974                                                     | 1969 to 1974                                 | 0.6667 | 答案表述：1969 to 1974 对比 1969 until 1974，年份一致，连接词不同。                                                                                               |
|   20 | `5a8133725542995ce29dcbdb` | Robert Erskine Childers DSC                                         | unknown                                      | 0.0000 | 证据与题目不完全匹配：已读 British/American 及原文；问题要求 England，证据是 British。不能把谨慎弃答简单归为检索缺失。                                            |
|   22 | `5ae6050f55429929b0807a5e` | Sonic                                                               | Dr. Robotnik                                 | 0.0000 | 标注疑点：原文明确 Jim Cummings 配音 Dr. Robotnik；标准却是 Sonic，问法与支撑段落不一致。                                                                         |
|   23 | `5adddccd5542997dc7907069` | keyboard function keys                                              | unknown                                      | 0.0000 | 导航+引用失败：找到了 Apple Remote/Front Row 线索，未读取 Front Row；之后反复原文引用校验失败。不能算已经获得最终答案。                                           |
|   24 | `5ae2070a5542994d89d5b313` | Badly Drawn Boy                                                     | unknown                                      | 0.0000 | 导航未覆盖：只读到 Badly Drawn Boy，缺 Wolf Alice，15 步 unknown。标题/轨迹诊断。                                                                                 |
|   25 | `5ae22b8d554299234fd0440f` | World's Best Goalkeeper                                             | IFFHS World's Best Goalkeeper                | 0.8571 | 答案表述：多输出 IFFHS；奖项核心正确。                                                                                                                            |
|   26 | `5a722b8655429971e9dc9329` | Barton Lee Hazlewood                                                | Lee Hazlewood                                | 0.8000 | 别名/全名：Lee Hazlewood 对比 Barton Lee Hazlewood。其一个来源回执失效，但相关原文仍被另一页引用。                                                                |
|   29 | `5a828c8355429966c78a6a50` | Henry J. Kaiser                                                     | unknown                                      | 0.0000 | 引文修复失败：第 3 步已提交 Henry J. Kaiser，完全相同的失败提交重复 13 次。                                                                                       |
|   30 | `5a8a3e745542996c9b8d5e70` | Arena of Khazan                                                     | Crusaders of Khazan                          | 0.6667 | 错误实体：Arena of Khazan 被错换为 Crusaders of Khazan，未读目标 adventure。                                                                                      |
|   31 | `5ac23ff0554299636651994d` | 2000                                                                | unknown                                      | 0.0000 | 建库覆盖被覆盖+导航：Shut Up, Make Love 原文还在，但旧输出 Poison 页面已经不再链接它；该题只读 Poison，15 步 unknown。                                            |
|   32 | `5ae4a3265542995ad6573de5` | Fujioka, Gunma                                                      | Japan                                        | 0.0000 | 答案粒度/约束：读到且引用 Fujioka, Gunma，最终却只回答 Japan。                                                                                                    |
|   33 | `5ae0361155429925eb1afc2c` | Charles Eugène                                                      | Charles Nungesser                            | 0.5000 | 全名/标注形式：Charles Nungesser 对比标准 Charles Eugène；原文全名 Charles Eugène Jules Marie Nungesser，同一实体。                                               |
|   35 | `5a7cc50e554299452d57ba3e` | Letters to Cleo                                                     | unknown                                      | 0.0000 | 引文修复失败：已提出 Letters to Cleo；将两个 facts 拼成一个连续 quote，五次失败至预算耗尽。                                                                       |
|   36 | `5abf63f15542997ec76fd3ea` | October 1922                                                        | 1923                                         | 0.0000 | 证据内部冲突：原文及知识页同时保留 October 1922 与 ended in 1923 的不同口径；模型选择后者。                                                                       |
|   42 | `5a74106b55429979e288289e` | Mumbai                                                              | Kochi                                        | 0.0000 | 第二跳遗漏：工作地点 Kochi 被当作公司总部；没有读取 Tata Consultancy Services 的 Mumbai 总部事实。                                                                |
|   43 | `5a79311755429970f5fffe67` | 1962                                                                | unknown                                      | 0.0000 | 导航未覆盖：读到 Animorphs 等无关小说页面；未找到 I's/Masakazu Katsura，15 步 unknown。标题/轨迹诊断。                                                            |
|   44 | `5ab2d3df554299194fa9352c` | sovereignty                                                         | Ethiopian sovereignty                        | 0.6667 | 答案表述：Ethiopian sovereignty 对比 sovereignty，多了修饰语。                                                                                                    |
|   45 | `5a760ab65542994ccc918697` | Nelson Rockefeller                                                  | unknown                                      | 0.0000 | 第二跳遗漏：Alfred Balk 页面已给 Nelson Rockefeller；转去 Gerald Ford，没有验证 Nelson 的副总统身份，提前 unknown。                                               |
|   46 | `5a7d54165542995f4f402256` | Yellowcraig                                                         | Yellowcraigs                                 | 0.0000 | 别名：Yellowcraigs 对比 Yellowcraig，词级归一化不做词形/实体别名等价。                                                                                            |
|   48 | `5add61d65542995b365fab21` | Organizations could come together to address global issues          | unknown                                      | 0.0000 | 第二跳遗漏：读了多个苏联人物及 Gorbachev 原文，未读 World Summit of Nobel Peace Laureates 的 forum 事实。                                                         |
|   55 | `5ab96ab755429970cfb8eacd` | Max Martin, Savan Kotecha and Ilya Salmanzadeh                      | unknown                                      | 0.0000 | 分支选择错误：读到 Delirium 的三个单曲，却沿 Army 和 Something in the Way You Move 查；未读取包含合作者的 On My Mind。                                            |
|   56 | `5a8a43eb5542996c9b8d5e82` | Marion, South Australia                                             | unknown                                      | 0.0000 | 导航未覆盖：进入 Kansas City 等无关页面，未读到 Marion/Westminster School；15 步 unknown。标题/轨迹诊断。                                                         |
|   58 | `5a7320565542991f9a20c61d` | Keith Bostic                                                        | unknown                                      | 0.0000 | 知识压缩+未按缺口读原文：Jerry Glanville 出生日期在原文里但未保留在知识页；已经明确缺出生日期，仍未 source_read 该来源。                                          |
|   60 | `5adc53f75542996e6852530a` | no                                                                  | yes                                          | 0.0000 | 实体替换：把 common-name Cypress 替换为 genus Cupressus，比较错对象；有效引用不保证实体匹配。                                                                     |
|   61 | `5a8b20335542996c9b8d5fb3` | shortest player ever to play in the National Basketball Association | unknown                                      | 0.0000 | 知识压缩+未按缺口读原文：Muggsy 页面保留最矮球员，却删去 Charlotte Sting 教练经历；为验证条件转去无关队伍，未读 Muggsy 原文。                                     |
|   62 | `5a85fb085542994775f606de` | Ronald Shusett                                                      | Steven Spielberg                             | 0.0000 | 知识压缩+错误实体+无来源关系：Alien 原文 Shusett was executive producer 被省略；随后用 Innerspace/Spielberg 作答。Related Pages 的 Goldsmith 关系无独立来源引文。 |
|   63 | `5a7be2595542997c3ec972ac` | Adeline Virginia Woolf                                              | Virginia Woolf                               | 0.8000 | 别名/全名：Virginia Woolf 对比 Adeline Virginia Woolf。                                                                                                           |
|   65 | `5a8f4c8d554299458435d5a3` | more than 70 countries                                              | more than 70                                 | 0.8571 | 答案表述：more than 70 少了 countries。                                                                                                                           |
|   68 | `5ae7ba7a5542993210983f12` | Usher                                                               | unknown                                      | 0.0000 | 引用校验+第二跳不足：最后一步提出 Usher，但把一句话当整段 L1-L11 引文；未读另一个标注支撑页 I Don't Wanna Know。                                                  |
|   70 | `5aba7cfe554299232ef4a2fd` | Carabao Cup                                                         | unknown                                      | 0.0000 | 知识压缩+第二跳遗漏：赛季知识页漏掉 EFL Cup 参赛信息；虽读原文，仍未读取 EFL Cup 的 Carabao Cup 事实。                                                            |
|   71 | `5ae5aba0554299546bf82f17` | Teen Titans Go!                                                     | Raven                                        | 0.0000 | 答案类型错误+无来源关系：要求电视剧名却回答人物 Raven；引用的是 Related Pages 关系说明，并读了漫画团队而非 Teen Titans Go! 页面。                                 |
|   72 | `5ae1f4cb554299234fd0436d` | 276,170 inhabitants                                                 | unknown                                      | 0.0000 | 第二跳遗漏：读错 122nd Division 后找到 122nd SS-Standarte，但未读 Strasbourg 人口页；15 步 unknown。标题/轨迹诊断。                                               |
|   74 | `5ae37c765542992f92d822d4` | Tromeo and Juliet                                                   | unknown                                      | 0.0000 | 导航未覆盖：读 Romeo + Juliet、Romeo and Juliet on screen、James Gunn，未读 Tromeo and Juliet。标题/轨迹诊断。                                                    |
|   75 | `5ae33c4d5542992f92d82262` | William Jefferson Clinton                                           | Bill Clinton                                 | 0.4000 | 别名/全名：Bill Clinton 对比 William Jefferson Clinton。                                                                                                          |
|   76 | `5a77cb335542997042120b3a` | John John Florence                                                  | unknown                                      | 0.0000 | 相近标题干扰：读各年份比赛页面，未读 John John Florence/MEO Rip Curl Pro Portugal 目标页。标题/轨迹诊断。                                                         |
|   78 | `5a713ea95542994082a3e6e4` | Apalachees                                                          | Ais                                          | 0.0000 | 标注疑点：Alvaro Mexia 段落明确 Ais；另一支撑段只是提及 Apalachees，与外交任务没有连接。                                                                          |
|   83 | `5a88658955429938390d3f47` | Conscription                                                        | requiring only men to register for the draft | 0.0000 | 答案类型/表达：用 requiring only men to register for the draft 描述判决，未回答征兵方式名 Conscription；不是无依据胡编。                                          |
|   84 | `5a86ebac55429960ec39b6d6` | Mondelez International, Inc.                                        | unknown                                      | 0.0000 | 导航未覆盖：7 次 summary_search、7 次 wiki_tree，0 页证据；Handi-Snacks/Mondelez 的原文均存在。                                                                   |
|   85 | `5aba5d2e55429901930fa799` | Monica Lewinsky                                                     | unknown                                      | 0.0000 | 建库覆盖被覆盖+第二跳遗漏：Their Lives 原文已失去所有知识页入口；旧输出 Monica Lewinsky 页面当前只剩出生日期核心事实，未建立书中人物连接。                        |
|   86 | `5ae5736e5542990ba0bbb2b3` | April 1, 1949                                                       | 1960                                         | 0.0000 | 错误实体/条件未验证：要求律师、游说者、政治顾问，选择了 pollster Tony Fabrizio；原文引用确实有效，但不是题目所指 Paul Manafort。                                  |
|   89 | `5a835478554299123d8c20ed` | 250 million                                                         | unknown                                      | 0.0000 | 导航+问题歧义：已读 Roald Dahl 全部作品销量 250 million，但题目问某篇故事销量；未读目标短篇，也不能把总销量直接当单篇销量。                                       |
|   90 | `5aba749055429901930fa7d8` | director                                                            | film director                                | 0.6667 | 答案表述：film director 对比 director。Aram Avakian 旧来源引用也已丢失，但此题已有足够职业证据。                                                                  |
|   93 | `5ae53b545542990ba0bbb23c` | Las Vegas Strip in Paradise                                         | Las Vegas, Nevada                            | 0.5000 | 第二跳与地点精度：从专辑里的 Las Vegas 直接结束，没有读取 Flamingo 的 Las Vegas Strip in Paradise 具体位置。                                                      |
|   94 | `5ae224da554299234fd043ee` | no                                                                  | unknown                                      | 0.0000 | 导航未覆盖：只读 Gibson，未读 Zurracapote；6 次 summary_search、7 次 tree 后 unknown。标题/轨迹诊断。                                                             |
|   95 | `5ae2b770554299495565db0f` | March and April                                                     | unknown                                      | 0.0000 | 引文修复失败：第 8 步即提出 March and April；拼接跨 facts 引文失败，转原文又不匹配行范围，6 次 finish 失败。                                                      |
|   98 | `5ac1b8ee5542994d76dccedc` | Levni Yilmaz                                                        | Lev Yilmaz                                   | 0.5000 | 别名/全名：Lev Yilmaz 对比 Levni Yilmaz。                                                                                                                         |
