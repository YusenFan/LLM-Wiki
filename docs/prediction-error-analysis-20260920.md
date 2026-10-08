# Prediction 错题分析（2026-09-20）

## 范围与复现

主分析对象：`wiki_output/hotpotqa/first-500/predictions.jsonl`，标准答案：`data/hotpotqa/qa_pairs.jsonl`。分析完全离线，使用预测时保存的证据和轨迹，没有调用模型、修改预测或重跑 QA。以下事实判断均基于本地快照，未进行外部事实核验。

```bash
python3 -m llm_wiki_bench.analyze_predictions \
  --predictions wiki_output/hotpotqa/first-500/predictions.jsonl \
  --qa-pairs data/hotpotqa/qa_pairs.jsonl \
  --output-dir results/hotpotqa/error_analysis
```

输出目录中的 `wrong_questions.csv` 便于筛选；`wrong_questions.jsonl` 保留每道题的完整原始 prediction、证据和工具调用；`report.md` 列出全部不匹配问题及诊断建议；`summary.json` 提供统计；`missing_predictions.jsonl` 单列未预测的问题。

另一份历史运行 `wiki_output/hotpotqa/test-ten-20260919T075323Z/predictions.jsonl` 也已独立分析，输出在 `results/hotpotqa/error_analysis_test_ten/`。10 题中 3 题 EM 不匹配，EM 70%，F1 80.50%。两次运行题目有重叠，不合并统计。

## 主要结论

500 题中 315 题 EM 匹配，185 题不匹配，EM **63.00%**，F1 **79.10%**，无漏预测或重复 ID。全部错题 ID 和分数与现有 `evaluate.py` 一致。这里“错题”严格指 **EM=0**，不等于 185 题全都事实错误。

| 互斥自动诊断信号 | 题数 | 含义 |
|---|---:|---|
| 预测包含标准答案片段 | 55 | 可能多解释、多枚举、额外限定，也可能答错范围 |
| 预测是标准答案的片段 | 38 | 可能简称、单位或限定缺失，也可能标准答案冗余 |
| 部分词项重叠 | 37 | 可能别名、同义表达、不同数值或不同实体 |
| 无词项重叠 | 34 | 可能实体/关系错误，也可能完全不同的别名 |
| 主动提交 unknown | 10 | 可能证据不足、歧义或问题前提冲突 |
| 布尔答案类型不匹配 | 2 | 返回 yes/no，gold 要求实体/短语 |
| 模型调用中断 | 6 | 日志确认均为输出截断 |
| 检索预算耗尽 | 3 | 需要连同最后一次提交校验失败检查 |

这不是人工语义根因占比。185 题中 130 题 F1>0，55 题 F1=0；不能把前者全部认作“格式错误”、后者全部认作“事实错误”。桥接题 404 题中 157 题不匹配（38.86%），比较题 96 题中 28 题不匹配（29.17%）。

另有三个交叉信号：51 道错题未读齐 gold supporting titles，134 道已读齐；164 道已读证据含 gold 的归一化文字片段；20 道出现过工具错误。它们相互重叠。读到标题/答案片段不能证明已具备全部推理条件；未读 gold 标题也不能证明缺证据，知识页可能合并了多个来源。

## 已核实的原因与典型案例

### 1. 答案抽取范围、别名和粒度差异

- `5a8c7595554299585d9e36b6`：gold 是 `Chief of Protocol`，预测列出两个大使职位及礼宾司长。证据已找到，主要是没有收敛到目标答案范围；问题本身也未明确限定唯一政府职务。
- `5a87ab905542996e4f3088c1`：`3,677 seated` 对 `3,677`，数值一致，单位/限定词导致 EM 不匹配。
- `5ae819ac55429952e35eaa16`：`National Broadcasting Company` 对 `NBC`，F1=0 也可能只是简称差异。
- `5ac32b565542995ef918c154`：问球场位置，gold 是 `North Avenue at Techwood Drive`，预测 `Atlanta, Georgia`；已引用证据同时包含街道和城市，答案选择的地点层级不同。
- `5ade0fb95542997545bbbe39`：gold `2844 km`，预测 `1472 km`；Darling River 知识页明确同时包含“河流本身”和“包含最长连续支流”的长度。是统计口径/问题歧义，不能当作凭空编造数字。

优化：在生成前明确目标类型与粒度，输出最短完整答案片段，解释放在 evidence_chain。别名以来源中的规范名和别名支持，不应凭 gold 反向定制输出。当前 `wiki_agent.py` 和 `qa_contract.py` 已有 shortest complete answer 指令，所以再加一句“简短回答”不足以证明有效；应做答案抽取阶段的独立实验。

### 2. 证据存在，但问题对象或关系处理错误

- `5a828c8355429966c78a6a50`：两跳引用均明确是 Henry J. Kaiser，却回答 `yes`。问题语法不完整，模型选错答案类型。
- `5ae75a4c5542997b22f6a6f1`：问节目，回答主持人 `Krusty the Clown`；引用中已经出现 gold 节目名。至少可以确认输出对象类型不符合问题。
- `5ade69e455429975fa854ec5`：问与 Vikram Bhatt 合作的导演，回答 Vikram Bhatt 本人；轨迹引用只证明某片由他导演，没有证明“与他合作”的关系。仍需复核 gold 对应影片。

`qa_contract.validate_answer` 只验证引用是否来自已读快照、版本和范围是否匹配，不校验引用是否在语义上支持 claim，也不证明最终答案满足问题。因此 `citations_validated` 不能当作“推理正确”。

优化：将问题结构化为“主体 → 关系 → 所求属性/对象”；提交前检查答案类型，以及每个 requirement 是否真正回答该子问题。比较题显式抽取两个值再比较。可先增加一个只看问题、候选答案和已读证据的轻量检查步骤，衡量增益与额外成本，不能用 gold 做运行时校验。

### 3. 检索链断裂、重复搜索与输入歧义

- `5ae6050f55429929b0807a5e`：已经读到歌曲由 Jim Cummings 演唱，但未读到 Jim Cummings 页面，第二跳 unresolved 后提交 unknown。当前 wiki 未找到文件名包含 cummings 的页面；这只是覆盖/索引缺口线索，不能仅凭文件名断言原文不存在。
- `5a75e05c55429976ec32bc5f`：题目写 `country`，gold 支撑标题却是 `Brown County, Kansas`。requirements 将其解释为 United States 的人口，后续搜索沿国家人口走偏。这是输入拼写歧义造成的问题解析与检索路径偏移。
- `5a85fb085542994775f606de`：gold 支撑 Alien，系统选择 Lionheart，并给出该片的执行制片人；现有引用支持这条替代链。问题只给出 Jerry Goldsmith 配乐，不能唯一确定电影。属于跨全库候选歧义，需要复核数据设定，不能直接宣称模型捏造。
- `5abcc96c5542996583600492`：重复搜索 Planet Terror 和遍历 media，最终 `finish_answer` 因 `evidence_ids must be a nonempty list` 失败后耗尽预算。需要处理检索无进展与提交失败共同作用。

优化：对桥接实体提供确定性标题/别名查找和直接原文定位，摘要搜索无新增证据时切换策略。按页/证据 ID 去重，记录连续无新增证据次数；对可能拼写混淆可生成两个候选解释并用证据消歧。单纯增大 t_max 可能只延长重复搜索。对预算末尾的提交增加本地预检，保留校验失败后的修复机会。

### 4. 6 次模型失败均是输出截断，不是网络失败的猜测

`qa.log` 六处 `Tool-call LLM failed` 全部是 `LLM output was truncated; increase the output token budget`，随后出现 `agent LLM call failed; stopping`：日志行 1091、1255、1294、1660、2835、2929。对应主运行第 179、205、210、268、471、488 题，均返回 unknown。

`llm_client.call_llm_with_tools` 的输出预算默认来自 `LLM_TOOL_MAX_TOKENS`，未配置时为 2048；这里只确认代码默认值，没有据此断言本次运行实际环境值。`wiki_agent.py` 收到 None 后标记 model_error，但 run_qa 保存的 error 仍为 null，详细失败原因丢失。

优化：持久化具体错误类型、finish reason、实际输出预算和重试次数；对明确截断有限增加输出预算重试，压缩重复 requirement/claim；提供按失败 ID 恢复能力。优先重跑这 6 题，最多涉及当前 EM 的 1.2 个百分点，但并非承诺能全部转对。

### 5. 标注/题目质量与全库歧义

以下是根据本地证据判定的待复核案例，不修改基准分数：

- `5ab9b29c554299743d22ebae`：问哪个案件更早。引用为 Paramount 1948、Craig 1976，模型选前者，gold 却为 Craig。现有证据支持模型的时间比较。
- `5a879ba55542993e715abfc3`：问谁导演生涯更长。Resnais 超过 60 年，Sidney 1913–1927；预测 Resnais，gold Sidney。现有证据支持模型。
- `5ab7f0015542992aa3b8c88b`：问 Star & Dagger 贝斯手嫁给谁。引用指出贝斯手 Sean Yseult 与 Chris Lee 结婚；预测 Chris Lee，gold 却是 Sean Yseult，疑似问答关系方向不一致。
- `5ac298f9554299657fa28fc9`：问两个人中谁入狱，预测 Ian Watkins，gold 是乐队解散的整句描述，答案类型不一致。
- `5abbf698554299114383a0b5`：题目把 WWII 战场与 Canberra 混在一起，知识页说明 Canberra 1951 年服役，相关中队在另一个时期使用该机。unknown 可能来自合理识别前提冲突。

优化：保留原 EM/F1，与“经人工复核的语义等价率/歧义率”并列报告；别名、标注冲突和真实错误分别记录，不能通过修改 gold 偷换基准。先复核 F1=0 的 55 题，再复核片段差异中可能遗漏实体和条件的题目。

## 建议的实施顺序与验证

1. **运行可靠性**：记录并恢复 6 个截断失败；单独测有限扩预算重试，统计恢复率与 token 成本。
2. **答案抽取与类型校验**：固定原有检索证据，只替换最后答案抽取，观察 EM/F1 和语义正确率；重点关注 93 个互相包含的答案，以及两道布尔类型错误。93 不代表 93 个都能靠缩短答案修好。
3. **第二跳检索与提交修复**：针对 unresolved、重复检索、预算耗尽案例加标题/别名和原文回退，统计新增有效证据率、循环次数及工具错误率。
4. **评估复核**：建立有证据依据的别名/歧义标记；保留基准原分，避免用题目 ID 或 gold 驱动运行时逻辑。

各方案先做单变量实验。当前 500 题已用于诊断，优化效果应在独立保留集确认，并同时报告成本、耗时、unknown 比例及正确题退化情况。没有实际重跑前，不给出预期准确率提升的确定承诺。

## 当前实现的验证

新增脚本只依赖 Python 标准库和项目评分函数。5 个单元测试覆盖别名、缺失/额外预测、运行失败、遥测缺失、完整记录导出、重复 ID 与损坏 JSONL 行号、QA 文件不匹配。主运行 500 题和历史 10 题均通过与现有 evaluator 的完整错题 ID 集合、EM、F1 对比。
