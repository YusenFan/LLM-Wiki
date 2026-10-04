# Wiki 目录树导航

QA agent 保留文件目录浏览能力。默认入口现已改为 [摘要混合检索](summary-hybrid-retrieval.md)：先检索少量 summary，再选择知识页与原文。`wiki_tree` 用于局部浏览和遗漏证据补查。使用 `--summary-mode tree` 可切回原有目录入口做对照；只有该模式在初始消息加载两层目录树。目录列表自身不使用相关性分数。

```text
问题 + 检索所得摘要（或显式 tree 模式的目录树）
  → LLM 选择路径
  → wiki_tree：展开目录 / 分页（按需）
  → wiki_read：阅读选中的摘要或知识页
  → source_read：信息不足或需要核验时，按需读取链接中的原文
  → 同一 agent 更新证据状态、补查或 finish_answer
```

已知路径可以直接阅读，目录工具不是强制前置阶段。这是 Wiki 的文件目录视图，不需要 Git，也不依赖 Git 历史。

## 工具

- `wiki_tree(path="/", depth=2, offset=0, limit=100)`：返回 path、depth、entries、total、next_offset。每项包含 path、type，文件还有 title、layer。目录按路径排序，不提供相关性分数。
- `wiki_read(paths, offset=0)`：在 token 预算内阅读 1–10 个选中文件。返回截断标记和字符 next_offset；续读时只传一个 path。摘要用于导航，知识页可直接支撑答案；原文文件返回路径与行数，提示按需调用 source_read。
- `source_read(article, start_line, end_line)`：读取原文，返回实际路径、版本和引文。

`depth` 范围为 1–10，`limit` 为 1–200。大目录可按子目录展开，或使用同一 path/depth 和 next_offset 继续。tree 对照模式的初始树最多展示 200 项，并明确提示后续分页入口。隐藏构建文件、非原文来源页和失效摘要不进入 QA 目录视图。

旧 `wiki_search`、自定义字段打分，以及 QA 的 `--patience` 和 `--select-pages` 参数已移除。新的 `summary_search` 只检索有效摘要，使用标准 BM25/dense/RRF。`--t-max`、证据状态和答案提交规则继续使用。

## 文件命名

知识页继续使用现有可读名称，例如：

```text
film/ed_wood.md                      — Ed Wood
film/ed_wood_film.md                 — Ed Wood (film)
summaries/ed-wood-and-ed-wood-film.md — Ed Wood and Ed Wood (film)
```

已有摘要的成员组身份由 frontmatter 中的 members 确定，新鲜度由 fingerprint 校验；文件名不参与检索评分。summary page 生成代码已移除，新 Wiki 使用 `--summary-mode tree`。

原文按 main 的标题规则保存为 `sources/articles/<source-title>.md`，重新包装 source frontmatter 和 H1。同标题不同内容用数字后缀保留。知识页直接链接到 `sources/articles/<source-title>`，需要原文证据时使用 source_read。sources 下仅原文目录进入检索；内部内容版本校验仍保留。现有 Wiki 输出不会自动迁移，旧版本需在新目录重建以更新来源链接。

## 已有摘要读取

读取端保留已有可读名称及旧哈希名称摘要的兼容能力，从页面标题展示名称。
`summary_catalog.current_summaries()` 只检查 members/fingerprint，不改名、不生成页面。
原来的生成和 `--rename-existing` 命令已随 `build_summaries.py` 删除。

## 验证与边界

`tests/test_tree_navigation.py` 覆盖目录发现、分页、标题展示、原文版本保持、旧哈希摘要读取、重复摘要选择、失效摘要隔离和只读行为。

目录树帮助模型看见同名的不同对象。它不保证模型选对实体，也不证明最终引用支持结论；这些仍需在 QA 结果中分别评估。
