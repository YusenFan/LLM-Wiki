# Wiki 目录树导航

QA agent 通过文件目录自主选择阅读内容。初始消息包含当前可读页面的两层目录树和标题；需要更多内容时调用 `wiki_tree`。没有 BM25、自定义字段权重、top-k 搜索或自动按分数选文件。

```text
问题 + Wiki 文件目录树
  → LLM 选择路径
  → wiki_tree：展开目录 / 分页（按需）
  → wiki_read：阅读选中的摘要或知识页
  → source_read：读取链接中的原文
  → 同一 agent 更新证据状态、补查或 finish_answer
```

已知路径可以直接阅读，目录工具不是强制前置阶段。这是 Wiki 的文件目录视图，不需要 Git，也不依赖 Git 历史。

## 工具

- `wiki_tree(path="/", depth=2, offset=0, limit=100)`：返回 path、depth、entries、total、next_offset。每项包含 path、type，文件还有 title、layer。目录按路径排序，不提供相关性分数。
- `wiki_read(paths)`：批量阅读选中文件。摘要与知识页是导航；原文文件返回路径与行数，提示调用 source_read。
- `source_read(article, start_line, end_line)`：读取原文，返回实际路径、版本和引文。

`depth` 范围为 1–10，`limit` 为 1–200。大目录可按子目录展开，或使用同一 path/depth 和 next_offset 继续。初始树最多展示 200 项，并明确提示后续分页入口。隐藏构建文件、旧 digest 和失效摘要不进入 QA 目录视图。

`wiki_search`、搜索打分代码、倒排索引，以及 QA 的 `--patience` 和 `--select-pages` 参数已移除。`--t-max`、证据状态和答案提交规则继续使用。

## 文件命名

知识页继续使用现有可读名称，例如：

```text
film/ed_wood.md                      — Ed Wood
film/ed_wood_film.md                 — Ed Wood (film)
summaries/ed-wood-and-ed-wood-film.md — Ed Wood and Ed Wood (film)
```

摘要生成后从 title 生成安全、长度受限的可读文件名。重名的不同成员组使用 `-2`、`-3` 等后缀。成员组身份由 frontmatter 中的 members 确定，缓存新鲜度由 fingerprint 校验；哈希仍可用作内容校验值，但不再作为摘要文件名或匹配信号。

原文归档仍保留 `sources/articles/<content-hash>.md` 文件名及内容版本校验，以保留已有证据链接。目录工具在原文路径旁显示真实标题。此次没有迁移原文文件或改动其内容。

## 已有摘要离线迁移

```bash
python -m llm_wiki_bench.build_summaries \
  --wiki-dir wiki_output/hotpotqa/test-one/wiki \
  --rename-existing
```

无需模型调用。有效哈希摘要按已有标题改名，内容不变；旧文件（包括失效摘要）移到 `.build/legacy-summaries/` 备份，并重建摘要索引。工具会输出原路径到新路径的映射；重复执行不会重复迁移。旧运行日志仍记录历史路径，备份可用于查看其原始摘要。

读取端也能识别尚未迁移的旧摘要，并从页面标题展示名称。构建和缓存以 members/fingerprint 检查摘要身份，不依赖某个固定文件名。

## 验证与边界

`tests/test_tree_navigation.py` 覆盖目录发现、分页、标题展示、原文版本保持、重名摘要、旧摘要迁移、失效摘要隔离和缓存复用。

目录树帮助模型看见同名的不同对象。它不保证模型选对实体，也不证明最终引用支持结论；这些仍需在 QA 结果中分别评估。
