# 增量知识、singleton 导航与证据 ID

## 保留旧知识

`bench_ingest._validate_proposal()` 先校验模型新增内容，再调用 `knowledge_updates.merge_knowledge()` 合并到同一页。Python 保留已有正文、Core Facts、来源链接、Related Pages 和旧元数据；aliases/tags 取并集。模型遗漏旧事实不会删除它。已有事实的稳定 ID 由事实文本生成，在生成上下文中与原文一起提供。

新事实在更新已有页时必须附带 `change`，例如：

```json
{
  "text": "In 2010, Alpha moved to Berlin.",
  "citations": [{ "article": "sources/articles/HASH.md" }],
  "change": {
    "relation": "temporal_update",
    "related_fact_ids": ["fact_ID_OF_EARLIER_RESIDENCE"],
    "reason": "The source describes a move in 2010 after the earlier residence in Paris.",
    "valid_at": "2010"
  }
}
```

`relation` 可为 addition、elaboration、temporal_update、correction、conflict。除了独立 addition，其余必须关联本页已有事实 ID。reason 必填；valid_at 只填写来源明确支持的时间，否则 null，不能用入库时间替代事实时间。旧的“2000 年住在巴黎”仍保留；更正或矛盾也只增加说明。结构化记录保存在 frontmatter 的 `knowledge_updates`，可读原因追加到正文 `Knowledge Updates`。关联 ID 和必填字段由 Python 校验，关系、原因和时间是否符合来源仍需模型判断。

合并前完成整批校验；坏更新不写知识页。重复完全相同的事实不会重复添加变化记录；不同措辞不会由 Python 擅自认定为同一事实。旧来源保留，因此后续更新不再因移除旧引用使原来的成功 receipt 失效。

这项修改保护后续更新，不会自动找回此前已经被覆盖的事实。历史缺失需要从不可变原文存档重新入库；如果旧 receipt 仍有效却遗漏部分事实，需单独检查或有针对性重建。

## Singleton summary

分组仍是每页与显式 Related Pages 的集合，去重并删除严格子集。保留剩下的单页组，使所有知识页都属于至少一个组。单页摘要由 Python 生成，不调用模型；多页组沿用原来的 LLM 摘要。

只补齐已有 Wiki 的孤立页面：

```bash
source .venv/bin/activate
python -m llm_wiki_bench.build_summaries \
  --wiki-dir wiki_output/hotpotqa/test-ten/wiki \
  --singletons-only
```

该命令无需加载 API 凭证。它保留有效多页摘要，报告尚未构建的多页组为 pending。检查 `covered_pages == knowledge_pages` 且 `uncovered_pages == []` 才能断言本次 Wiki 覆盖完整。只有完整构建成功且无限额时才能保证覆盖全部知识页。知识页修改后重建 summaries，再启动新的 QA 进程加载新快照。

## 模型选证据，Python 生成引用

1. `wiki_read` 为实际返回的知识页事实/说明片段生成 ID；`source_read` 为实际返回的非空原文行生成 ID。summary、目录和关系说明仅用于导航。
2. `WikiAgent` 将这些 ID 与精确文本、版本、路径、字符或行范围登记到本题独立的 `evidence_registry`。
3. 模型在 requirements 与 evidence_chain 中选择 `evidence_ids`。claim 可以概括推理关系，不再承担精确复制引文的责任。
4. Python 查表生成 citations，再对实际已读快照校验版本、范围和文本；未知 ID、未读页面及未读的截断部分均被拒绝。不同问题不会共享已读权限。
5. 输出保存 `evidence_snapshots`，以及包含 evidence_ids 和生成 citations 的 evidence_chain，便于回放核验。

例如一个答案依赖同页两个不相邻事实，模型选择两个 ID 即可，无需拼接一段不存在的连续 quote。知识页证据充分时仍无需读取原文。确定性出处校验不能证明事实正确或推理成立，也不会自动解决检索遗漏。

离线回归：

```bash
.venv/bin/python -m unittest discover -s tests -q
```

`tests/test_incremental_evidence.py` 覆盖旧知识保留、时间关联、receipt 有效性、singleton 无模型生成与失效切换、非连续事实、快照版本及未读证据拒绝；QA loop 测试覆盖实际工具提交和截断续读。
