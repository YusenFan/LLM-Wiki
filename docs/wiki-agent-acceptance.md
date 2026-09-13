# Tool-driven Wiki acceptance contract

Before implementation, the following regression gates are required:

- Full original input survives archival, including text beyond the old 15,000-character cutoff; identity and content version are independent.
- Distinct titles that sanitize to one filename and changed text under one title survive preprocessing.
- Source references use immutable source/version IDs and exact Unicode character offsets `[start, end)` plus matching quotes. Event time is a separate fact field.
- A local update preserves every unrelated fact and all legacy Markdown; stale revisions, traversal paths, invalid links and invalid quotations fail before writing.
- The filesystem tree and reads see newly created/uncommitted pages; YAML block lists and whole-page Markdown fences parse correctly.
- BM25 has no substring bonuses; exact entity lookup remains explicit and ambiguous identities remain separate. Retrieval experiments expose k=5/10/15.
- Failed/partial document builds remain retryable; cached success requires existing, verified products and complete original-read coverage.
- Answer exploration can request additional evidence, suppress repeated calls, respect a shared subtask budget, finish with a premium model, and report evidence gaps/citations separately from the short benchmark answer.
- Repair is deterministic with backups; ambiguous legacy links are reported, never guessed. Offline scripted LLM tests establish protocol behavior; live quality/cost claims require measured runs.

One-layer summary acceptance:

- Summary groups use at least two distinct original identities and source-backed current knowledge facts; source duplicates, invented supports, and recursive summary children are rejected.
- Summary overview/claim reads do not disclose original quotes; selected claim evidence is revealed separately, then verified with source_read in one QA context.
- Source/child changes invalidate summaries at search/read/answer time. Missing dependencies and malformed summaries do not prevent detail search. Rebuilds preserve old summary revisions.
- Final summary references require prior expansion and original citations covering every support; a missing hop or stale reference is rejected.
- Summary generation runs after ingestion, is bounded and retryable, caches unchanged outputs/rejections, exposes pending groups and usage, and refuses to elevate uncited legacy prose.
- CLI QA never enables delegation. Summary-free precise searches still work.
