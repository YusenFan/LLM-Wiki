# Wiki schema: tool-driven compilation

Directories grow from actual content. `page_types.yaml` is an optional description
registry, not a mandatory taxonomy. No article sampling defines categories.
The live tree includes all actual directories, including uncommitted files.

## Sources

`sources/versions/<source_id>--<version_id>.md` stores the exact UTF-8 source text.
The companion JSON stores origin identity, title, IDs, path and length.
`source_id = sha256(explicit origin identity)`; `version_id = sha256(original text)`.
Preprocessed benchmark sources use dataset + Wikipedia title as origin identity.
Custom articles should set `source_identity` to a stable URI or upstream ID;
otherwise the absolute input path URI is used. Neither source title nor filename
establishes the identity of an entity in the knowledge graph.

References contain `source_id`, `version_id`, `start`, `end`, and `quote`.
Positions are Unicode character offsets in the archived text, end-exclusive.
The quote must exactly equal that range. Event time belongs in the fact, never
in the source version. Legacy archives remain accessible; exact source links
can be snapshotted and read without guessing origins from similar filenames.

## Knowledge pages

Pages live in content directories chosen by the builder. New page frontmatter:

```yaml
---
title: Alice Example
entity_id: person:alice-example:born-1980
aliases: [A. Example]
type: people
description: Alice Example, researcher born in 1980.
---
```

Each page has one managed JSON fact block, delimited by
`<!-- wiki-facts:start -->` and `<!-- wiki-facts:end -->`.
Existing Markdown outside this block is preserved byte-for-byte by fact updates.
`wiki_read(view="facts")` returns its structured facts; `view="relations"` filters
relationship facts. The default text view keeps the entire legacy page accessible.

Each fact has:

- `id`: computed from the complete qualified statement, independent of citations.
- `statement`, `event_time`, `conditions`, `polarity`, `certainty`, `kind`.
- `citations`: one or more exact original references.
- `conflicts_with`, `supersedes`: existing fact IDs; prior facts remain stored.

`kind` is `fact`, `relation`, or `summary`. Summaries require evidence too and do
not replace original verification. A relation requires two distinct existing
linked pages. There is no arbitrary minimum link count for other facts.

`fact_apply` accepts a page path, identity, expected revision, facts, and links.
For an existing page, its expected revision comes from `wiki_read`; stale updates
fail. New pages require a null revision. The server reads the complete old page,
merges only the supplied increments, validates citations and links, and atomically
replaces the file under a write lock. Same-name entities require disambiguation.
Adding the same qualified fact adds evidence idempotently. Changed claims get new
IDs and explicit conflict/supersession relationships. No tool deletes old claims.

## Read and exploration tools

- `wiki_tree`: all actual directories with descriptions and counts.
- `wiki_read`: directory browsing (paged), text windows, sections, summaries,
  structured facts, or relations. Follow `next_start` for more content.
- `wiki_search`: standard BM25; optional exact-full-entity-first ordering.
- `entity_lookup`: exact complete title/alias lookup; returns all ambiguous candidates
  up to the configured limit, without merging their identities.
- `source_read`: immutable source windows and their exact offsets.

## Build and answer completion

Builds are document scoped. `.build/receipts/` records source versions, products,
status, tool trace, elapsed time, calls, and token usage when returned by the API.
Successful receipts require complete source-read coverage and verified fact products.
Partial writes remain retryable and increments are idempotent. Old SHA-only success
caches are not trusted. Legacy heuristic whole-page repair passes are not called.

The premium Answer Agent explores and submits `finish_answer` with separate answer,
reasoning, citations, optional summary_refs, gaps, and status. Original citations must
have been read. Unknown/conflicting/unfinished results are explicit. The default
and CLI use one QA agent with no delegation. The older explicit Python subtask
option remains for compatibility; it is disabled by default.

## One cross-document summary layer

`summaries/<hash-of-child-paths>.md` is owned exclusively by `SummaryStore`; ordinary
fact writes cannot modify it. Its managed `wiki-summary` block contains level=1,
child page paths/revisions, and 1..8 claims. Each claim has an ID, statement,
direct/synthesis kind, supporting page/fact IDs, and server-derived original citations.
Children must be 2..4 knowledge pages, with evidence from at least two distinct
source identities. Current generation proposes pairs. Source copies, legacy prose,
superseded facts, and summary facts cannot serve as child facts. Explicit conflicts
remain visible. Coverage is selected facts, not an exhaustive summary of every source.

The summary compiler runs after document ingestion, with `SUMMARY_BUILD_LIMIT=20`
model calls by default (0 disables). A separate `build_summaries` CLI resumes pending
groups. Failures remain retryable; skipped groups are cached until their input changes.
Page revisions and the known source-version sets are captured before generation and
checked before committing. Old summary revisions are kept in `.summary-history/`.
Changed/missing child pages, changed source-version sets, or invalid original sources
make a summary stale. Stale summaries return navigation only and are excluded from
search. `current` means dependencies unchanged, not semantic correctness or latest truth.

`wiki_search(layer="summaries"|"details"|"all")` supports both entry points.
`summary_read(path, view="overview")` gives prose and child navigation;
`view="claims"` gives the claim/support map; `view="evidence", claim_ids=[id]`
reveals supporting facts and citations for one claim at a time. No summary read
counts as an original read: `source_read` is still required. Knowledge-page reads
also expose current `related_summaries` as reverse navigation.

`finish_answer.summary_refs` contains `{path, revision, claim_ids}`. Every referenced
claim must have been expanded to evidence, remain current, and have all its supporting
original citation ranges covered by final citations that were actually read.
This checks provenance and coverage, not semantic entailment. The QA agent must
evaluate each reasoning step and report unresolved gaps.
