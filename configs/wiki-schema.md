# Wiki schema: article evidence and related-page summaries

Knowledge facts retain links to their **processed articles**; QA may cite sufficiently
explicit knowledge-page evidence directly and read articles for additional detail or verification:

```text
summaries/ (navigation, tags, member knowledge pages)
    → knowledge pages (facts, explained Related Pages)
        → sources/articles/ (direct article links)
```

## Directories

- `sources/articles/<sha256>.md`: byte-for-byte copies of processed articles.
  The filename is the full content hash; changed text creates a different file.
- Knowledge directories declared in `page_types.yaml`: e.g. `entities/`,
  `concepts/`, `events/`, `relations/`.
- `summaries/<readable-title>.md`: one level of high-level navigation summaries.
  Duplicate titles use numeric suffixes; membership and freshness are checked
  from frontmatter, not filenames.
- `_index.md` files and root `index.md`: deterministic navigation lists.

`sources`, `summaries`, and `syntheses` are reserved names. No new digest pages
are generated. Article evidence does not need a link to pre-article raw material.

## Knowledge pages

Python renders the page from a validated JSON proposal:

```markdown
---
type: entities
schema: article-evidence-v1
aliases: [A]
tags: [history]
---

# Alpha

> Alpha and its organization

## Core Facts

- Alpha founded Beta. [[sources/articles/<sha256>]]

## Related Pages

- [[entities/beta]] — organization founded by Alpha

## Related Sources

- [[sources/articles/<sha256>]]
```

Every fact has at least one article citation. Citation input includes only the
`.md` article path. Python checks that it resolves to a nonempty archived article
and that its content matches its hash. The model does not choose a fragment,
count line numbers, or reproduce a quote during knowledge-page generation.
Python reads the actual article content; it never persists model-written quotes
as source text. Existing line-range links remain readable for compatibility.

Related Pages can have zero, one or many reliable links. Each target must be an
existing knowledge page or a knowledge page created in the same batch; each link
has a short relationship description. Do not manufacture links to meet a quota.

Updates are append-only on the same knowledge page. Python preserves previous
facts, sources, relationships and metadata, and unions aliases/tags. Each new fact
on an existing page includes `change: {relation, reason, related_fact_ids, valid_at}`.
Relations are `addition`, `elaboration`, `temporal_update`, `correction` or `conflict`.
Non-additions must identify earlier facts from that page's supplied fact catalog;
all changes need a reason. `valid_at` is a source-supported time or null, never an
inferred ingestion timestamp. Changes are recorded in `knowledge_updates` metadata
and a readable `Knowledge Updates` section; even corrected facts remain available.

## Summary grouping

For every knowledge page, take `{itself} ∪ {existing Related Pages targets}`.
Only explicit Related Pages entries with relationship descriptions participate.
Articles, summaries, indexes and other links do not participate.

1. Include singleton sets for isolated pages.
2. Deduplicate identical sets regardless of member order.
3. Remove a set if it is a strict subset of **one** other candidate set.
4. Keep overlapping sets when neither contains the other.
5. Do not compute connected components or require a second topic classifier.

Example: retain `{A,B,C}` and `{A,B,D}`; omit `{A,B}`. If only A and B link to
each other, produce one `{A,B}` summary. Coverage by a union of several groups
is not sufficient to remove another group.
An isolated `{C}` remains as a singleton group. Every knowledge page belongs to
at least one group; actual summary coverage also requires those groups to build successfully.

## Summary pages

```markdown
---
type: summary
schema: related-summary-v1
title: Alpha and Beta
tags: [history, organizations]
members: [entities/alpha.md, entities/beta.md]
fingerprint: <hash of member paths and current content>
---

# Alpha and Beta

> Founding and location

A high-level overview of the supplied knowledge pages.

## Member Pages

- [[entities/alpha]]
- [[entities/beta]]

Navigation only. Read member knowledge pages for answers; consult original articles when needed.
```

Python owns membership and deduplication. For groups of two or more pages, the LLM
writes title, description, tags and overview. Singleton navigation pages are generated
by Python from the member page, with no model call. `generation` metadata records
`python-singleton-v1` or `llm-related`. A small fingerprint check avoids serving a cached summary
when its membership or member contents have changed; no history/repair agent
is introduced. Obsolete files may remain on disk but are excluded from retrieval
and regenerated summary indexes.

## QA contract

- Initial navigation retrieves current summaries with BM25/dense/RRF (default hybrid),
  at most 5 summaries within a 4000-token serialized payload budget. This is navigation, not proof.
- `summary_search(query, limit?, offset?, exclude_paths?)` allows missing-fact requery,
  candidate paging, and exclusion of already-read summaries. Only current summaries are indexed;
  directory traversal remains available for pages with no summary or omitted summary facts.
- `wiki_tree(path?, depth?, offset?, limit?)` lists unranked directories and files
  with readable titles. Expand a subdirectory or continue with `next_offset`.
  Tree listings do not use relevance scores; summary ranking has no custom field bonuses.
- `wiki_read(paths, offset?)` opens 1-10 pages within the same token budget,
  including evidence IDs and text for knowledge-page facts and descriptions.
  A truncated text returns `next_offset`, a character offset into the Markdown body;
  continue with one path. For an article it returns the path
  and line count, prompting `source_read` rather than treating navigation as proof.
- `source_read(article, start_line, end_line)` returns actual article lines,
  the version read and evidence IDs. A call reads up to 200 lines, with an 80-line default window.
- The same agent maintains evidence requirements and submits a short answer
  through `finish_answer`, with `evidence_ids` in requirements and each evidence-chain hop.
  Python constructs citations from the question-local registry of delivered knowledge-page
  and article snapshots. The model does not write quotes, paths, versions or ranges.
  Summaries, directories and relationship descriptions do not supply evidence IDs.
- Python verifies that citations lie inside read passages with matching versions
  and exact quotes. Invalid submissions return a tool error for correction within
  budget; failure to submit a validated answer leaves `unknown`.

Valid ranges and exact quotes establish provenance. They do **not** establish
that the source entails the claim, or that the model listed every necessary hop.
Those remain semantic judgments requiring evaluation.
