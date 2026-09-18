# Wiki Research Direction

## Core Positioning

Structured knowledge base compiled from Wikipedia paragraphs in a multi-hop QA
benchmark (HotpotQA / MuSiQue / 2WikiMultiHopQA). Used as the retrieval backend
for downstream multi-hop question answering.

## Key Knowledge Domains

General encyclopedic knowledge: people, organizations, places, events,
artistic and literary works, scientific concepts, and the relations between
them.

## Ingestion Focus

- Priority extraction: key facts, entity relationships, temporal information,
  causal connections.
- Moderate extraction: background context, categorical information.
- Reuse existing fact wording for redundant information and retain every supporting input article citation. Python preserves earlier facts and appends additions; never replace old facts. Explain each new fact's relationship to earlier knowledge, including source-supported time changes. Each input article must support at least one output fact.

## Target Use Case

Multi-hop question answering — questions that require combining information
from multiple paragraphs or entities to derive the answer.
