"""Deterministic citation checks shared by QA entry points. Not semantic verification."""

from wiki_documents import resolve_wiki_link


def validate_answer(proposal: dict, evidence: list[dict], page_evidence: list[dict] | None = None) -> dict:
    """Check every submitted citation against passages read, without claiming semantic proof."""
    if not isinstance(proposal, dict) or not isinstance(proposal.get("answer"), str):
        raise ValueError("answer must be a JSON object with an answer string")
    answer = proposal["answer"].strip()
    if not answer:
        raise ValueError("answer cannot be empty")
    if answer.casefold() == "unknown":
        return {"prediction": "unknown", "evidence_chain": [], "evidence_status": "insufficient"}
    chain = proposal.get("evidence_chain")
    if not isinstance(chain, list) or not chain:
        raise ValueError("a factual answer requires knowledge-page or article evidence for each hop")
    known_paths = {passage["article"] for passage in evidence}
    page_evidence = page_evidence or []
    known_pages = {passage["page"] for passage in page_evidence}
    normalized_chain = []
    for hop in chain:
        if not isinstance(hop, dict) or not isinstance(hop.get("claim"), str) or not hop["claim"].strip():
            raise ValueError("each hop requires a claim")
        citations = hop.get("citations")
        if not isinstance(citations, list) or not citations:
            raise ValueError("each hop requires knowledge-page or article citations")
        normalized_citations = []
        for citation in citations:
            if not isinstance(citation, dict):
                raise ValueError("citation must be an object")
            if "page" in citation:
                if any(key in citation for key in ("article", "version", "start_line", "end_line")):
                    raise ValueError("do not mix knowledge-page and article citation fields")
                page = resolve_wiki_link(citation["page"], known_pages)
                quote = citation.get("quote")
                if not isinstance(quote, str) or not quote.strip():
                    raise ValueError("knowledge-page citation requires a nonempty exact quote")
                if not any(p["page"] == page and quote in p["text"] for p in page_evidence):
                    raise ValueError("knowledge-page citation is unread or quote does not match a wiki_read excerpt")
                normalized_citations.append({"page": page, "quote": quote})
                continue
            citation = {**citation, "article": resolve_wiki_link(citation.get("article", ""), known_paths)}
            start, end = citation.get("start_line"), citation.get("end_line")
            if type(start) is not int or type(end) is not int or end < start:
                raise ValueError("citation requires an inclusive integer line range")
            valid = False
            error = "answer cites an unread article or changed version; use the article and version returned by source_read"
            for passage in evidence:
                if (citation.get("article") == passage["article"]
                        and citation.get("version") == passage["version"]):
                    if not passage["start_line"] <= start <= end <= passage["end_line"]:
                        continue
                    lines = passage["quote"].split("\n")
                    quote = "\n".join(lines[start - passage["start_line"]:end - passage["start_line"] + 1])
                    valid = bool(quote.strip()) and quote == citation.get("quote")
                    if valid:
                        break
                    error = (f"quote mismatch for {citation['article']} L{start}-L{end}: "
                             "quote must match the complete cited lines exactly, including any headings "
                             "or metadata in that range. Copy the full lines from source_read, or cite "
                             "a narrower already-read line range; do not shorten or paraphrase the quote.")
            if not valid:
                same_version = [p for p in evidence if p["article"] == citation["article"]
                                and p["version"] == citation.get("version")]
                if same_version and not any(p["start_line"] <= start <= end <= p["end_line"] for p in same_version):
                    error = f"answer cites an unread line range L{start}-L{end}; use a range supplied by source_read"
                raise ValueError(error)
            normalized_citations.append(citation)
        normalized_chain.append({**hop, "citations": normalized_citations})
    return {"prediction": answer, "evidence_chain": normalized_chain, "evidence_status": "citations_validated"}


ARTICLE_CITATION_SCHEMA = {
    "type": "object",
    "properties": {
        "article": {"type": "string"}, "version": {"type": "string"},
        "start_line": {"type": "integer"}, "end_line": {"type": "integer"},
        "quote": {"type": "string"},
    },
    "required": ["article", "version", "start_line", "end_line", "quote"],
}
CITATION_SCHEMA = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "page": {"type": "string", "description": "Knowledge-page path actually read with wiki_read; not a summary or article."},
                "quote": {"type": "string", "description": "Exact supporting text copied from a returned wiki_read excerpt."},
            },
            "required": ["page", "quote"],
            "additionalProperties": False,
        },
        ARTICLE_CITATION_SCHEMA,
    ],
}
REQUIREMENTS_SCHEMA = {
    "type": "array",
    "description": "Upsert requirements by id. Include every hop or compared entity; omitted ids remain. Empty list means no updates.",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"}, "question": {"type": "string"},
            "status": {"type": "string", "enum": ["unresolved", "supported"]},
            "citations": {"type": "array", "items": CITATION_SCHEMA},
        },
        "required": ["id", "question", "status", "citations"],
    },
}
FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish_answer",
        "description": "Submit a short answer supported by read knowledge pages or optional original article passages, or unknown if evidence is unavailable. Validation errors allow further retrieval within budget. Requirement updates are applied before submission checks.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Only the shortest complete answer: exactly lowercase yes/no for yes/no questions; otherwise the requested name, place, date, number, or brief phrase. Include necessary units and all requested items. No explanation, question restatement, or citations; put support in evidence_chain. Use unknown if evidence is unavailable.",
                },
                "requirements": REQUIREMENTS_SCHEMA,
                "evidence_chain": {
                    "type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "requirement_id": {"type": "string"},
                            "claim": {"type": "string"},
                            "citations": {"type": "array", "items": CITATION_SCHEMA},
                        },
                        "required": ["requirement_id", "claim", "citations"],
                    },
                },
            },
            "required": ["answer", "requirements", "evidence_chain"],
        },
    },
}
QA_CONTROL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_evidence_state",
            "description": "Record unresolved subquestions or support them with exact read knowledge-page or article citations. This is a factual task state, not a reasoning transcript.",
            "parameters": {
                "type": "object", "properties": {"requirements": REQUIREMENTS_SCHEMA},
                "required": ["requirements"],
            },
        },
    },
    FINISH_TOOL,
]
