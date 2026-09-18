"""IDs address exact, actually delivered text; models never assemble citations."""
import hashlib
import json
import re


def evidence_id(citation: dict) -> str:
    return 'ev_' + hashlib.sha256(json.dumps(citation, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def page_units(body: str) -> list[tuple[int, int]]:
    """Citable description/fact/prose ranges; relationships and update notes are navigation."""
    units = []
    section = ''
    for match in re.finditer(r'(?m)^.+$', body):
        line = match[0]
        if line.startswith('## '):
            section = line[3:].strip().casefold()
            continue
        if line.startswith('#') or section not in {'', 'core facts'}:
            continue
        prefix = 2 if line.startswith(('- ', '> ')) else 0
        text = line[prefix:].split('[[', 1)[0].rstrip()
        if text.strip():
            start = match.start() + prefix
            units.append((start, start + len(text)))
    return units


def decorate_page(row: dict, units: list[tuple[int, int]], version: str) -> dict:
    start = row.get('start_offset', 0)
    end = start + len(row['text'])
    spans = []
    for left, right in units:
        left, right = max(left, start), min(right, end)
        if left >= right:
            continue
        quote = row['text'][left-start:right-start]
        if not quote.strip():
            continue
        citation = {'page': row['path'], 'page_version': version, 'start_offset': left,
                    'end_offset': right, 'quote': quote}
        spans.append({'evidence_id': evidence_id(citation), 'text': quote,
                      'start_offset': left, 'end_offset': right})
    return {**row, 'page_version': version, 'evidence': spans}


def decorate_article(passage: dict) -> dict:
    spans = []
    for line, text in enumerate(passage['quote'].split('\n'), passage['start_line']):
        # Exact line excerpts, including legacy single-line Hotpot paragraphs.
        if not text.strip():
            continue
        citation = {k: passage[k] for k in ('article', 'version')}
        citation.update(start_line=line, end_line=line, quote=text)
        spans.append({'evidence_id': evidence_id(citation), 'text': text,
                      'start_line': line, 'end_line': line})
    return {**passage, 'evidence': spans}


def register_page(row: dict, registry: dict) -> None:
    for span in row.get('evidence', []):
        citation = {'page': row['path'], 'page_version': row['page_version'],
                    'start_offset': span['start_offset'], 'end_offset': span['end_offset'], 'quote': span['text']}
        registry[span['evidence_id']] = citation


def register_article(passage: dict, registry: dict) -> None:
    for span in passage.get('evidence', []):
        citation = {k: passage[k] for k in ('article', 'version')}
        citation.update(start_line=span['start_line'], end_line=span['end_line'], quote=span['text'])
        registry[span['evidence_id']] = citation


def resolve_evidence(ids: object, registry: dict) -> list[dict]:
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
        raise ValueError('evidence_ids must be a nonempty list of IDs returned by wiki_read/source_read')
    missing = [x for x in ids if x not in registry]
    if missing:
        raise ValueError(f'unread or unknown evidence_ids: {missing}; select IDs from this question\'s read results')
    return [dict(registry[x]) for x in dict.fromkeys(ids)]
