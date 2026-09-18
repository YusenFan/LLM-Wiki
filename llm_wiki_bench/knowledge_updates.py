"""Append knowledge and its update context without replacing earlier facts."""
import hashlib
import re

from wiki_documents import parse_document, render_document, text_field


def fact_id(text: str) -> str:
    return 'fact_' + hashlib.sha256(text.encode('utf-8')).hexdigest()[:20]


def _sections(body: str) -> dict[str, str]:
    parts = re.split(r'(?m)^## (.+)\n', body)
    sections = {'': parts[0]}
    for i in range(1, len(parts), 2):
        if parts[i] in sections:
            raise ValueError(f'duplicate knowledge section: {parts[i]}')
        sections[parts[i]] = parts[i + 1]
    return sections


def fact_catalog(text: str) -> list[dict]:
    """Stable page-local IDs for existing facts, including pre-ID Markdown pages."""
    _, body = parse_document(text)
    facts = []
    for line in _sections(body).get('Core Facts', '').splitlines():
        if line.startswith('- '):
            claim = re.split(r'\s*\[\[', line[2:], maxsplit=1)[0].strip()
            facts.append({'id': fact_id(claim), 'text': claim})
    return facts


def merge_knowledge(previous: str, rendered: str, proposal: dict) -> str:
    """Preserve old body sections; append facts, sources, links and explained updates."""
    meta, body = parse_document(previous)
    incoming_meta, incoming_body = parse_document(rendered)
    sections, incoming = _sections(body), _sections(incoming_body)
    old_facts = {f['text']: f['id'] for f in fact_catalog(previous)}
    known_ids = set(old_facts.values())
    updates = list(meta.get('knowledge_updates', []))
    for fact in proposal['facts']:
        if fact['text'] in old_facts:
            continue
        change = fact.get('change')
        if not isinstance(change, dict):
            raise ValueError('new facts on an existing page require change: relation, reason, related_fact_ids, valid_at')
        relation = change.get('relation')
        if relation not in {'addition', 'elaboration', 'temporal_update', 'correction', 'conflict'}:
            raise ValueError('invalid fact change relation')
        reason = text_field(change.get('reason'), 'change reason')
        related = change.get('related_fact_ids')
        if not isinstance(related, list) or any(not isinstance(r, str) or r not in known_ids for r in related):
            raise ValueError('change related_fact_ids must refer to existing facts on this page')
        if relation != 'addition' and not related:
            raise ValueError('a non-addition change must identify the earlier related facts')
        valid_at = change.get('valid_at')
        if valid_at is not None:
            valid_at = text_field(valid_at, 'change valid_at')
        record = {'fact_id': fact_id(fact['text']), 'text': fact['text'], 'relation': relation,
                  'related_fact_ids': list(dict.fromkeys(related)), 'reason': reason, 'valid_at': valid_at}
        if record not in updates:
            updates.append(record)

    # Keep the old title/description and every earlier line; no LLM rewrite can remove them.
    for name, content in incoming.items():
        if not name:
            continue
        old_lines = sections.get(name, '').strip('\n').splitlines()
        old_lines = old_lines if old_lines != [''] else []
        for line in content.strip('\n').splitlines():
            if line and line not in old_lines:
                old_lines.append(line)
        sections[name] = '\n'.join(old_lines) + '\n\n'
    if updates:
        meta['knowledge_updates'] = updates
        # The structured records preserve exact old/new fact identities; prose is for navigation.
        notes = []
        for item in updates:
            notes.append(f"- {item['text']} [{item['fact_id']}] ({item['relation']}; related: "
                         f"{', '.join(item['related_fact_ids']) or 'none'}; valid at: "
                         f"{item['valid_at'] or 'not specified'}) — {item['reason']}")
        existing_notes = sections.get('Knowledge Updates', '').strip('\n').splitlines()
        sections['Knowledge Updates'] = '\n'.join(dict.fromkeys(existing_notes + notes)).strip() + '\n\n'
    for field in ('aliases', 'tags'):
        meta[field] = list(dict.fromkeys(meta.get(field, []) + incoming_meta.get(field, [])))
    combined = sections[''].rstrip() + '\n\n' + ''.join(
        f'## {name}\n{content.rstrip()}\n\n' for name, content in sections.items() if name)
    return render_document(meta, combined)
