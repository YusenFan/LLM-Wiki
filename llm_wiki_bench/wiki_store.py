"""Immutable source versions and atomic, revision-checked fact increments.

Offsets are Unicode character offsets in the exact archived input, end-exclusive.
Managed facts live inside each Markdown page, so one atomic rename commits a page.
Legacy content outside the managed block is never rewritten by fact mutations.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import yaml

class MetadataLoader(yaml.SafeLoader):
    # Source dates are data, not Python date objects (nor evidence versions).
    yaml_implicit_resolvers = {
        key: [(tag, pattern) for tag, pattern in entries if tag != 'tag:yaml.org,2002:timestamp']
        for key, entries in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }


START = '<!-- wiki-facts:start -->' 
END = '<!-- wiki-facts:end -->'
BLOCK = re.compile(re.escape(START) + r'\n```json\n(.*?)\n```\n' + re.escape(END), re.S)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def unwrap_markdown(text: str) -> str:
    match = re.fullmatch(r'\s*(`{3,}|~{3,})(?:markdown|md)?\s*\n(.*)\n\1\s*', text, re.S)
    return match.group(2) + '\n' if match else text


def frontmatter(text: str) -> tuple[dict, str]:
    text = unwrap_markdown(text)
    match = re.match(r'\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|$)', text, re.S)
    if not match:
        return {}, text
    meta = yaml.load(match.group(1), Loader=MetadataLoader) or {}
    if not isinstance(meta, dict):
        raise ValueError('frontmatter must be a mapping')
    return meta, text[match.end():]


def safe_path(root: Path, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or any(x.startswith('.') for x in p.parts):
        raise ValueError('path must be relative and may not contain hidden or parent components')
    root = root.resolve()
    target = (root / p).resolve()
    if not target.is_relative_to(root):
        raise ValueError('path escapes wiki')
    return target


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.wiki-tmp-')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def locked(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.wiki-write.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class SourceStore:
    def __init__(self, wiki_dir: Path):
        self.root = Path(wiki_dir)

    def archive(self, identity: str, text: str, title: str = '') -> dict:
        if not identity:
            raise ValueError('explicit source identity required')
        sid, vid = digest(identity), digest(text)
        relative = f'sources/versions/{sid}--{vid}.md'
        path = safe_path(self.root, relative)
        record = dict(source_id=sid, version_id=vid, identity=identity, title=title,
                      path=relative, length=len(text))
        with locked(self.root):
            if path.exists() and path.read_bytes().decode('utf-8') != text:
                raise ValueError('immutable source was modified')
            if not path.exists():
                atomic_write(path, text)
            metadata = path.with_suffix('.json')
            if not metadata.exists():
                atomic_write(metadata, json.dumps(record, ensure_ascii=False, indent=2))
        return record

    def get(self, source_id: str, version_id: str) -> tuple[dict, str]:
        if not all(re.fullmatch(r'[0-9a-f]{64}', x) for x in (source_id, version_id)):
            raise ValueError('invalid source/version ID')
        path = safe_path(self.root, f'sources/versions/{source_id}--{version_id}.md')
        meta = json.loads(path.with_suffix('.json').read_bytes().decode('utf-8'))
        text = path.read_bytes().decode('utf-8')
        if (digest(text) != version_id or digest(meta['identity']) != source_id
                or meta.get('source_id') != source_id or meta.get('version_id') != version_id
                or meta.get('length') != len(text)):
            raise ValueError('source integrity check failed')
        return meta, text

    def read(self, source_id: str, version_id: str, start: int = 0, length: int = 6000) -> dict:
        meta, text = self.get(source_id, version_id)
        if not isinstance(start, int) or not 0 <= start <= len(text):
            raise ValueError('invalid source offset')
        if not isinstance(length, int) or not 1 <= length <= 12000:
            raise ValueError('length must be 1..12000')
        end = min(start + length, len(text))
        paragraphs = [{'start': max(start, m.start()), 'end': min(end, m.end()),
                       'paragraph_start': m.start(), 'paragraph_end': m.end()}
                      for m in re.finditer(r'\S.*?(?=\r?\n[ \t]*\r?\n|\Z)', text, re.S)
                      if m.start() < end and m.end() > start]
        return {**meta, 'start': start, 'end': end, 'text': text[start:end],
                'paragraphs': paragraphs, 'next_start': end if end < len(text) else None}

    def validate_citation(self, citation: dict) -> dict:
        meta, text = self.get(citation['source_id'], citation['version_id'])
        start, end, quote = citation['start'], citation['end'], citation['quote']
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
            raise ValueError('invalid citation range')
        if not isinstance(quote, str) or text[start:end] != quote:
            raise ValueError('citation quote does not match source range')
        return {k: citation[k] for k in ('source_id', 'version_id', 'start', 'end', 'quote')}


def fact_state(text: str) -> dict:
    match = BLOCK.search(text)
    if not match:
        if START in text or END in text:
            raise ValueError('damaged managed fact block')
        return {'facts': [], 'links': []}
    if len(BLOCK.findall(text)) != 1:
        raise ValueError('multiple managed fact blocks')
    return json.loads(match.group(1))


class FactStore:
    def __init__(self, wiki_dir: Path):
        self.root = Path(wiki_dir)
        self.sources = SourceStore(self.root)

    def apply(self, path: str, expected_revision: str | None, identity: str,
              facts: list[dict], links: list[str] | None = None,
              title: str = '', description: str = '', aliases: list[str] | None = None) -> dict:
        target = safe_path(self.root, path)
        if target.suffix != '.md' or len(Path(path).parts) < 2 or Path(path).parts[0] in ('sources', 'summaries') or target.name == '_index.md':
            raise ValueError('write a knowledge page in a content directory')
        if not identity or not facts:
            raise ValueError('entity identity and evidence-backed facts are required')
        with locked(self.root):
            exists = target.exists()
            text = target.read_bytes().decode('utf-8') if exists else ''
            revision = digest(text) if exists else None
            if revision != expected_revision:
                raise ValueError('stale revision: read the page and retry')
            state = fact_state(text)
            if state.get('identity') and state['identity'] != identity:
                raise ValueError('entity identity mismatch; create a disambiguated page')
            state['identity'] = identity
            by_id = {f['id']: f for f in state['facts']}
            for raw in facts:
                fact = dict(raw)
                for key in ('statement', 'event_time', 'conditions', 'polarity', 'certainty', 'kind'):
                    if not isinstance(fact.get(key), str):
                        raise ValueError(f'fact requires string field {key}')
                if not fact['statement'].strip() or fact['polarity'] not in ('positive', 'negative') or fact['kind'] not in ('fact', 'relation', 'summary'):
                    raise ValueError('invalid fact statement, polarity or kind')
                citations = fact.get('citations', [])
                if not citations:
                    raise ValueError('each fact requires original evidence')
                fact['citations'] = [self.sources.validate_citation(c) for c in citations]
                semantic = {k: fact[k] for k in ('statement', 'event_time', 'conditions', 'polarity', 'certainty', 'kind')}
                fid = digest(json.dumps(semantic, sort_keys=True, ensure_ascii=False))
                if fact.get('id', fid) != fid:
                    raise ValueError('fact ID is derived from its full qualified statement')
                fact['id'] = fid
                for relation in ('conflicts_with', 'supersedes'):
                    refs = fact.get(relation, [])
                    if not isinstance(refs, list) or any(x not in by_id or x == fid for x in refs):
                        raise ValueError(f'{relation} must reference existing facts on this page')
                    fact[relation] = refs
                if fid in by_id:
                    old = by_id[fid]
                    for key in ('citations', 'conflicts_with', 'supersedes'):
                        fact[key] = old.get(key, []) + [v for v in fact[key] if v not in old.get(key, [])]
                by_id[fid] = fact
            all_links = list(dict.fromkeys(state.get('links', []) + (links or [])))
            for link in all_links:
                destination = safe_path(self.root, link)
                if destination.suffix != '.md' or not destination.is_file():
                    raise ValueError(f'link target does not exist: {link}')
            if any(f['kind'] == 'relation' for f in facts) and len(set(all_links) - {path}) < 2:
                raise ValueError('relation facts require two distinct existing linked pages')
            # Links in free text also must be valid; do not allow hidden unchecked wikilinks.
            for raw_link in re.findall(r'\[\[([^\]]+)\]\]', json.dumps(facts) + description):
                link = raw_link.split('|')[0].split('#')[0]
                destination = safe_path(self.root, link if link.endswith('.md') else link + '.md')
                if not destination.is_file():
                    raise ValueError(f'unresolved inline link: {raw_link}')
            state.update(facts=list(by_id.values()), links=all_links)
            block = START + '\n```json\n' + json.dumps(state, ensure_ascii=False, indent=2) + '\n```\n' + END
            if not exists:
                meta = dict(title=title or target.stem, aliases=aliases or [], description=description,
                            type=Path(path).parts[0], entity_id=identity)
                text = '---\n' + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + '---\n\n# ' + meta['title'] + '\n'
            if BLOCK.search(text):
                text = BLOCK.sub(lambda _: block, text)
            else:
                text = text + '\n\n## Verified facts\n\n' + block + '\n'
            atomic_write(target, text)
            return {'path': path, 'revision': digest(text), 'fact_ids': list(by_id),
                    'links': all_links, 'review_related': all_links}
