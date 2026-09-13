"""One summary layer over source-backed knowledge facts, with lazy invalidation."""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

try:
    from .wiki_store import SourceStore, atomic_write, digest, fact_state, locked, safe_path
except ImportError:
    from wiki_store import SourceStore, atomic_write, digest, fact_state, locked, safe_path


START = '<!-- wiki-summary:start -->'
END = '<!-- wiki-summary:end -->'
BLOCK = re.compile(re.escape(START) + r'\n```json\n(.*?)\n```\n' + re.escape(END), re.S)


def summary_state(text: str) -> dict:
    matches = BLOCK.findall(text)
    if len(matches) != 1:
        raise ValueError('missing or damaged summary block')
    state = json.loads(matches[0])
    if not isinstance(state, dict) or state.get('schema_version') != 1 or state.get('level') != 1:
        raise ValueError('only level-one summaries are supported')
    def require(condition):
        if not condition:
            raise ValueError('malformed summary record')
    try:
        require(isinstance(state['claims'], list) and 1 <= len(state['claims']) <= 8)
        require(isinstance(state['children'], list) and 2 <= len(state['children']) <= 4)
        require(isinstance(state['source_versions'], dict) and len(state['source_versions']) >= 2)
        require(isinstance(state['title'], str) and isinstance(state['coverage'], str))
        for child in state['children']:
            require(isinstance(child['path'], str) and isinstance(child['revision'], str))
        for claim in state['claims']:
            require(isinstance(claim['statement'], str) and claim['kind'] in ('direct', 'synthesis'))
            require(isinstance(claim['id'], str) and isinstance(claim['supports'], list) and claim['supports'])
            require(isinstance(claim['citations'], list) and claim['citations'])
            for support in claim['supports']:
                require(isinstance(support['path'], str) and isinstance(support['fact_id'], str))
        for sid, versions in state['source_versions'].items():
            require(isinstance(sid, str) and re.fullmatch('[a-f0-9]{64}', sid))
            require(isinstance(versions, list) and all(isinstance(v, str) and re.fullmatch('[a-f0-9]{64}', v) for v in versions))
    except (TypeError, KeyError):
        raise ValueError('malformed summary record') from None
    return state


def current_facts(text: str) -> list[dict]:
    """Retain explicit conflicts, exclude superseded facts and generated summaries."""
    facts = fact_state(text)['facts']
    superseded = {fid for fact in facts for fid in fact.get('supersedes', [])}
    return [f for f in facts if f['kind'] in ('fact', 'relation') and f['id'] not in superseded]


def summary_text(state: dict) -> str:
    return '\n'.join(('Synthesis: ' if c['kind'] == 'synthesis' else '') + c['statement']
                     for c in state['claims'])


class SummaryStore:
    def __init__(self, wiki_dir: Path):
        self.root = Path(wiki_dir)
        self.sources = SourceStore(self.root)

    @staticmethod
    def path_for(children: list[str]) -> str:
        return 'summaries/' + digest(json.dumps(sorted(children))) + '.md'

    def leaf(self, path: str) -> tuple[str, dict[str, dict]]:
        if path.startswith(('summaries/', 'sources/')):
            raise ValueError('summary children must be knowledge pages, never summaries or source copies')
        target = safe_path(self.root, path)
        if target.suffix != '.md':
            raise ValueError('summary children must be Markdown pages')
        text = target.read_bytes().decode('utf-8')
        if START in text:
            raise ValueError('recursive summaries are not supported')
        return digest(text), {f['id']: f for f in current_facts(text)}

    def versions(self, source_ids) -> dict[str, list[str]]:
        directory = self.root / 'sources' / 'versions'
        return {sid: sorted(p.stem.split('--', 1)[1] for p in directory.glob(sid + '--*.md'))
                for sid in sorted(source_ids)}

    def apply(self, *, children: list[dict], title: str, claims: list[dict],
              expected_revision: str | None = None, source_versions: dict | None = None) -> dict:
        if not isinstance(children, list) or not 2 <= len(children) <= 4:
            raise ValueError('a summary needs two to four distinct knowledge pages')
        paths = [c['path'] for c in children]
        if len(set(paths)) != len(paths):
            raise ValueError('duplicate summary child')
        if not isinstance(title, str) or not title.strip() or len(title) > 200:
            raise ValueError('summary title must contain 1..200 characters')
        if not isinstance(claims, list) or not 1 <= len(claims) <= 8:
            raise ValueError('a summary needs 1..8 claims')
        if sum(len(c.get('statement', '')) for c in claims if isinstance(c.get('statement'), str)) > 4000:
            raise ValueError('summary overview exceeds 4000 characters')
        path = self.path_for(paths)
        target = safe_path(self.root, path)
        with locked(self.root):
            old = target.read_bytes().decode('utf-8') if target.exists() else None
            if (digest(old) if old is not None else None) != expected_revision:
                raise ValueError('stale summary revision')
            leaves = {}
            for child in children:
                revision, facts = self.leaf(child['path'])
                if revision != child['revision']:
                    raise ValueError('summary child changed; rebuild with fresh facts')
                leaves[child['path']] = facts
            validated, used_pages, source_ids = [], set(), set()
            for claim in claims:
                statement, kind = claim.get('statement'), claim.get('kind')
                if not isinstance(statement, str) or not statement.strip() or len(statement) > 1200:
                    raise ValueError('summary statements must contain 1..1200 characters')
                if kind not in ('direct', 'synthesis'):
                    raise ValueError('claim kind must be direct or synthesis')
                supports = claim.get('supports')
                if not isinstance(supports, list) or not 1 <= len(supports) <= 16:
                    raise ValueError('each summary claim needs bounded fact support')
                clean_supports, citations = [], []
                for support in supports:
                    page, fid = support['path'], support['fact_id']
                    fact = leaves.get(page, {}).get(fid)
                    if fact is None or not fact.get('citations'):
                        raise ValueError('support must reference a current source-backed child fact')
                    clean = {'path': page, 'fact_id': fid}
                    if clean not in clean_supports:
                        clean_supports.append(clean)
                    used_pages.add(page)
                    for citation in fact['citations']:
                        citation = self.sources.validate_citation(citation)
                        source_ids.add(citation['source_id'])
                        if citation not in citations:
                            citations.append(citation)
                qualified = dict(statement=statement.strip(), kind=kind, supports=clean_supports)
                if sum(len(c['quote']) for c in citations) > 32000:
                    raise ValueError('summary claim evidence exceeds the disclosure budget')
                validated.append({**qualified, 'id': digest(json.dumps(qualified, sort_keys=True)), 'citations': citations})
            if used_pages != set(paths) or len(source_ids) < 2:
                raise ValueError('summary must use every child and at least two distinct original source identities')
            versions = self.versions(source_ids)
            if source_versions is not None and any(source_versions.get(sid) != vids for sid, vids in versions.items()):
                raise ValueError('source versions changed during summary generation')
            state = dict(schema_version=1, level=1, title=title.strip(),
                         children=sorted(children, key=lambda c: c['path']), claims=validated,
                         source_versions=versions, coverage='selected facts; not exhaustive document coverage')
            meta = dict(title=state['title'], type='cross_document_summary', level=1,
                        description=summary_text(state))
            text = ('---\n' + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + '---\n\n'
                    + START + '\n```json\n' + json.dumps(state, ensure_ascii=False, indent=2) + '\n```\n' + END + '\n')
            if old is not None and old != text:
                atomic_write(self.root / '.summary-history' / digest(old) / target.name, old)
            atomic_write(target, text)
        return {'path': path, 'revision': digest(text), 'claim_ids': [c['id'] for c in validated]}

    def get(self, path: str) -> tuple[dict, str]:
        if not path.startswith('summaries/'):
            raise ValueError('expected a summary path')
        text = safe_path(self.root, path).read_bytes().decode('utf-8')
        state = summary_state(text)
        if self.path_for([c['path'] for c in state['children']]) != path:
            raise ValueError('summary identity does not match its children')
        return state, digest(text)

    def freshness(self, state: dict) -> list[str]:
        reasons = []
        try:
            for child in state['children']:
                revision, _ = self.leaf(child['path'])
                if revision != child['revision']:
                    reasons.append('child changed: ' + child['path'])
            if self.versions(state['source_versions']) != state['source_versions']:
                reasons.append('source version set changed; review and rebuild')
            for claim in state['claims']:
                for citation in claim['citations']:
                    self.sources.validate_citation(citation)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reasons.append('dependency unavailable or invalid: ' + str(exc))
        return reasons

    def read(self, path: str, view: str = 'overview', claim_ids: list[str] | None = None) -> dict:
        if view not in ('overview', 'claims', 'evidence'):
            raise ValueError('summary view must be overview, claims or evidence')
        state, revision = self.get(path)
        reasons = self.freshness(state)
        result = dict(path=path, revision=revision, level=1, title=state['title'], view=view,
                      status='stale' if reasons else 'current', stale_reasons=reasons,
                      children=state['children'], coverage=state['coverage'])
        if reasons:
            # Keep child navigation available, but never disclose an obsolete summary as evidence.
            return {**result, 'text': 'Summary needs rebuilding. Read the child pages or search original evidence.', 'claims': []}
        claims = state['claims']
        if claim_ids is not None:
            if not isinstance(claim_ids, list) or not claim_ids or any(cid not in {c['id'] for c in claims} for cid in claim_ids):
                raise ValueError('unknown or empty summary claim selection')
            claims = [c for c in claims if c['id'] in claim_ids]
        if view == 'evidence' and len(claims) > 1:
            raise ValueError('select one claim_id at a time to reveal supporting evidence')
        if view == 'overview':
            result['text'] = summary_text(state)
        else:
            result['claims'] = [{k: c[k] for k in ('id', 'statement', 'kind', 'supports')} for c in claims]
            if view == 'evidence':
                leaves = {c['path']: self.leaf(c['path'])[1] for c in state['children']}
                for output, claim in zip(result['claims'], claims):
                    output['citations'] = claim['citations']
                    output['facts'] = [{'path': s['path'], **leaves[s['path']][s['fact_id']]} for s in claim['supports']]
        return result
