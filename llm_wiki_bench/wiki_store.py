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


# 【确定性基础】计算 UTF-8 文本的 SHA-256；用于来源身份、内容版本、页面 revision 和事实 ID，哈希相等不等于实体语义相同。
def digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# 【旧格式兼容】仅当整篇文本被同一 Markdown 围栏包住时解包；返回正文，否则原样返回。
def unwrap_markdown(text: str) -> str:
    match = re.fullmatch(r'\s*(`{3,}|~{3,})(?:markdown|md)?\s*\n(.*)\n\1\s*', text, re.S)
    return match.group(2) + '\n' if match else text


# 【确定性解析】解开旧整页围栏后拆分 YAML 元数据和正文；要求元数据是字典，日期保留字符串，格式错误向上传递。
def frontmatter(text: str) -> tuple[dict, str]:
    text = unwrap_markdown(text)
    match = re.match(r'\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|$)', text, re.S)
    if not match:
        return {}, text
    meta = yaml.load(match.group(1), Loader=MetadataLoader) or {}
    if not isinstance(meta, dict):
        raise ValueError('frontmatter must be a mapping')
    return meta, text[match.end():]


# 【路径边界】把相对路径解析到 wiki 根目录内；拒绝绝对路径、父目录、隐藏组件及逃逸的符号链接，返回 Path。
def safe_path(root: Path, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or any(x.startswith('.') for x in p.parts):
        raise ValueError('path must be relative and may not contain hidden or parent components')
    root = root.resolve()
    target = (root / p).resolve()
    if not target.is_relative_to(root):
        raise ValueError('path escapes wiki')
    return target


# 【写盘】同目录临时文件写入并 fsync 后用 replace 替换目标；保证单文件提交，不提供多页面事务。
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


# 【并发控制】用根目录下的 fcntl 文件锁串行化写入；yield 期间持锁，退出时释放，适用于 Unix。
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
    # 【初始化】保存原文仓库根路径；本身不读写文件，也不调用模型。
    def __init__(self, wiki_dir: Path):
        self.root = Path(wiki_dir)

    # 【原文归档】输入明确来源 identity 和完整 text；分别哈希为 source_id/version_id，保存不可变 Markdown 与 JSON 元数据，返回来源记录。
    # 已有同版本文本不符时拒绝写入。
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

    # 【原文读取】按两个 64 位十六进制 ID 找快照；重算文本与身份哈希并检查长度，返回 (元数据, 全文)，篡改或缺失时失败。
    def get(self, source_id: str, version_id: str) -> tuple[dict, str]:
        if not all(isinstance(x, str) and re.fullmatch(r'[0-9a-f]{64}', x) for x in (source_id, version_id)):
            raise ValueError('invalid source/version ID: source_id and version_id must be the 64-hex IDs returned by source_read or listed in a page\'s fact citations / source_refs; a page path or page revision is not a source')
        path = safe_path(self.root, f'sources/versions/{source_id}--{version_id}.md')
        if not path.exists() or not path.with_suffix('.json').exists():
            raise ValueError('unknown source/version ID: no archived source has this source_id/version_id pair. '
                             'Use the pair exactly as listed in a page\'s fact citations or source_refs, or as returned by source_read; '
                             'page revisions are not source IDs')
        meta = json.loads(path.with_suffix('.json').read_bytes().decode('utf-8'))
        text = path.read_bytes().decode('utf-8')
        if (digest(text) != version_id or digest(meta['identity']) != source_id
                or meta.get('source_id') != source_id or meta.get('version_id') != version_id
                or meta.get('length') != len(text)):
            raise ValueError('source integrity check failed')
        return meta, text

    # 【原文窗口】返回 Unicode 字符区间 [start,end)、文本、段落范围和 next_start；默认 6000、最多 12000 字符，end 参数仅 Python 接口支持。
    def read(self, source_id: str, version_id: str, start: int = 0, length: int = 6000, end: int | None = None) -> dict:
        meta, text = self.get(source_id, version_id)
        if end is not None:
            if not isinstance(end, int) or not isinstance(start, int) or end <= start:
                raise ValueError('end must be an integer greater than start')
            length = min(end - start, 12000)
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

    # 【引用校验与纠偏】检查来源完整性和区间，优先逐字匹配；位置不准时搜索引文，还容忍空白、大小写及尾标点差异，返回对齐后的原文切片。
    # 多处匹配可能取首处；不判断引文是否支持 statement。
    def validate_citation(self, citation: dict) -> dict:
        meta, text = self.get(citation['source_id'], citation['version_id'])
        start, end, quote = citation['start'], citation['end'], citation['quote']
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
            raise ValueError('invalid citation range')
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError('citation quote must be a verbatim excerpt of the source')
        if text[start:end] != quote:
            # Models are unreliable with character offsets; the verbatim quote is the anchor.
            # Snap [start, end) to the quote, preferring an occurrence inside the given range.
            found = text.find(quote, start, end)
            if found < 0:
                found = text.find(quote)
            if found >= 0:
                start, end = found, found + len(quote)
            else:
                # Tolerate whitespace-run and letter-case differences, then store the exact source slice.
                pattern = r'\s+'.join(re.escape(part) for part in quote.split())
                m = re.search(pattern, text) or re.search(pattern, text, re.IGNORECASE)
                if not m:
                    # Models often cut a clause short and append their own terminal punctuation.
                    core = quote.rstrip().rstrip('.,;:!?')
                    if core and core != quote.rstrip():
                        pattern = r'\s+'.join(re.escape(part) for part in core.split())
                        m = re.search(pattern, text) or re.search(pattern, text, re.IGNORECASE)
                if not m:
                    raise ValueError('citation quote does not match source range: copy one contiguous excerpt verbatim from source_read text (no paraphrase, no joining separate sentences)')
                start, end, quote = m.start(), m.end(), m.group(0)
        return {'source_id': citation['source_id'], 'version_id': citation['version_id'],
                'start': start, 'end': end, 'quote': quote}


# 【事实解析】从 Markdown 的 wiki-facts JSON 块取 facts/links；旧页无块返回空集合，损坏或多个块报错。
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
    # 【初始化】绑定 wiki 路径及 SourceStore，供事实写入时核对每一条原文引用。
    def __init__(self, wiki_dir: Path):
        self.root = Path(wiki_dir)
        self.sources = SourceStore(self.root)

    # 【事实提交】接收路径、预期 revision、实体 identity、事实和链接；加锁后校验版本、身份、引文与目标存在性，再原子更新受控 JSON 块。
    # 相同限定陈述合并证据，不同陈述新增；旧正文保留。
    # 返回新 revision、全部 fact_ids 和关联页；不会自动修复关联页，也不做语义去重。
    def apply(self, path: str, expected_revision: str | None, identity: str,
              facts: list[dict], links: list[str] | None = None,
              title: str = '', description: str = '', aliases: list[str] | None = None) -> dict:
        if path and not Path(path).suffix:
            path = f'{path}.md'
        target = safe_path(self.root, path)
        if target.suffix != '.md' or len(Path(path).parts) < 2 or Path(path).parts[0] in ('sources', 'summaries') or target.name == '_index.md':
            raise ValueError("write a knowledge page in a content directory: path must look like '<directory>/<page>.md' and not be under sources/ or summaries/")
        if not identity or not facts:
            raise ValueError('entity identity and evidence-backed facts are required')
        with locked(self.root):
            exists = target.exists()
            text = target.read_bytes().decode('utf-8') if exists else ''
            revision = digest(text) if exists else None
            if revision != expected_revision:
                raise ValueError(f'stale revision: page {path} already exists with revision {revision!r}; call wiki_read with paths=[{path!r}] (no section) to see its content, then retry with expected_revision set to that value' if exists else 'stale revision: page does not exist yet; use expected_revision null')
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
                try:
                    fact['citations'] = [self.sources.validate_citation(c) for c in citations]
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"fact {facts.index(raw)} ({fact['statement'][:60]!r}): {exc}") from exc
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
