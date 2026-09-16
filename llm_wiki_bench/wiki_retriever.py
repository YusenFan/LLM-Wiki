"""Live filesystem navigation, BM25 candidates, exact entities and precise reads."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml
try:
    from .wiki_store import SourceStore, digest, fact_state, frontmatter, safe_path, unwrap_markdown
    from .summary_store import SummaryStore, summary_state, summary_text
except ImportError:
    from wiki_store import SourceStore, digest, fact_state, frontmatter, safe_path, unwrap_markdown
    from summary_store import SummaryStore, summary_state, summary_text


@dataclass
class WikiPage:
    name: str
    dir_name: str
    rel_path: str
    text: str
    body: str
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    description: str = ''
    links_to: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    page: WikiPage
    score: float
    matched_fields: list[str] = field(default_factory=list)


class WikiRetriever:
    # 【检索初始化】绑定 wiki、BM25 模式及可选 reranker 回调，创建内存索引容器和来源／摘要存储对象；不调用 LLM。
    def __init__(self, wiki_dir: Path, search_mode: str = 'bm25', reranker=None):
        if search_mode not in ('bm25', 'exact_then_bm25'):
            raise ValueError('unsupported search mode')
        self.wiki_dir = Path(wiki_dir)
        self.search_mode, self.reranker = search_mode, reranker
        self.pages: dict[str, WikiPage] = {}
        self.dir_indexes: dict[str, str] = {}
        self.directories: dict[str, str] = {}
        self.errors: list[dict] = []
        self._signature = None
        self.sources = SourceStore(self.wiki_dir)
        self.summaries = SummaryStore(self.wiki_dir)
        self.related_summaries: dict[str, list[str]] = {}

    # 【索引刷新】扫描非隐藏、非符号链接的实际文件，用路径/mtime/size 判断是否变化；解析 Markdown、目录及摘要状态，再重建倒排索引和反向摘要导航。
    # 坏元数据降级为正文并记录错误。
    def load(self) -> None:
        # stat snapshots include untracked files, deletions and new directories.
        files = sorted(p for p in self.wiki_dir.rglob('*')
                       if not any(x.startswith('.') for x in p.relative_to(self.wiki_dir).parts)
                       and not p.is_symlink())
        files = [p for p in files if p.resolve().is_relative_to(self.wiki_dir.resolve())]
        signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
        if signature == self._signature:
            return
        self.pages, self.dir_indexes, self.directories, self.errors = {}, {}, {}, []
        descriptions = {}
        registry = self.wiki_dir / 'page_types.yaml'
        if registry.is_file():
            try:
                descriptions = (yaml.safe_load(registry.read_text()) or {}).get('page_types', {})
            except (ValueError, yaml.YAMLError, AttributeError):
                pass
        for path in files:
            rel = path.relative_to(self.wiki_dir).as_posix()
            if path.is_dir():
                info = descriptions.get(rel, {})
                self.directories[rel] = info.get('description', rel.rsplit('/', 1)[-1]) if isinstance(info, dict) else str(info)
                continue
            if path.suffix != '.md' or path.name in ('index.md', 'overview.md', 'log.md'):
                continue
            text = path.read_bytes().decode('utf-8')
            if path.name == '_index.md':
                self.dir_indexes[str(Path(rel).parent)] = text
                continue
            try:
                page = self._parse_page(path.stem, Path(rel), text)
                self.pages[rel] = page
            except (ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
                # Broken metadata must not make the original body inaccessible.
                self.errors.append({'path': rel, 'error': str(exc)})
                self.pages[rel] = WikiPage(path.stem, str(Path(rel).parent), rel, text, text)
        self.related_summaries = {}
        for rp, page in self.pages.items():
            if page.metadata.get('type') == 'cross_document_summary' and page.metadata.get('summary_status') == 'current':
                for child in page.links_to:
                    self.related_summaries.setdefault(child, []).append(rp)
        self._build_bm25()
        self._signature = signature

    # 【页面解析】转换为 WikiPage；普通页抽取标题、别名、标签及链接，原文版本页补侧车元数据，摘要页检查新鲜度并仅以可用概览作为正文。
    def _parse_page(self, stem: str, rel: Path, text: str) -> WikiPage:
        meta, body = frontmatter(text)
        if rel.as_posix().startswith('summaries/'):
            state, _ = self.summaries.get(rel.as_posix())
            reasons = self.summaries.freshness(state)
            body = summary_text(state) if not reasons else ''
            meta = {**meta, 'type': 'cross_document_summary', 'level': 1, 'title': state['title'],
                    'description': body, 'summary_status': 'stale' if reasons else 'current'}
            return WikiPage(state['title'], str(rel.parent), rel.as_posix(), text, body,
                            description=body, links_to=[c['path'] for c in state['children']], metadata=meta)
        if rel.as_posix().startswith('sources/versions/'):
            sidecar = self.wiki_dir / rel.with_suffix('.json')
            if sidecar.is_file():
                meta = {**meta, **json.loads(sidecar.read_bytes().decode('utf-8'))}
        # 【字段兼容】把元数据中的列表或单个值转成字符串列表，缺失或空值返回 []。
        def strings(key):
            value = meta.get(key, [])
            return [str(x) for x in value] if isinstance(value, list) else [str(value)] if value else []
        description = str(meta.get('description', ''))
        if not description:
            description = next((line[2:] for line in body.splitlines() if line.startswith('> ')), '')
        state = fact_state(text)
        links = [x.split('|')[0].split('#')[0] for x in re.findall(r'\[\[(.*?)\]\]', body)]
        links += state.get('links', [])
        return WikiPage(str(meta.get('title') or meta.get('source_title') or stem), str(rel.parent), rel.as_posix(), text, body,
                        strings('aliases'), strings('tags'), description, list(dict.fromkeys(links)), meta)

    # 【词法分词】casefold 后用 Unicode 字母数字片段分词；不做语义向量、词干化或中文词语切分，连续中文可能成为一个 token。
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'[^\W_]+', text.casefold(), re.UNICODE)

    # 【名称归一】连字符替为空格，再分词拼回字符串；用于完整标题／别名匹配，不证明两个同名条目是同一实体。
    @classmethod
    def _entity_key(cls, text: str) -> str:
        return ' '.join(cls._tokenize(text.replace('-', ' ')))

    # 【索引构建】索引标题、别名、标签、描述及正文和事实 statement；剔除受控 JSON 中的哈希与重复引文，建立词频和文长表。
    # 这里没有过滤所有被 supersedes 的旧 statement。
    def _build_bm25(self):
        self._inv, self._doc_len = {}, {}
        for rp, page in self.pages.items():
            if rp.startswith('summaries/') and page.metadata.get('summary_status') != 'current':
                continue
            # Index readable statements, not hashes, JSON keys or repeated quoted evidence.
            from_block = fact_state(page.text) if '<!-- wiki-facts:start -->' in page.text and not any(e['path'] == rp for e in self.errors) else {}
            body = re.sub(r'<!-- wiki-facts:start -->.*?<!-- wiki-facts:end -->', '', page.body, flags=re.S)
            body += ' '.join(f['statement'] for f in from_block.get('facts', []))
            tokens = self._tokenize(' '.join([page.name, *page.aliases, *page.tags, page.description, body]))
            self._doc_len[rp] = len(tokens)
            for word, tf in Counter(tokens).items():
                self._inv.setdefault(word, {})[rp] = tf
        self._avg_dl = max(1, sum(self._doc_len.values()) / max(1, len(self.pages)))

    # 【确定性排序】按词频、逆文档频率和长度归一计算每页 BM25 分数；k1=1.2、b=0.75，无子串额外加分，返回路径到分数的映射。
    def _bm25_score(self, query_tokens):
        scores = {}
        for token in set(query_tokens):
            postings = self._inv.get(token, {})
            idf = math.log(1 + (len(self.pages) - len(postings) + .5) / (len(postings) + .5))
            for rp, tf in postings.items():
                scores[rp] = scores.get(rp, 0) + idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * self._doc_len[rp] / self._avg_dl))
        return scores

    # 【候选检索】刷新索引后按 layer、directory 和模式过滤、排序，折叠内容相同的来源副本，再可选重排并截到 limit。
    # details 也包含来源页；exact_then_bm25 优先完整实体匹配，不是语义搜索。
    def search(self, query: str, limit: int = 10, directory: str = '', mode: str | None = None,
               layer: str = 'all') -> list[SearchResult]:
        self.load()
        if layer not in ('all', 'summaries', 'details'):
            raise ValueError('search layer must be all, summaries or details')
        if not isinstance(limit, int) or not 1 <= limit <= 50:
            raise ValueError('limit must be 1..50')
        mode = mode or self.search_mode
        if mode not in ('bm25', 'exact', 'exact_then_bm25'):
            raise ValueError('invalid search mode')
        query_key = self._entity_key(query)
        if not query_key:
            return []
        scores = self._bm25_score(self._tokenize(query))
        results = []
        for rp, page in self.pages.items():
            is_summary = rp.startswith('summaries/')
            if is_summary and page.metadata.get('summary_status') != 'current':
                continue
            if (layer == 'summaries' and not is_summary) or (layer == 'details' and is_summary):
                continue
            if directory and not rp.startswith(directory.rstrip('/') + '/'):
                continue
            exact = query_key in {self._entity_key(x) for x in [page.name, *page.aliases]}
            if mode == 'exact' and not exact:
                continue
            if exact or scores.get(rp, 0) > 0:
                results.append(SearchResult(page, scores.get(rp, 0), ['exact_entity'] if exact else ['bm25']))
        results.sort(key=lambda r: ((mode != 'bm25' and 'exact_entity' in r.matched_fields), r.score, r.page.rel_path), reverse=True)
        # Collapse copies of the SAME source version only; never collapse entities by title.
        seen, unique = set(), []
        for result in results:
            page = result.page
            key = ('source', digest(page.text)) if page.rel_path.startswith('sources/') else ('page', page.rel_path)
            if key not in seen:
                unique.append(result)
                seen.add(key)
        if self.reranker and mode != 'exact':
            unique = self.reranker(query, unique)
        return unique[:limit]

    # 【导航入口】返回所有可见实际目录、描述、直接页面数、根页及解析错误；不是把所有子目录页面全文塞进上下文，需 wiki_read 分页浏览。
    def tree(self) -> dict:
        self.load()
        return {'directories': [{'path': d, 'description': desc,
                                 'page_count': sum(p.dir_name == d for p in self.pages.values())}
                                for d, desc in sorted(self.directories.items())],
                'root_pages': [p.rel_path for p in self.pages.values() if p.dir_name == '.'],
                'parse_errors': self.errors,
                'summary_layer': {'levels': 1, 'current': sum(p.metadata.get('summary_status') == 'current' for p in self.pages.values()),
                                  'stale': sum(p.metadata.get('summary_status') == 'stale' for p in self.pages.values()),
                                  'read_tool': 'summary_read: overview -> claims -> evidence -> source_read'}}

    # 【兼容输出】把实时 tree 序列化为 JSON 字符串；不使用旧 index.md 概览挑选主题。
    def wiki_map(self) -> str:
        # All actual directories, no sampled index entries or hard-coded topic selection.
        tree = self.tree()
        return json.dumps(tree, ensure_ascii=False)

    # 【目录／页面读取】一次处理 1..15 个路径；目录按页数分页，文件按选定 view/section 的字符分页，并返回整页 revision、链接和来源导航。
    # facts 视图含历史事实；本函数的窗口坐标不能直接当原文引文坐标。
    def read(self, paths: list[str], section: str | None = None, start: int = 0,
             length: int = 6000, view: str = 'text', page_limit: int = 50) -> list[dict]:
        self.load()
        if not isinstance(paths, list) or not 1 <= len(paths) <= 15:
            raise ValueError('read 1..15 paths per call')
        if not isinstance(start, int) or start < 0 or not isinstance(length, int) or not 1 <= length <= 12000:
            raise ValueError('invalid read window')
        if not isinstance(page_limit, int) or not 1 <= page_limit <= 200:
            raise ValueError('page_limit must be 1..200')
        out = []
        for raw in paths:
            if raw == '/':
                out.append({'path': '/', 'type': 'root', **self.tree()})
                continue
            raw = raw.split('|', 1)[0].split('#', 1)[0]
            if raw.endswith('/_index.md'):
                raw = raw.removesuffix('/_index.md')
            path = safe_path(self.wiki_dir, raw)
            if not path.exists() and not raw.endswith('.md'):
                path = safe_path(self.wiki_dir, raw + '.md')
            rp = path.relative_to(self.wiki_dir.resolve()).as_posix()
            if path.is_dir():
                pages = [{'path': p.rel_path, 'name': p.name, 'description': p.description,
                          **({'summary_status': p.metadata['summary_status']} if 'summary_status' in p.metadata else {})}
                         for p in self.pages.values() if p.dir_name == rp]
                end = min(start + page_limit, len(pages))
                out.append({'path': rp, 'type': 'directory',
                            'directories': [d for d in self.directories if str(Path(d).parent) == rp],
                            'pages': pages[start:end], 'start': start, 'end': end, 'total_pages': len(pages),
                            'next_start': end if end < len(pages) else None})
                continue
            page = self.pages.get(rp)
            if not page:
                out.append({'path': rp, 'error': 'not found', 'type': 'error'})
                continue
            text = page.text
            extra = {}
            if rp.startswith('summaries/'):
                revealed = self.summaries.read(rp)
                text = revealed['text']
                extra = {'summary_status': revealed['status'], 'stale_reasons': revealed['stale_reasons'],
                         'next_tool': {'name': 'summary_read', 'arguments': {'path': rp, 'view': 'claims'}}}
            elif view == 'summary':
                state = fact_state(text)
                superseded = {fid for f in state['facts'] for fid in f.get('supersedes', [])}
                summaries = [f['statement'] for f in state['facts'] if f['kind'] == 'summary' and f['id'] not in superseded]
                text = '\n'.join(summaries) or page.description
            elif view in ('facts', 'relations'):
                state = fact_state(text)
                selected = state['facts'] if view == 'facts' else [f for f in state['facts'] if f['kind'] == 'relation']
                text = json.dumps(selected, ensure_ascii=False)
            elif view != 'text':
                raise ValueError('unknown view')
            if section:
                lines = text.splitlines(keepends=True)
                headings = [(i, len(m[1]), m[2].strip()) for i, line in enumerate(lines)
                            if (m := re.match(r'^(#{1,6})\s+(.+?)\s*$', line))]
                matches = [(i, level) for i, level, name in headings if name.casefold() == section.casefold()]
                if len(matches) != 1:
                    out.append({'path': rp, 'type': 'error', 'error': 'section missing or ambiguous; omit section to read the whole page',
                                'available_sections': [name for _, _, name in headings], 'revision': digest(page.text)})
                    continue
                i, level = matches[0]
                end_line = next((j for j, lev, _ in headings if j > i and lev <= level), len(lines))
                text = ''.join(lines[i:end_line])
            end = min(start + length, len(text))
            if start > len(text):
                raise ValueError('offset exceeds selected content')
            out.append({'path': rp, 'type': 'file', 'name': page.name, 'revision': digest(page.text),
                        'text': text[start:end], 'start': start, 'end': end, 'length': len(text),
                        'next_start': end if end < len(text) else None,
                        'meta': {**page.metadata, 'links_to': page.links_to}, 'source_refs': self._source_refs(page),
                        'related_summaries': self.related_summaries.get(rp, []), 'view': view, 'section': section, **extra})
        return out

    # 【来源导航／兼容写盘】只根据明确 source_article 或来源路径链接获取原文引用；旧 sources/articles 页会被即时归档为不可变快照，所以旧页读取可能写 sources/versions。
    # 不会凭标题相似猜来源。
    def _source_refs(self, page: WikiPage) -> list[dict]:
        refs = []
        candidates = [page.rel_path] if page.rel_path.startswith(('sources/articles/', 'sources/versions/')) else []
        article = page.metadata.get('source_article')
        if isinstance(article, str):
            candidates.append('sources/articles/' + article.removesuffix('.md') + '.md')
        candidates += [link if link.endswith('.md') else link + '.md' for link in page.links_to
                       if link.startswith(('sources/articles/', 'sources/versions/'))]
        for relative in dict.fromkeys(candidates):
            try:
                target = safe_path(self.wiki_dir, relative)
                if not target.is_file():
                    continue
                if relative.startswith('sources/versions/'):
                    ref = json.loads(target.with_suffix('.json').read_bytes().decode('utf-8'))
                    self.sources.get(ref['source_id'], ref['version_id'])
                else:
                    original = target.read_bytes().decode('utf-8')
                    ref = self.sources.archive(target.as_uri(), original, self.pages[relative].name if relative in self.pages else target.stem)
                refs.append(ref)
            except (OSError, ValueError, KeyError):
                continue
        return refs

    # 【工具分发】把模型工具名和参数路由到真实 Python 方法，将结果或捕获的 error 序列化成 JSON。
    # 不会执行模型生成的 Python 代码；entity_lookup 强制 exact 模式。
    def execute_tool(self, name: str, arguments: dict) -> str:
        try:
            if not isinstance(arguments, dict):
                raise ValueError('tool arguments must be an object')
            if name in ('wiki_search', 'entity_lookup'):
                args = dict(arguments)
                if name == 'entity_lookup':
                    args['mode'] = 'exact'
                results = self.search(**args)
                payload = {'matched': len(results), 'results': [
                    {'path': r.page.rel_path, 'name': r.page.name, 'score': r.score,
                     'matched_fields': r.matched_fields,
                     'meta': {**r.page.metadata, 'description': r.page.description}} for r in results]}
            elif name == 'wiki_read':
                payload = self.read(**arguments)
            elif name == 'wiki_tree':
                payload = self.tree()
            elif name == 'source_read':
                payload = self.sources.read(**arguments)
            elif name == 'summary_read':
                payload = self.summaries.read(**arguments)
            else:
                raise ValueError(f'unknown tool: {name}')
        except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
            payload = {'error': str(exc)}
        return json.dumps(payload, ensure_ascii=False)


# 【协议描述】组装 OpenAI 风格的函数名、描述、参数 JSON Schema；它向模型说明如何请求，真正执行和硬校验由本地分发器完成。
def tool_schema(name, description, properties, required=()):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties, 'required': list(required), 'additionalProperties': False}}}

S = {'type': 'string'}
I = {'type': 'integer'}
STRINGS = {'type': 'array', 'items': S}
CITATION = {'type': 'object', 'properties': {'source_id': S, 'version_id': S, 'start': I, 'end': I, 'quote': S},
            'required': ['source_id', 'version_id', 'start', 'end', 'quote'], 'additionalProperties': False}
WIKI_TOOL_SCHEMAS = [
    tool_schema('wiki_tree', 'Show every live directory and description, including uncommitted files.', {}),
    tool_schema('wiki_search', 'BM25 candidate search, metadata only. Change queries/directories if evidence is missing.',
                {'query': S, 'limit': I, 'directory': S, 'mode': {'enum': ['bm25', 'exact_then_bm25']},
                 'layer': {'enum': ['all', 'summaries', 'details']}}, ['query']),
    tool_schema('entity_lookup', 'Exact full name/alias lookup; multiple matches need disambiguation.', {'query': S, 'limit': I, 'directory': S}, ['query']),
    tool_schema('wiki_read', 'Browse directories or read a page/view/section in windows. Follow next_start until done when needed.',
                {'paths': STRINGS, 'section': S, 'start': I, 'length': I, 'page_limit': I, 'view': {'enum': ['text', 'summary', 'facts', 'relations']}}, ['paths']),
    tool_schema('source_read', 'Read immutable original evidence by source/version IDs and Unicode character offsets; next_start exposes the rest.',
                {'source_id': S, 'version_id': S, 'start': I, 'length': I}, ['source_id', 'version_id']),
    tool_schema('summary_read', 'Progressively reveal a cross-document summary: overview, claim map, then evidence for ONE claim_id per call. Stale summaries only return child navigation. Read originals with source_read before answering.',
                {'path': S, 'view': {'enum': ['overview', 'claims', 'evidence']}, 'claim_ids': STRINGS}, ['path']),
]
