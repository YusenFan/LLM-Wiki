"""Deterministic legacy repair and source/fact/link integrity audit (no LLM)."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

try:
    from .wiki_store import SourceStore, atomic_write, digest, fact_state, frontmatter, safe_path, unwrap_markdown
    from .wiki_retriever import WikiRetriever
    from .summary_store import SummaryStore
except ImportError:
    from wiki_store import SourceStore, atomic_write, digest, fact_state, frontmatter, safe_path, unwrap_markdown
    from wiki_retriever import WikiRetriever
    from summary_store import SummaryStore
import yaml


# 【有限修复】给部分可确定的 YAML 列表或标量补 JSON 引号；只改可识别格式，不推测缺失事实。
def repair_metadata(text: str) -> str:
    """Quote unquoted punctuation in known scalar/list metadata, never infer values."""
    match = re.match(r'\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|$)', text, re.S)
    if not match:
        return text
    lines = match.group(1).splitlines()
    for i, line in enumerate(lines):
        array = re.fullmatch(r'(aliases|tags):\s*\[(.*)\]\s*', line)
        if array:
            try:
                yaml.safe_load(line)
            except yaml.YAMLError:
                values = [value.strip() for value in array[2].split(',')]
                if all(not value.startswith(('"', "'")) for value in values):
                    lines[i] = array[1] + ': ' + json.dumps(values, ensure_ascii=False)
        scalar = re.fullmatch(r'(source_article|source_title):\s*(&quot;.*)', line)
        if scalar:
            lines[i] = scalar[1] + ': ' + json.dumps(scalar[2], ensure_ascii=False)
    return text[:match.start(1)] + '\n'.join(lines) + text[match.end(1):]


# 【审计／可选修复】扫描可见 Markdown，检查原文完整性、摘要失效、事实引用和链接；repair=True 时备份后修围栏／可确定元数据／唯一裸链接并归档旧原文。
# 不会用 LLM，也不重写受控摘要或原文版本；返回问题和数量，不保证语义正确。
def audit(wiki_dir: Path, repair=False) -> dict:
    wiki_dir = Path(wiki_dir).resolve()
    sources = SourceStore(wiki_dir)
    files = [p for p in wiki_dir.rglob('*.md') if not any(x.startswith('.') for x in p.relative_to(wiki_dir).parts)
             and not p.is_symlink() and p.resolve().is_relative_to(wiki_dir)]
    by_stem = {}
    for p in files:
        by_stem.setdefault(p.stem, []).append(p.relative_to(wiki_dir).as_posix())
    report = {'pages': len(files), 'repaired_pages': 0, 'archived_sources': 0, 'facts': 0,
              'citations': 0, 'summaries': 0, 'stale_summaries': 0, 'issues': [], 'backups': []}
    for p in files:
        relative = p.relative_to(wiki_dir).as_posix()
        original = p.read_bytes().decode('utf-8')
        if relative.startswith('sources/versions/'):
            try:
                ref = json.loads(p.with_suffix('.json').read_bytes().decode('utf-8'))
                sources.get(ref['source_id'], ref['version_id'])
            except (OSError, ValueError, KeyError) as exc:
                report['issues'].append({'path': relative, 'kind': 'source_integrity', 'detail': str(exc)})
            continue
        if relative.startswith('summaries/'):
            try:
                summary = SummaryStore(wiki_dir).read(relative)
                report['summaries'] += 1
                if summary['status'] == 'stale':
                    report['stale_summaries'] += 1
                    report['issues'].append({'path': relative, 'kind': 'stale_summary', 'detail': summary['stale_reasons']})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                report['issues'].append({'path': relative, 'kind': 'summary_integrity', 'detail': str(exc)})
            # Generated summary records and embedded original quotes are never repaired in place.
            continue
        text = unwrap_markdown(original)
        if text != original:
            report['issues'].append({'path': relative, 'kind': 'whole_page_fence', 'repaired': repair})
        try:
            meta, body = frontmatter(text)
        except (ValueError, yaml.YAMLError) as exc:
            fixed_text = repair_metadata(text)
            repaired = False
            try:
                meta, body = frontmatter(fixed_text)
                if repair:
                    text, repaired = fixed_text, True
            except (ValueError, yaml.YAMLError):
                meta = {}
            report['issues'].append({'path': relative, 'kind': 'frontmatter', 'detail': str(exc), 'repaired': repaired})
        # 【局部链接处理】检查 wikilink 目标并记录问题；仅 repair 模式且裸名称恰有一个候选时替换路径，保留标签和锚点。
        def resolve(match):
            link = match.group(1)
            target, sep, label = link.partition('|')
            target, anchor_sep, anchor = target.partition('#')
            path = target if target.endswith('.md') else target + '.md'
            try:
                found = safe_path(wiki_dir, path).is_file()
            except ValueError:
                found = False
            if found:
                return match.group(0)
            # Only unique exact bare names can be repaired; no substring/fuzzy guesses.
            candidates = by_stem.get(target.removesuffix('.md'), []) if '/' not in target else []
            fixed = repair and len(candidates) == 1
            report['issues'].append({'path': relative, 'kind': 'broken_link', 'target': link,
                                     'candidates': candidates, 'repaired': fixed})
            if fixed:
                return '[[' + candidates[0].removesuffix('.md') + (anchor_sep + anchor if anchor_sep else '') + (sep + label if sep else '') + ']]'
            return match.group(0)
        # Do not rewrite managed blocks or citations; only validate their links below.
        if '<!-- wiki-facts:start -->' not in text:
            text = re.sub(r'\[\[([^\]]+)\]\]', resolve, text)
        try:
            state = fact_state(text)
            ids = {f['id'] for f in state['facts']}
            for f in state['facts']:
                report['facts'] += 1
                for key in ('conflicts_with', 'supersedes'):
                    if any(fid not in ids for fid in f.get(key, [])):
                        raise ValueError(f'dangling {key}')
                for c in f['citations']:
                    sources.validate_citation(c)
                    report['citations'] += 1
            for link in state.get('links', []):
                if not safe_path(wiki_dir, link).is_file():
                    report['issues'].append({'path': relative, 'kind': 'broken_managed_link', 'target': link})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report['issues'].append({'path': relative, 'kind': 'fact_integrity', 'detail': str(exc)})
        if relative.startswith('sources/articles/') and repair:
            sources.archive(p.as_uri(), original, str(meta.get('source_title', p.stem)))
            report['archived_sources'] += 1
        if relative.startswith('sources/digests/') and not relative.endswith('_index.md'):
            article = meta.get('source_article')
            if not isinstance(article, str) or not (wiki_dir / 'sources/articles' / (article.removesuffix('.md') + '.md')).is_file():
                report['issues'].append({'path': relative, 'kind': 'unresolved_legacy_source', 'source_article': article})
        if repair and text != original:
            backup = wiki_dir / '.repair-backups' / digest(original) / relative
            atomic_write(backup, original)
            atomic_write(p, text)
            report['repaired_pages'] += 1
            report['backups'].append(str(backup))
    retriever = WikiRetriever(wiki_dir)
    retriever.load()
    report['indexed_pages'] = len(retriever.pages)
    report['parse_errors'] = retriever.errors
    report['unresolved'] = sum(not issue.get('repaired', False) for issue in report['issues'])
    return report


# 【审计 CLI】读取 wiki-dir 和 repair 选项，执行 audit 并写 output；有未解决问题不自动转成非零退出码，需查看报告。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wiki-dir', type=Path, required=True)
    parser.add_argument('--repair', action='store_true', help='Apply deterministic fixes with per-file backups and archive original sources.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.wiki_dir, args.repair)
    atomic_write(args.output, json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('issues', 'backups', 'parse_errors')}))


if __name__ == '__main__':
    main()
