"""Compare BM25/entity lookup at k=5/10/15 using title coverage as a proxy.

This is candidate recall, not verified evidence coverage or final-answer quality.
No LLM calls; an optional Python reranker callback can be passed to compare().
"""
import argparse
import json
from pathlib import Path
try:
    from .wiki_retriever import WikiRetriever
    from .wiki_store import atomic_write
except ImportError:
    from wiki_retriever import WikiRetriever
    from wiki_store import atomic_write


# 【检索实验】用完整问题比较 BM25、实体匹配及可选重排在 k=5/10/15 的支撑标题覆盖；无模型调用（外部 reranker 行为由调用者决定），不是最终回答质量或来源蕴含评估。
def compare(wiki_dir, qa_pairs, reranker=None):
    results = []
    retriever = WikiRetriever(wiki_dir)
    retriever.load()
    modes = ['bm25', 'exact_then_bm25', 'exact'] + (['reranked_bm25'] if reranker else [])
    for mode in modes:
        retriever.reranker = reranker if mode == 'reranked_bm25' else None
        totals = {k: {'title_recall': 0, 'all_titles': 0, 'queries': 0} for k in (5, 10, 15)}
        for qa in qa_pairs:
            gold = {retriever._entity_key(t) for t in qa.get('supporting_titles', [])}
            if not gold:
                continue
            hits = retriever.search(qa['question'], 15, mode='bm25' if mode == 'reranked_bm25' else mode)
            for k, counts in totals.items():
                found = set()
                for hit in hits[:k]:
                    pg = hit.page
                    found.update(retriever._entity_key(str(v)) for v in
                                 [pg.name, *pg.aliases, pg.metadata.get('source_title', '')])
                counts['queries'] += 1
                counts['title_recall'] += len(gold & found) / len(gold)
                counts['all_titles'] += gold <= found
        for k, counts in totals.items():
            n = counts['queries']
            results.append({'mode': mode, 'k': k, 'queries': n,
                            'mean_supporting_title_recall': counts['title_recall'] / n if n else None,
                            'all_supporting_titles_rate': counts['all_titles'] / n if n else None})
    return {'metric': 'candidate supporting-title coverage proxy; whole question is the query, exact entity mode needs agent query reformulation', 'results': results}


# 【检索实验 CLI】读取前 N 条 QA，运行 compare 并写报告；不执行 QA Agent。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wiki-dir', type=Path, required=True)
    parser.add_argument('--qa', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    qa = [json.loads(line) for line in args.qa.read_text().splitlines() if line.strip()][:args.limit]
    result = compare(args.wiki_dir, qa)
    atomic_write(args.output, json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
