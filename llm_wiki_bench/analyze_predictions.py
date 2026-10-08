"""Offline error extraction; uses the repository evaluator without changing scores.

python3 -m llm_wiki_bench.analyze_predictions --predictions PATH --qa-pairs PATH
Diagnostic categories describe observable signals, not proven semantic causes.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .evaluate import DATA_DIR, _normalize_answer, score_against_aliases

CATEGORIES = {
    'runtime_failure': ('运行中断', '检查 model_error/error 日志；补充错误详情、有限重试及断点续跑。'),
    'budget_exhausted': ('检索预算耗尽', '检查重复检索和提交校验失败；预留提交及修复预算，按新增证据决定是否继续检索。'),
    'abstention': ('未作答', '检查 unresolved requirement；核对问题歧义、知识页缺失及原文回退，不强迫无依据回答。'),
    'answer_type_mismatch': ('布尔答案类型不匹配', '先确定所求对象及答案类型，检查是否误将实体问题当成是非题。'),
    'prediction_contains_gold': ('预测包含标准答案片段', '检查多余解释、括号及枚举；优先抽取最短完整证据片段，保留必要单位与限定条件。'),
    'gold_contains_prediction': ('预测是标准答案的片段', '检查全名、单位、地点层级及遗漏限定；区分别名差异与实质信息缺失。'),
    'partial_overlap': ('部分词项重叠', '复核别名、词序、数值及限定条件；词项重叠不代表事实正确。'),
    'no_overlap': ('无词项重叠', '复核选错实体、关系、属性及标准答案异常；无词项重叠也可能是简称或同义表达。'),
}


def read_jsonl(path: Path) -> tuple[dict, list]:
    """Last prediction wins, as in evaluate.py; report duplicate IDs explicitly."""
    records, duplicates = {}, []
    with path.open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict) or not isinstance(item.get('id'), str):
                    raise ValueError('each record requires a string id')
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path}:{line_no}: {exc}') from exc
            if item['id'] in records:
                duplicates.append(item['id'])
            records[item['id']] = item
    return records, duplicates


def contains_span(text: str, span: str) -> bool:
    text, span = _normalize_answer(text), _normalize_answer(span)
    return bool(span) and f' {span} ' in f' {text} '


def diagnose(qa: dict, pred: dict, scores: dict) -> dict:
    answer = pred.get('prediction') or ''
    gold = [qa['answer'], *(qa.get('answer_aliases') or [])]
    norm = _normalize_answer(answer)
    stop = pred.get('stop_reason')
    if pred.get('error') or stop in {'model_error', 'error'}:
        category = 'runtime_failure'
    elif stop in {'budget_exhausted', 'turn_limit'}:
        category = 'budget_exhausted'
    elif norm in {'', 'unknown'}:
        category = 'abstention'
    elif (norm in {'yes', 'no'}) != any(_normalize_answer(g) in {'yes', 'no'} for g in gold):
        category = 'answer_type_mismatch'
    elif any(contains_span(answer, g) for g in gold):
        category = 'prediction_contains_gold'
    elif any(contains_span(g, answer) for g in gold):
        category = 'gold_contains_prediction'
    elif scores['f1'] > 0:
        category = 'partial_overlap'
    else:
        category = 'no_overlap'

    titles_known = 'retrieved_titles' in pred
    read = {_normalize_answer(t) for t in pred.get('retrieved_titles', []) or []}
    supporting = qa.get('supporting_titles', []) or []
    missing = [t for t in supporting if _normalize_answer(t) not in read] if titles_known else None
    recall = (len(supporting) - len(missing)) / len(supporting) if supporting and titles_known else None
    passages = [*pred.get('knowledge_evidence', []), *pred.get('article_evidence', [])]
    hits = []
    # Lexical presence only: deliberately excludes yes/no, which are not factual evidence spans.
    for passage in passages:
        text = passage.get('text') or passage.get('quote') or ''
        if any(_normalize_answer(g) not in {'yes', 'no'} and contains_span(text, g) for g in gold):
            hits.append(passage.get('page') or passage.get('article'))
    tool_errors = []
    for call in pred.get('tool_calls', []):
        result = call.get('result')
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                continue
        if isinstance(result, dict) and result.get('error'):
            tool_errors.append({'step': call.get('step'), 'tool': call.get('tool'), 'error': result['error']})
    return {
        'category': category, 'category_label': CATEGORIES[category][0],
        'diagnosis_status': 'heuristic_requires_review',
        'suggestion': CATEGORIES[category][1],
        'missing_supporting_titles': missing, 'supporting_title_recall_proxy': recall,
        'gold_span_in_read_evidence': bool(hits) if passages else None,
        'gold_span_evidence_pages': sorted(set(filter(None, hits))),
        'tool_errors': tool_errors,
    }


def analyze(qa_pairs: dict, predictions: dict) -> tuple[dict, list[dict], list[dict]]:
    errors, missing, categories, types = [], [], Counter(), {}
    em_sum = f1_sum = 0.0
    matched = 0
    for qid, qa in qa_pairs.items():
        if qid not in predictions:
            missing.append(qa)
            continue
        pred = predictions[qid]
        if not isinstance(qa.get('answer'), str) or not isinstance(pred.get('prediction', '') or '', str):
            raise ValueError(f'{qid}: answer and prediction must be strings')
        if pred.get('question') and pred['question'] != qa['question']:
            raise ValueError(f'{qid}: prediction question differs from QA question')
        if 'gold_answer' in pred and pred['gold_answer'] != qa['answer']:
            raise ValueError(f'{qid}: embedded gold_answer differs from QA answer')
        scores = score_against_aliases(pred.get('prediction') or '', [qa['answer'], *(qa.get('answer_aliases') or [])])
        matched += 1
        em_sum += scores['em']
        f1_sum += scores['f1']
        group = types.setdefault(qa.get('type', 'unknown'), {'total': 0, 'errors': 0})
        group['total'] += 1
        if scores['em'] == 1:
            continue
        diagnosis = diagnose(qa, pred, scores)
        categories[diagnosis['category']] += 1
        group['errors'] += 1
        errors.append({
            'id': qid, 'question': qa['question'], 'gold_answer': qa['answer'],
            'answer_aliases': qa.get('answer_aliases', []), 'prediction': pred.get('prediction') or '',
            **scores, 'type': qa.get('type', 'unknown'), 'supporting_titles': qa.get('supporting_titles', []),
            **diagnosis, 'stop_reason': pred.get('stop_reason'),
            'evidence_gaps': pred.get('evidence_gaps', []),
            'prediction_record': pred,
        })
    summary = {
        'criterion': 'repository evaluate.score_against_aliases EM == 0; not a semantic correctness verdict',
        'qa_total': len(qa_pairs), 'prediction_unique_total': len(predictions), 'evaluated': matched,
        'correct_em': int(em_sum), 'wrong_em': len(errors), 'missing_predictions': len(missing),
        'unmatched_prediction_ids': sorted(set(predictions) - set(qa_pairs)),
        'em': em_sum / matched if matched else None, 'f1': f1_sum / matched if matched else None,
        'zero_f1_errors': sum(e['f1'] == 0 for e in errors),
        'categories': dict(categories), 'by_type': types,
        'error_stop_reasons': dict(Counter(e['stop_reason'] for e in errors)),
        'errors_missing_supporting_titles': sum(bool(e['missing_supporting_titles']) for e in errors),
        'errors_with_gold_span_in_read_evidence': sum(e['gold_span_in_read_evidence'] is True for e in errors),
        'errors_with_tool_errors': sum(bool(e['tool_errors']) for e in errors),
    }
    return summary, errors, missing


def write_outputs(output: Path, summary: dict, errors: list, missing: list) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for name, rows in [('wrong_questions.jsonl', errors), ('missing_predictions.jsonl', missing)]:
        with (output / name).open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    fields = ['id', 'question', 'gold_answer', 'prediction', 'em', 'f1', 'type', 'category_label',
              'stop_reason', 'missing_supporting_titles', 'gold_span_in_read_evidence', 'suggestion']
    with (output / 'wrong_questions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for error in errors:
            writer.writerow({**error, 'missing_supporting_titles': json.dumps(error['missing_supporting_titles'], ensure_ascii=False)})
    lines = ['# Prediction 错题诊断', '',
             f"输入：`{summary['predictions_path']}`", '',
             f"按仓库 EM 规则评估 {summary['evaluated']} 题，匹配 {summary['correct_em']}，不匹配 {summary['wrong_em']}；其中 F1=0 有 {summary['zero_f1_errors']} 题。",
             f"QA 中未预测 {summary['missing_predictions']} 题；这些题单独导出，不计入错题/评分分母。", '',
             '自动类别只是可复现的表面信号，不能证明语义正确或根因。标题召回是代理指标：未读 gold 标题不等于证据缺失；读到标准答案片段也不等于完成推理。', '',
             '| 诊断信号（互斥） | 数量 | 优化/复核方向 |', '|---|---:|---|']
    for key, count in summary['categories'].items():
        label, suggestion = CATEGORIES[key]
        lines.append(f'| {label} | {count} | {suggestion} |')
    lines += ['', '## 全部 EM 不匹配问题', '']
    for i, error in enumerate(errors, 1):
        lines += [f"### {i}. {error['id']}", '', error['question'], '',
                  f"- 标准答案：{error['gold_answer']}", f"- 预测：{error['prediction']}",
                  f"- F1：{error['f1']:.4f}；诊断信号：{error['category_label']}；停止：{error['stop_reason']}",
                  f"- 未读支撑标题（代理）：{error['missing_supporting_titles']}",
                  f"- 优化/复核：{error['suggestion']}", '']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', '-p', type=Path, required=True)
    parser.add_argument('--dataset', default='hotpotqa', choices=['hotpotqa', 'musique', '2wikimhqa'])
    parser.add_argument('--qa-pairs', type=Path)
    parser.add_argument('--output-dir', '-o', type=Path)
    args = parser.parse_args()
    qa_path = args.qa_pairs or DATA_DIR / args.dataset / 'qa_pairs.jsonl'
    output = args.output_dir or args.predictions.parent / 'error_analysis'
    try:
        qa, qa_duplicates = read_jsonl(qa_path)
        if qa_duplicates:
            raise ValueError(f'duplicate QA IDs: {qa_duplicates}')
        predictions, duplicates = read_jsonl(args.predictions)
        summary, errors, missing = analyze(qa, predictions)
        if not summary['evaluated']:
            raise ValueError('no predictions matched QA IDs')
        summary.update(predictions_path=str(args.predictions), qa_pairs_path=str(qa_path), duplicate_prediction_ids=duplicates)
        write_outputs(output, summary, errors, missing)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Reports saved to {output}')


if __name__ == '__main__':
    main()
