"""Resume summary batches with live model-call progress, without ingestion or QA."""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from .build_summaries import build_summaries
from .llm_client import call_llm_with_tools
from .wiki_store import atomic_write


# 【进度输出】带当前本地时分秒打印并立即 flush，让长摘要任务的进度可见。
def log(text):
    print(f'[{datetime.now():%H:%M:%S}] {text}', flush=True)


# 【摘要续跑／LLM】按轮调用 build_summaries 并保存独立报告，处理输入完整性、无进展及轮数上限；复用成功／跳过／失败缓存，因此队列清空不等于全部摘要成功。
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wiki-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--max-rounds', type=int, default=200)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.wiki_dir.is_dir() or args.batch_size < 1 or args.max_rounds < 1:
        parser.error('existing wiki directory and positive batch size / max rounds required')
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    calls, consecutive_failures = 0, 0

    # 【调用包装／LLM】打印本次候选组和耗时，转发真实模型调用并计连续无响应；已有三次连续失败时，在下一次调用入口停止，不把格式校验失败算无响应。
    def call(messages, **kwargs):
        nonlocal calls, consecutive_failures
        if consecutive_failures >= 3:
            raise SystemExit('Stopped after 3 consecutive model-call failures; completed summaries are saved.')
        calls += 1
        pages = json.loads(messages[1]['content'])['pages']
        log(f'Call {calls} START: ' + ' + '.join(p['path'] for p in pages))
        started = time.monotonic()
        result = call_llm_with_tools(messages, **kwargs)
        consecutive_failures = consecutive_failures + 1 if result is None else 0
        log(f'Call {calls} response {"received (validation follows)" if result is not None else "FAILED"}; {time.monotonic() - started:.1f}s')
        return result

    for round_no in range(1, args.max_rounds + 1):
        log(f'Round {round_no}: scanning candidates and cache...')
        report = build_summaries(args.wiki_dir, call=call, limit=args.batch_size, dry_run=args.dry_run)
        if not args.dry_run:
            atomic_write(args.output_dir / f'summary-resume-{run_id}-{round_no:03d}.json',
                         json.dumps(report, ensure_ascii=False, indent=2))
        log('Round result: ' + ' | '.join(f'{k}={report[k]}' for k in
            ('candidate_groups', 'created', 'skipped', 'failed', 'cached', 'pending')))
        for item in report['items']:
            if item['status'] == 'failed':
                log(f"FAILED {item['path']}: {item['error']}")
        if report['input_issues']:
            raise SystemExit('Input integrity issues detected; inspect the round report before continuing.')
        if args.dry_run:
            return
        if report['pending'] == 0 and not report['failed']:
            log('Queue drained. Cached entries can include previously failed groups; this is not an all-success claim.')
            return
        if not report['llm_calls']:
            raise SystemExit('No model-call progress; inspect failed items in the round report.')
    raise SystemExit('Round limit reached; rerun to continue from saved cache.')


if __name__ == '__main__':
    main()
