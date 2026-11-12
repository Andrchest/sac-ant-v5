"""Orchestrate the SAC × Ant-v5 ablation campaign.

Runs experiments from campaign_experiments.py in priority order across N
parallel worker subprocesses (each a separate run_experiment.py process, so
they are fully parallel-safe). Respects a wall-clock deadline: an experiment
is only started if its estimated runtime fits before the deadline.

Writes a live campaign_summary.json and campaign_report.md into the output
directory. Ctrl+C marks remaining experiments as pending and finalises the
report; completed experiments are never lost.

Example:
    python ablation_campaign.py --workers 3 --threads 4 \
        --deadline 2026-11-13T07:00:00+03:00 \
        --baseline-seconds-per-million 9000 \
        --output artifacts/campaign
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from campaign_experiments import EXPERIMENTS, REFERENCE  # noqa: E402

NET_SCALE = {'256,256': 1.0, '128,128': 0.6, '64,64': 0.4}
ALGO_SCALE = {'sac': 1.0, 'td3': 0.9, 'ppo': 1.1}
EVAL_OVERHEAD_S = 120  # random + initial + final + checkpoint evals, ~2 min


def parse_deadline(text):
    return datetime.fromisoformat(text)


def estimate_seconds(exp, baseline_s_per_m):
    net = exp.get('net', '256,256')
    scale = NET_SCALE.get(net, 1.0) * ALGO_SCALE.get(exp['algo'], 1.0)
    return exp['steps'] / 1e6 * baseline_s_per_m * scale + EVAL_OVERHEAD_S


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--threads', type=int, default=4,
                   help='torch threads per worker (workers*threads <= CPU cores)')
    p.add_argument('--deadline', required=True, help='ISO wall-clock deadline (MSK)')
    p.add_argument('--baseline-seconds-per-million', type=float, default=9000,
                   help='measured seconds per 1M steps for SAC [256,256] at this thread count')
    p.add_argument('--output', default='artifacts/campaign')
    p.add_argument('--limit', type=int, default=0, help='run at most N experiments (0 = all)')
    args = p.parse_args()

    assert args.workers * args.threads <= 12, 'workers*threads must be <= 12 cores'
    deadline = parse_deadline(args.deadline)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    state = {e['id']: dict(status='pending', exp=e) for e in
             sorted(EXPERIMENTS, key=lambda x: (x['priority'], x['id']))}
    if args.limit:
        kept = list(state)[:args.limit]
        state = {k: state[k] for k in kept}

    stop = {'flag': False}
    lock = threading.Lock()

    def on_sigint(signum, frame):
        stop['flag'] = True
        print('SIGINT: finishing running experiments, marking rest pending...', flush=True)

    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    def write_report():
        with lock:
            _write_report_locked()
    def _write_report_locked():
        rows = []
        for exp_id, s in state.items():
            e = s['exp']
            rows.append(dict(id=exp_id, priority=e['priority'], algo=e['algo'], seed=e['seed'],
                             steps=e['steps'], net=e.get('net'), lr=e.get('lr'),
                             batch=e.get('batch'), status=s['status'],
                             final_return_mean=s.get('final_return_mean'),
                             final_return_std=s.get('final_return_std'),
                             final_x_progress_mean=s.get('final_x_progress_mean'),
                             final_fall_rate=s.get('final_fall_rate'),
                             train_seconds=s.get('train_seconds'), run_dir=s.get('run_dir')))
        summary = dict(campaign='sac_ant_v5_ablation', deadline=args.deadline,
                       workers=args.workers, threads=args.threads,
                       baseline_seconds_per_million=args.baseline_seconds_per_million,
                       reference=REFERENCE, experiments=rows,
                       updated=datetime.now().isoformat())
        (out / 'campaign_summary.json').write_text(json.dumps(summary, indent=2))
        write_markdown(out / 'campaign_report.md', summary)

    def run_one(exp_id):
        e = state[exp_id]['exp']
        cmd = [sys.executable, 'run_experiment.py', '--algo', e['algo'],
               '--seed', str(e['seed']), '--steps', str(e['steps']),
               '--net', str(e.get('net', '256,256')), '--lr', str(e.get('lr', 3e-4)),
               '--batch', str(e.get('batch', 256)), '--gamma', str(e.get('gamma', 0.99)),
               '--tau', str(e.get('tau', 0.005)), '--ent-coef', str(e.get('ent_coef', 'auto')),
               '--learning-starts', str(e.get('learning_starts', 10_000)),
               '--eval-episodes', str(e['eval_episodes']), '--threads', str(args.threads),
               '--output', 'artifacts', '--tag', e.get('tag', exp_id)]
        if e.get('checkpoint_every'):
            cmd += ['--checkpoint-every', str(e['checkpoint_every']),
                    '--checkpoint-eval-episodes', str(e.get('checkpoint_eval_episodes', 10))]
        log = out / f'{exp_id}.log'
        with lock:
            state[exp_id]['status'] = 'running'
            _write_report_locked()
        t0 = time.time()
        with open(log, 'w') as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
        dt = time.time() - t0
        # Parse the authoritative summary event from this run's own log.
        s = None
        try:
            for line in reversed(log.read_text().splitlines()):
                line = line.strip()
                if not line.startswith('{'):
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get('event') == 'summary':
                    s = ev.get('summary')
                    break
        except OSError:
            pass
        if s:
            with lock:
                state[exp_id].update(status='done',
                                     final_return_mean=s.get('final_return_mean'),
                                     final_return_std=s.get('final_return_std'),
                                     final_x_progress_mean=s.get('final_x_progress_mean'),
                                     final_fall_rate=s.get('final_fall_rate'),
                                     train_seconds=s.get('train_seconds'),
                                     run_dir=s.get('run_dir'),
                                     checkpoints=s.get('checkpoints'))
        else:
            with lock:
                state[exp_id]['status'] = 'error' if proc.returncode != 0 else 'done_no_summary'
        with lock:
            state[exp_id]['wall_seconds'] = round(dt, 1)
        print(f'[{exp_id}] {state[exp_id]["status"]} in {dt:.0f}s', flush=True)
        write_report()

    write_report()
    queue = list(state.keys())
    running = {}
    while queue or running:
        if stop['flag']:
            break
        # launch as many as fit
        while queue and len(running) < args.workers:
            exp_id = queue[0]
            est = estimate_seconds(state[exp_id]['exp'], args.baseline_seconds_per_million)
            now = time.time()
            if now + est > deadline.timestamp():
                with lock:
                    state[exp_id]['status'] = 'skipped_deadline'
                    _write_report_locked()
                print(f'[{exp_id}] skipped: est {est/60:.0f}min does not fit before deadline',
                      flush=True)
                queue.pop(0)
                continue
            queue.pop(0)
            print(f'[{exp_id}] launch (est {est/60:.0f}min)', flush=True)
            running[exp_id] = _Thread(run_one, (exp_id,))
            running[exp_id].start()
        # reap finished
        for exp_id in list(running):
            th = running[exp_id]
            if not th.is_alive():
                th.join()
                del running[exp_id]
        time.sleep(2)

    for exp_id in queue:
        with lock:
            state[exp_id]['status'] = 'pending'
            _write_report_locked()
    print('campaign finalised. See', out / 'campaign_report.md', flush=True)


class _Thread(threading.Thread):
    def __init__(self, fn, arg):
        super().__init__(daemon=True)
        self._fn, self._arg = fn, arg

    def run(self):
        self._fn(*self._arg)


def write_markdown(path, summary):
    ref = summary['reference']
    lines = ['# SAC × Ant-v5 — отчёт кампании абляций', '',
             f'Дедлайн: `{summary["deadline"]}` · воркеры: {summary["workers"]} · '
             f'потоков/воркер: {summary["threads"]}', '',
             '## Итоги по экспериментам', '',
             '| id | prio | algo | seed | steps | net | lr | batch | статус | final mean | final std | x-progress | fall | сек |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in summary['experiments']:
        def f(v, nd=1):
            return f'{v:.{nd}f}' if isinstance(v, (int, float)) else (v if v is not None else '—')
        lines.append(f'| {r["id"]} | {r["priority"]} | {r["algo"]} | {r["seed"]} | {r["steps"]:,} '
                     f'| {r["net"]} | {r["lr"]} | {r["batch"]} | {r["status"]} '
                     f'| {f(r["final_return_mean"])} | {f(r["final_return_std"])} '
                     f'| {f(r["final_x_progress_mean"])} | {f(r["final_fall_rate"], 2)} '
                     f'| {f(r["train_seconds"], 0)} |')
    lines += ['', '## Базовый эталон (предшествующий полный запуск)', '',
              f'- SAC seed {ref["seed"]}, {ref["steps"]:,} шагов, net {ref["net"]}: '
              f'final **{ref["final_return_mean"]:.2f} ± {ref["final_return_std"]:.2f}** '
              f'(n={ref["eval_episodes"]}), x-progress {ref["final_x_progress_mean"]:.2f} m, '
              f'fall {ref["final_fall_rate"]:.0%}.', '',
              '## Конвергенция (checkpoint-оценки)', '']
    for r in summary['experiments']:
        if r.get('checkpoints'):
            lines.append(f'### {r["id"]}')
            lines.append('| step | return mean | return std | x-progress | fall |')
            lines.append('|---|---|---|---|---|')
            for c in r['checkpoints']:
                lines.append(f'| {c["step"]:,} | {c["return_mean"]:.1f} | {c["return_std"]:.1f} '
                             f'| {c["x_progress_mean"]:.2f} | {c["fall_rate"]:.2f} |')
            lines.append('')
    path.write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
