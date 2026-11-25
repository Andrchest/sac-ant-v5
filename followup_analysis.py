#!/usr/bin/env python3
"""Validate and export the preregistered follow-up at training-run level.

Run from repository root. CSVs and summaries are copied unchanged into a small,
portable research directory, with source hashes. No model or replay is copied.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from followup_campaign import experiment_jobs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(name, job):
    if job.get('status') != 'done' or job.get('exit_code') != 0:
        raise ValueError(f'{name}: run not successfully completed')
    root = Path(job['run_dir'])
    summary_path, csv_path = root / 'summary.json', root / 'evaluation_all.csv'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    args = job['args']
    arg = lambda key: args[args.index(key) + 1]
    algo, seed = arg('--algo'), int(arg('--seed'))
    steps, n = int(arg('--steps')), int(arg('--eval-episodes'))
    config = summary['config']
    if (summary['algo'] != algo or summary['seed'] != seed or summary['steps'] != steps or
            summary['eval_episodes'] != n or not summary['budget_completed']):
        raise ValueError(f'{name}: summary does not match declared job')
    if (config['net'] != [256, 256] or config['lr'] != .0003 or config['batch'] != 256 or
            config['learning_starts'] != 10000 or config['gamma'] != .99 or
            (algo in ('sac', 'td3') and config['tau'] != .005) or
            (algo == 'td3' and config['action_noise_std'] != .1) or
            (algo == 'ppo' and config['ppo_n_steps'] != 2000)):
        raise ValueError(f'{name}: configuration violates protocol')
    with csv_path.open(newline='', encoding='utf-8') as stream:
        rows = [r for r in csv.DictReader(stream) if r['policy'] == f'{algo}_final']
    if len(rows) != n or [int(r['episode_seed']) for r in rows] != list(range(seed + 100000, seed + 100000 + n)):
        raise ValueError(f'{name}: unexpected evaluation rows or seeds')
    values = [float(r['return_']) for r in rows]
    xs = [float(r['x_progress']) for r in rows]
    if not all(math.isfinite(v) for v in values + xs):
        raise ValueError(f'{name}: non-finite evaluation values')
    flags = [r['terminated'] for r in rows]
    if any(x not in ('True', 'False') for x in flags):
        raise ValueError(f'{name}: invalid termination flag')
    falls = sum(v == 'True' for v in flags)
    for actual, claimed in [(statistics.mean(values), summary['final_return_mean']),
                            (statistics.stdev(values), summary['final_return_std']),
                            (statistics.mean(xs), summary['final_x_progress_mean']),
                            (falls / n, summary['final_fall_rate'])]:
        if not math.isclose(actual, claimed, rel_tol=1e-8, abs_tol=1e-7):
            raise ValueError(f'{name}: summary does not match per-episode CSV')
    return summary, rows, summary_path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('research/followup'))
    parser.add_argument('--plot', type=Path, help='also save plot to a published location')
    args = parser.parse_args()
    data = json.loads(args.status.read_text(encoding='utf-8'))
    jobs = data['jobs']
    expected = dict(experiment_jobs())
    if set(jobs) != set(expected):
        raise ValueError(f'Unexpected job IDs: {set(jobs) ^ set(expected)}')
    verified = {}
    for name, job in jobs.items():
        if job['args'] != expected[name]:
            raise ValueError(f'{name}: job differs from preregistered settings')
        verified[name] = validate(name, job)
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = {'status_sha256': sha(args.status), 'sources': {}}
    for name, (summary, rows, summary_path, csv_path) in verified.items():
        dst = args.out / 'runs' / name
        dst.mkdir(parents=True, exist_ok=True)
        for source in (summary_path, csv_path):
            (dst / source.name).write_bytes(source.read_bytes())
        provenance['sources'][name] = {source.name: sha(source) for source in (summary_path, csv_path)}
    # Preserve the exact input bytes whose digest is recorded; the portable status
    # below has different paths and is intentionally a separate file.
    (args.out / 'source_status.json').write_bytes(args.status.read_bytes())
    (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    portable = {'protocol': data['protocol'], 'started': data['started'], 'jobs': {}}
    for name, job in jobs.items():
        portable['jobs'][name] = {'status': job['status'], 'exit_code': job['exit_code'],
                                  'args': job['args'], 'run_dir': (args.out / 'runs' / name).as_posix()}
    (args.out / 'status.json').write_text(json.dumps(portable, indent=2) + '\n', encoding='utf-8')
    lines = ['# Follow-up: preregistered Ant-v5 comparison', '',
             'Protocol: `EXPERIMENT_PROTOCOL.md`. Three independent training seeds per algorithm;',
             '30 paired deterministic reset seeds per seed. Results describe fixed configurations,',
             'not tuned algorithms or a statistically established general ranking.', '',
             '| algorithm | training seed | steps | return mean ± episode SD | x progress (m) | falls |',
             '|---|---:|---:|---:|---:|---:|']
    seed_means = {}
    for algo in ('sac', 'td3', 'ppo'):
        seed_means[algo] = []
        for seed in (60, 61, 62):
            s, rows, _, _ = verified[f'{algo}_{seed}']
            seed_means[algo].append(s['final_return_mean'])
            falls = sum(r['terminated'] == 'True' for r in rows)
            lines.append(f'| {algo.upper()} | {seed} | {s["steps"]:,} | '
                         f'{s["final_return_mean"]:.1f} ± {s["final_return_std"]:.1f} | '
                         f'{s["final_x_progress_mean"]:.1f} | {falls}/30 |')
    lines += ['', '## Descriptive training-seed aggregates', '']
    for algo, vals in seed_means.items():
        lines.append(f'- {algo.upper()}: mean {statistics.mean(vals):.1f}; range {min(vals):.1f}–{max(vals):.1f} (n=3 training runs).')
    lines += ['', 'Paired SAC − TD3 differences by training seed: ' +
              ', '.join(f'{seed}: {a-b:+.1f}' for seed, a, b in zip((60,61,62), seed_means['sac'], seed_means['td3'])) + '.',
              'These are descriptive; no episode-pooled p-value or population superiority claim.', '']
    c, crows, _, _ = verified['sac_control_42_500k']
    lines += ['## Single-seed 500k SAC base control', '',
              f'Seed 42, same final reset protocol as 500k ablations: {c["final_return_mean"]:.1f} ± '
              f'{c["final_return_std"]:.1f}, {sum(r["terminated"] == "True" for r in crows)}/10 falls.',
              'Compare to existing 500k arms only as single-seed exploratory observations; no independent replication.', '',
              'Evaluation episode standard deviations describe reset variability, not training-seed uncertainty.',
              'TD3 uses Gaussian exploration noise σ=0.1. Equal budgets are environment steps, not compute time.',
              'Three seeds and one environment limit generalization. Reset sets overlap across seeds.', '']
    (args.out / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (algo, vals) in enumerate(seed_means.items()):
        ax.scatter([i] * 3, vals, s=65, label=algo.upper())
        for seed, val in zip((60, 61, 62), vals):
            ax.annotate(str(seed), (i, val), xytext=(7, 3), textcoords='offset points', fontsize=8)
        ax.plot([i-.15, i+.15], [statistics.mean(vals)] * 2, color='black')
    ax.set_xticks(range(3), ['SAC', 'TD3', 'PPO'])
    ax.set_ylabel('Mean return per trained seed (30 resets)')
    ax.set_title('Ant-v5, 1M steps: three training seeds per configuration')
    ax.grid(axis='y', alpha=.25)
    fig.tight_layout()
    fig.savefig(args.out / 'followup.png', dpi=140)
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=140)
    plt.close(fig)
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
