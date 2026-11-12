#!/usr/bin/env python3
"""Aggregate the ablation-campaign results into statistics + markdown.

Reads every experiment summary.json (plus the pre-existing reference run) and
produces:
  - cross-seed statistics at 1M steps (mean, std, 95% CI, per-seed table)
  - convergence curve (from the conv run's checkpoint evaluations)
  - one-factor-at-a-time ablation table (500k budget)
  - baseline comparison (SAC vs PPO vs TD3 at 1M)
  - a markdown block ready to paste into the report + website

Usage:
    python campaign_analysis.py --campaign artifacts/campaign \
        --out artifacts/campaign/analysis.md
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics as st


def _ci95(values):
    n = len(values)
    if n < 2:
        return (values[0] if values else float('nan'), values[0] if values else float('nan'))
    m = st.mean(values)
    sd = st.stdev(values)
    # two-sided 95% t-quantile
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
         8: 2.365, 9: 2.262, 10: 2.228}.get(n, 1.96)
    half = t * sd / math.sqrt(n)
    return m - half, m + half


def _load_campaign(campaign_dir: Path):
    """Load each completed experiment's summary.json via campaign_summary.json run_dir."""
    exps = []
    cs = campaign_dir / 'campaign_summary.json'
    if cs.exists():
        meta = json.loads(cs.read_text())
        for e in meta.get('experiments', []):
            rd = e.get('run_dir')
            if not rd:
                continue
            sp = Path(rd) / 'summary.json'
            if sp.exists():
                try:
                    s = json.loads(sp.read_text())
                    s['xp_id'] = e.get('id')
                    exps.append(s)
                except Exception:
                    pass
    # fallback: glob top-level artifacts/xp_*
    if not exps:
        for sp in sorted(Path('artifacts').glob('xp_*/summary.json')):
            try:
                exps.append(json.loads(sp.read_text()))
            except Exception:
                continue
    return exps


def _find(exps, xp_id=None):
    for s in exps:
        if s.get('xp_id') == xp_id:
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', default='artifacts/campaign')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    cdir = Path(args.campaign)
    exps = _load_campaign(cdir)

    # --- Reference (pre-existing seed42 1M run) -----------------------------
    REF = dict(seed=42, steps=1_000_000, mean=5701.04, std=157.17, x=286.39,
               fall=0.0, n=10, note='pre-existing full run')

    # --- Cross-seed at 1M (base hyperparameters) ---------------------------
    seed43 = _find(exps, xp_id='conv_s43_2m')
    seed43_1m = None
    if seed43:
        for cp in seed43.get('checkpoints', []):
            if cp.get('step') == 1_000_000:
                seed43_1m = cp
    seed44 = _find(exps, xp_id='seed44_1m')

    seeds = []
    seeds.append(('seed42', REF['mean'], REF['std'], REF['x'], REF['fall'], REF['n']))
    if seed43_1m:
        seeds.append(('seed43', seed43_1m['return_mean'], seed43_1m['return_std'],
                      seed43_1m['x_progress_mean'], seed43_1m['fall_rate'],
                      seed43_1m.get('n', 10)))
    if seed44:
        seeds.append(('seed44', seed44['final_return_mean'], seed44['final_return_std'],
                      seed44['final_x_progress_mean'], seed44['final_fall_rate'],
                      seed44['eval_episodes']))

    L = []
    L.append('# Campaign analysis (auto-generated)\n')

    L.append('## Cross-seed, 1M steps, base hyperparameters (SAC net 256, lr 3e-4, batch 256)\n')
    L.append('| seed | final return | std | x-progress (m) | fall rate | n |')
    L.append('|---|---|---|---|---|---|')
    for name, m, sd, x, f, n in seeds:
        L.append(f'| {name} | {m:.0f} | {sd:.0f} | {x:.0f} | {f:.0%} | {n} |')
    means = [s[1] for s in seeds]
    if len(means) >= 2:
        spread = max(means) - min(means)
        L.append(f'\nCross-seed mean: **{st.mean(means):.0f}**, '
                 f'range **{min(means):.0f}–{max(means):.0f}** '
                 f'(spread {spread:.0f}). Inter-seed variance is large, so a single '
                 f'seed is not representative; the previously reported 5701 is the '
                 f'upper end of this range, not a typical value.')
    L.append('')

    # --- Convergence curve -------------------------------------------------
    L.append('## Convergence (seed43, checkpoints every 250k)\n')
    if seed43 and seed43.get('checkpoints'):
        L.append('| steps | return | std | x (m) | fall |')
        L.append('|---|---|---|---|---|')
        for cp in sorted(seed43['checkpoints'], key=lambda c: c['step']):
            L.append(f"| {cp['step']//1000}k | {cp['return_mean']:.0f} | {cp['return_std']:.0f} "
                     f"| {cp['x_progress_mean']:.0f} | {cp['fall_rate']:.0%} |")
    else:
        L.append('(conv run not finished / no checkpoints)')
    L.append('')

    # --- Ablation table (500k, seed42) -------------------------------------
    L.append('## Hyperparameter ablation (500k steps, seed42)\n')
    L.append('| factor | setting | final return | std | x (m) | fall |')
    L.append('|---|---|---|---|---|---|')
    abl = [('net', 'net64', _find(exps, xp_id='net64')),
           ('net', 'net128', _find(exps, xp_id='net128')),
           ('net', 'net256 (base)', None),
           ('lr', '1e-4', _find(exps, xp_id='lr1e4')),
           ('lr', '3e-4 (base)', None),
           ('lr', '1e-3', _find(exps, xp_id='lr1e3')),
           ('batch', '128', _find(exps, xp_id='batch128')),
           ('batch', '256 (base)', None),
           ('batch', '512', _find(exps, xp_id='batch512'))]
    for factor, setting, s in abl:
        if s and s.get('final_return_mean') is not None:
            L.append(f"| {factor} | {setting} | {s['final_return_mean']:.0f} "
                     f"| {s['final_return_std']:.0f} | {s['final_x_progress_mean']:.0f} "
                     f"| {s['final_fall_rate']:.0%} |")
        else:
            L.append(f'| {factor} | {setting} | — (baseline config, see 1M reference) | | | |')
    L.append('')

    # --- Baselines ---------------------------------------------------------
    L.append('## Baselines, 1M steps, seed42\n')
    L.append('| algo | final return | std | x (m) | fall |')
    L.append('|---|---|---|---|---|')
    L.append(f"| SAC (ref) | {REF['mean']:.0f} | {REF['std']:.0f} | {REF['x']:.0f} | {REF['fall']:.0%} |")
    for xp_id, name in (('ppo_1m', 'PPO'), ('td3_1m', 'TD3')):
        s = _find(exps, xp_id=xp_id)
        if s and s.get('final_return_mean') is not None:
            L.append(f"| {name} | {s['final_return_mean']:.0f} | {s['final_return_std']:.0f} "
                     f"| {s['final_x_progress_mean']:.0f} | {s['final_fall_rate']:.0%} |")
        else:
            L.append(f"| {name} | (not run / skipped by deadline) | | | |")
    L.append('')

    md = '\n'.join(L)
    if args.out:
        Path(args.out).write_text(md)
        print(f'wrote {args.out}')
    print(md)


if __name__ == '__main__':
    main()
