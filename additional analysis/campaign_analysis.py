#!/usr/bin/env python3
"""Aggregate the ablation-campaign results into statistics + markdown.

Reads every experiment summary.json (plus the pre-existing reference run) and
produces:
  - descriptive cross-seed statistics at 1M steps (per-seed table and range)
  - checkpoint returns along the seed 43 training run
  - one-factor-at-a-time ablation table (500k budget)
  - single-seed baseline comparison at a requested 1M steps
  - a markdown block ready to paste into the report + website

Usage:
    python "additional analysis/campaign_analysis.py" --campaign research/campaign \
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


REQUIRED_EXPERIMENT_IDS = frozenset({
    'conv_s43_2m', 'seed44_1m', 'net128', 'net64', 'lr1e4', 'lr1e3',
    'batch128', 'batch512', 'ppo_1m', 'td3_1m',
})
REQUIRED_CHECKPOINT_STEPS = frozenset(range(250_000, 2_000_001, 250_000))


def _validate_complete(exps):
    """Refuse to generate a full report from incomplete or invalid inputs."""
    ids = [s.get('xp_id') for s in exps]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate experiment IDs in campaign index')
    missing = REQUIRED_EXPERIMENT_IDS - set(ids)
    if missing:
        raise ValueError(f'Missing required experiments: {", ".join(sorted(missing))}')
    by_id = {s['xp_id']: s for s in exps}
    metrics = ('final_return_mean', 'final_return_std', 'final_x_progress_mean',
               'final_fall_rate', 'eval_episodes')
    for xp_id in sorted(REQUIRED_EXPERIMENT_IDS):
        s = by_id[xp_id]
        if s.get('xp_status') != 'done' or s.get('status') != 'done':
            raise ValueError(f'Experiment {xp_id} is not completed')
        for key in metrics:
            value = s.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Invalid {key} for {xp_id}: {value!r}')
        if s['eval_episodes'] <= 0 or not 0 <= s['final_fall_rate'] <= 1 or s['final_return_std'] < 0:
            raise ValueError(f'Invalid evaluation metrics for {xp_id}')
    checkpoints = by_id['conv_s43_2m'].get('checkpoints')
    if not isinstance(checkpoints, list) or any(not isinstance(c, dict) for c in checkpoints):
        raise ValueError('Invalid convergence checkpoints')
    steps = [c.get('step') for c in checkpoints]
    if len(steps) != len(REQUIRED_CHECKPOINT_STEPS) or set(steps) != REQUIRED_CHECKPOINT_STEPS:
        raise ValueError('Missing, duplicate or unexpected convergence checkpoints')
    for c in checkpoints:
        for key in ('return_mean', 'return_std', 'x_progress_mean', 'fall_rate'):
            value = c.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Invalid checkpoint {key} at {c["step"]}: {value!r}')
        if c['return_std'] < 0 or not 0 <= c['fall_rate'] <= 1:
            raise ValueError(f'Invalid checkpoint metrics at {c["step"]}')


def _load_campaign(campaign_dir: Path):
    """Load each completed experiment's summary.json via campaign_summary.json run_dir."""
    exps = []
    cs = campaign_dir / 'campaign_summary.json'
    if not cs.is_file():
        derived = campaign_dir / 'derived_summary.json'
        if not derived.is_file():
            raise FileNotFoundError(f'Missing required campaign index: {cs}')
        cs = derived
    if cs.exists():
        meta = json.loads(cs.read_text())
        for e in meta.get('experiments', []):
            rd = e.get('run_dir')
            if not rd:
                if e.get('status') == 'done':
                    raise FileNotFoundError(f'Missing run_dir for completed experiment: {e.get("id")}')
                continue
            sp = Path(rd) / 'summary.json'
            if not sp.is_file():
                raise FileNotFoundError(f'Missing experiment summary: {sp}')
            if sp.exists():
                try:
                    s = json.loads(sp.read_text())
                    s['xp_id'] = e.get('id')
                    s['xp_status'] = e.get('status')
                    exps.append(s)
                except (OSError, ValueError) as exc:
                    raise ValueError(f'Invalid experiment summary: {sp}') from exc
    if not exps:
        raise ValueError(f'No completed experiment summaries in {cs}')
    return exps


def _find(exps, xp_id=None):
    for s in exps:
        if s.get('xp_id') == xp_id:
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    default_campaign = ('artifacts/campaign' if Path('artifacts/campaign/campaign_summary.json').is_file()
                        else 'research/campaign')
    ap.add_argument('--campaign', default=default_campaign)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    cdir = Path(args.campaign)
    exps = _load_campaign(cdir)
    _validate_complete(exps)

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
    seeds.append((42, 'Final policy', REF['mean'], REF['std'], REF['x'], REF['fall'], REF['n']))
    if seed43_1m:
        seeds.append((43, 'Checkpoint', seed43_1m['return_mean'], seed43_1m['return_std'],
                      seed43_1m['x_progress_mean'], seed43_1m['fall_rate'],
                      seed43_1m.get('n', 10)))
    if seed44:
        seeds.append((44, 'Final policy', seed44['final_return_mean'], seed44['final_return_std'],
                      seed44['final_x_progress_mean'], seed44['final_fall_rate'],
                      seed44['eval_episodes']))

    L = []
    L.append('# Campaign analysis (auto-generated)\n')

    L.append('## Cross-seed, 1M steps, base hyperparameters (SAC net 256, lr 3e-4, batch 256)\n')
    L.append('| seed | evaluation | mean return | episode SD | x-progress (m) | fall rate | n |')
    L.append('|---|---|---|---|---|---|---|')
    for seed, evaluation, m, sd, x, f, n in seeds:
        L.append(f'| {seed} | {evaluation} | {m:.0f} | {sd:.0f} | {x:.0f} | {f:.0%} | {n} |')
    means = [s[2] for s in seeds]
    if len(means) >= 2:
        spread = max(means) - min(means)
        L.append(f'\nThe unweighted mean of the three policy means was **{st.mean(means):.0f}**, '
                 f'with a range of **{min(means):.0f}–{max(means):.0f}** '
                 f'(spread {spread:.0f}). The comparison mixes checkpoint and final policies '
                 f'and uses different evaluation episode counts.')
    L.append('')

    # --- Checkpoint returns ------------------------------------------------
    L.append('## Checkpoint returns (seed 43, every 250,000 steps)\n')
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
    L.append('## Single-seed baselines (seed 42, requested 1,000,000 steps)\n')
    L.append('| algorithm | actual steps | eval episodes | mean return | episode SD | x (m) | fall rate |')
    L.append('|---|---|---|---|---|---|---|')
    L.append(f"| SAC (reference) | {REF['steps']:,} | {REF['n']} | {REF['mean']:.0f} | "
             f"{REF['std']:.0f} | {REF['x']:.0f} | {REF['fall']:.0%} |")
    for xp_id, name in (('ppo_1m', 'PPO'), ('td3_1m', 'TD3')):
        s = _find(exps, xp_id=xp_id)
        if s and s.get('final_return_mean') is not None:
            L.append(f"| {name} | {s['steps']:,} | {s['eval_episodes']} | "
                     f"{s['final_return_mean']:.0f} | {s['final_return_std']:.0f} | "
                     f"{s['final_x_progress_mean']:.0f} | {s['final_fall_rate']:.0%} |")
        else:
            L.append(f"| {name} | (not run / skipped by deadline) | | | | | |")
    ppo = _find(exps, xp_id='ppo_1m')
    if ppo and ppo['steps'] != ppo['config']['steps']:
        L.append(f"\nPPO completed {ppo['steps']:,} steps because it collects "
                 f"{ppo['config']['ppo_n_steps']:,}-step rollouts toward the requested "
                 f"{ppo['config']['steps']:,}-step budget.")
    L.append('')

    md = '\n'.join(L)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(md, encoding='utf-8')
        print(f'wrote {args.out}')
    print(md)


if __name__ == '__main__':
    main()
