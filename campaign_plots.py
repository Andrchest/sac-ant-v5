#!/usr/bin/env python3
"""Generate hardening charts from the campaign results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CAMPAIGN = Path('artifacts/campaign')
OUT = Path('artifacts/ant_sac_full_20261021T194131399885Z/hardening')
OUT.mkdir(parents=True, exist_ok=True)

# --- load campaign summary -------------------------------------------------
meta = json.loads((CAMPAIGN / 'campaign_summary.json').read_text())
by_id = {e['id']: e for e in meta['experiments']}

def get(xp_id):
    e = by_id[xp_id]
    if not e.get('run_dir'):
        return None
    return json.loads((Path(e['run_dir']) / 'summary.json').read_text())

REF = 5701.04  # seed42 1M reference

# --- 1. convergence (seed43) ----------------------------------------------
conv = get('conv_s43_2m')
if conv and conv.get('checkpoints'):
    cps = sorted(conv['checkpoints'], key=lambda c: c['step'])
    xs = [c['step'] / 1e6 for c in cps]
    ys = [c['return_mean'] for c in cps]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, 'o-', color='#2563eb', lw=2)
    ax.axvline(1.0, color='#dc2626', ls='--', lw=1.5, label='1M budget')
    ax.axhline(REF, color='#16a34a', ls=':', lw=1.5, label='best seed (42) @1M')
    for x, y in zip(xs, ys):
        ax.annotate(f'{y:.0f}', (x, y), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=8)
    ax.set_xlabel('Training steps (M)')
    ax.set_ylabel('Eval return (10 eps)')
    ax.set_title('SAC convergence on Ant-v5 (seed 43)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'convergence.png', dpi=130)
    plt.close(fig)

# --- 2. ablation (500k) ----------------------------------------------------
abl = [
    ('net64', get('net64')), ('net128', get('net128')),
    ('lr 1e-4', get('lr1e4')), ('lr 1e-3', get('lr1e3')),
    ('batch 128', get('batch128')), ('batch 512', get('batch512')),
]
labels, vals, errs = [], [], []
for name, s in abl:
    if s and s.get('final_return_mean') is not None:
        labels.append(name)
        vals.append(s['final_return_mean'])
        errs.append(s['final_return_std'])
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = ['#dc2626' if v < 1500 else ('#16a34a' if v > 3500 else '#2563eb') for v in vals]
bars = ax.bar(labels, vals, yerr=errs, capsize=5, color=colors, alpha=0.85)
ax.set_ylabel('Eval return (10 eps, 500k steps)')
ax.set_title('Hyperparameter ablation (SAC, seed 42, 500k steps)')
ax.grid(axis='y', alpha=0.3)
for b, v in zip(bars, vals):
    ax.annotate(f'{v:.0f}', (b.get_x() + b.get_width() / 2, v),
                textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9)
fig.tight_layout()
fig.savefig(OUT / 'ablation.png', dpi=130)
plt.close(fig)

# --- 3. baselines (1M) -----------------------------------------------------
baselines = [('SAC (seed42)', REF, 157.17)]
ppo = get('ppo_1m')
td3 = get('td3_1m')
if ppo and ppo.get('final_return_mean') is not None:
    baselines.append(('PPO', ppo['final_return_mean'], ppo['final_return_std']))
if td3 and td3.get('final_return_mean') is not None:
    baselines.append(('TD3', td3['final_return_mean'], td3['final_return_std']))
fig, ax = plt.subplots(figsize=(6.5, 4.5))
names = [b[0] for b in baselines]
bv = [b[1] for b in baselines]
be = [b[2] for b in baselines]
bc = ['#16a34a' if 'SAC' in n else '#f59e0b' for n in names]
bars = ax.bar(names, bv, yerr=be, capsize=5, color=bc, alpha=0.85)
ax.set_ylabel('Eval return (1M steps, seed 42)')
ax.set_title('Algorithm comparison on Ant-v5 (1M steps)')
ax.grid(axis='y', alpha=0.3)
for b, v in zip(bars, bv):
    ax.annotate(f'{v:.0f}', (b.get_x() + b.get_width() / 2, v),
                textcoords='offset points', xytext=(0, 8), ha='center', fontsize=10)
fig.tight_layout()
fig.savefig(OUT / 'baselines.png', dpi=130)
plt.close(fig)

# --- 4. seeds (1M) ---------------------------------------------------------
seed43 = None
if conv and conv.get('checkpoints'):
    for c in conv['checkpoints']:
        if c['step'] == 1_000_000:
            seed43 = c
seed44 = get('seed44_1m')
seeds = [('seed 42', 5701.04, 157.17)]
if seed43:
    seeds.append(('seed 43', seed43['return_mean'], seed43['return_std']))
if seed44:
    seeds.append(('seed 44', seed44['final_return_mean'], seed44['final_return_std']))
fig, ax = plt.subplots(figsize=(6.5, 4.5))
sn = [s[0] for s in seeds]
sv = [s[1] for s in seeds]
se = [s[2] for s in seeds]
bars = ax.bar(sn, sv, yerr=se, capsize=5, color='#2563eb', alpha=0.85)
ax.set_ylabel('Eval return (1M steps, base hyperparams)')
ax.set_title('SAC inter-seed variance (Ant-v5, 1M steps)')
ax.grid(axis='y', alpha=0.3)
for b, v in zip(bars, sv):
    ax.annotate(f'{v:.0f}', (b.get_x() + b.get_width() / 2, v),
                textcoords='offset points', xytext=(0, 8), ha='center', fontsize=10)
fig.tight_layout()
fig.savefig(OUT / 'seeds.png', dpi=130)
plt.close(fig)

print('charts written to', OUT)
for f in sorted(OUT.glob('*.png')):
    print(' ', f.name, f.stat().st_size, 'bytes')
