#!/usr/bin/env python3
"""Charts comparing best-config SAC (lr1e-3/batch128) vs base config."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('artifacts/ant_sac_full_20261021T194131399885Z/hardening')
OUT.mkdir(parents=True, exist_ok=True)

best = []
bundle_best = Path('artifacts/campaign/best')
directories = sorted(bundle_best.glob('seed_*')) if bundle_best.is_dir() else sorted(Path('artifacts').glob('xp_sac_5[012]_*best*'))
if not directories:
    directories = sorted(Path('research/campaign/best').glob('seed_*'))
for d in directories:
    sp = d / 'summary.json'
    if sp.exists():
        s = json.loads(sp.read_text())
        best.append((s['seed'], s['final_return_mean'], s['final_return_std'],
                     s['final_x_progress_mean'], s['final_fall_rate']))
best.sort()
if len(best) != 3 or [row[0] for row in best] != [50, 51, 52]:
    raise ValueError('Expected complete best-config summaries for seeds 50, 51 and 52')
base = [(42, 5701.04, 157.17, 286.39, 0.0),
        (43, 3286.0, 904.0, 171.0, 0.5),
        (44, 2571.0, 1108.0, 133.0, 0.43)]

# --- comparison: best vs base seeds ---------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.8))
x = range(len(base))
bw = 0.38
b_vals = [b[1] for b in base]
b_err = [b[2] for b in base]
s_vals = [b[1] for b in best]
s_err = [b[2] for b in best]
ax.bar([i - bw/2 for i in x], b_vals, yerr=b_err, capsize=4, width=bw,
       color='#8aa0bd', alpha=0.85, label='base config (lr 3e-4, batch 256)')
ax.bar([i + bw/2 for i in x], s_vals, yerr=s_err, capsize=4, width=bw,
       color='#4f9cf7', alpha=0.9, label='best config (lr 1e-3, batch 128)')
ax.set_xticks(list(x))
ax.set_xticklabels([f'run {i+1}\nbase seed {base[i][0]}\nbest seed {best[i][0]}' for i in x],
                   fontsize=8)
ax.set_ylabel('Eval return (1M steps)')
ax.set_title('SAC 1M: base n=10/10/30; best n=30 per seed')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
for i in x:
    ax.annotate(f'{b_vals[i]:.0f}', (i - bw/2, b_vals[i]), textcoords='offset points',
                xytext=(0, 6), ha='center', fontsize=8)
    ax.annotate(f'{s_vals[i]:.0f}', (i + bw/2, s_vals[i]), textcoords='offset points',
                xytext=(0, 6), ha='center', fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'best_vs_base.png', dpi=130)
plt.close(fig)

# --- best config seeds only ----------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
names = [f'seed {b[0]}' for b in best]
bv = [b[1] for b in best]
be = [b[2] for b in best]
bars = ax.bar(names, bv, yerr=be, capsize=5, color='#4f9cf7', alpha=0.9)
ax.axhline(3853, color='#8aa0bd', ls='--', lw=1.5, label='base-config mean (3853)')
ax.set_ylabel('Eval return (1M steps)')
ax.set_title('Best config: SAC 256 / lr 1e-3 / batch 128 (3 seeds)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for b, v in zip(bars, bv):
    ax.annotate(f'{v:.0f}', (b.get_x()+b.get_width()/2, v), textcoords='offset points',
                xytext=(0, 6), ha='center', fontsize=10)
fig.tight_layout()
fig.savefig(OUT / 'best_config.png', dpi=130)
plt.close(fig)

print('charts written')
for f in sorted(OUT.glob('best*.png')):
    print(' ', f.name, f.stat().st_size, 'bytes')
