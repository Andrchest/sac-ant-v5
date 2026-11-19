#!/usr/bin/env python3
"""Generate a single results slide (16:9) for the presentation deck."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

OUT = Path('artifacts/ant_sac_full_20261021T194131399885Z/hardening/results_slide.png')
BASE = OUT.parent
DATA = Path('research/campaign')
if DATA.is_dir():
    index_path = DATA / 'derived_summary.json'
else:
    DATA = Path('artifacts/campaign')
    index_path = DATA / 'campaign_summary.json'
index = json.loads(index_path.read_text(encoding='utf-8'))
rows = {e['id']: e for e in index['experiments']}
def summary(xp_id):
    return json.loads((Path(rows[xp_id]['run_dir']) / 'summary.json').read_text(encoding='utf-8'))
conv = summary('conv_s43_2m')
checkpoints = {c['step']: c for c in conv['checkpoints']}
best = [json.loads((DATA / 'best' / f'seed_{seed}' / 'summary.json').read_text(encoding='utf-8'))
        for seed in (50, 51, 52)]
ref = index['reference']
seed44 = summary('seed44_1m')
base = [(ref['final_return_mean'], ref['final_return_std']),
        (checkpoints[1_000_000]['return_mean'], checkpoints[1_000_000]['return_std']),
        (seed44['final_return_mean'], seed44['final_return_std'])]

fig = plt.figure(figsize=(13.333, 7.5), dpi=150)
fig.patch.set_facecolor('#0d1b2a')
gs = gridspec.GridSpec(2, 2, left=0.05, right=0.97, top=0.80, bottom=0.06,
                       wspace=0.28, hspace=0.42)

# title
fig.text(0.05, 0.93, 'SAC × Ant-v5 — результаты (hardening)',
         color='#eef4ff', fontsize=26, fontweight='bold')
fig.text(0.05, 0.875, 'Мульти-сид, сходимость, абляции, бейзлайны · 1М шагов, CPU',
         color='#8aa0bd', fontsize=13)

def ax_style(a):
    a.set_facecolor('#12233a')
    a.tick_params(colors='#c3d6ee')
    a.title.set_color('#eef4ff')
    a.xaxis.label.set_color('#c3d6ee'); a.yaxis.label.set_color('#c3d6ee')
    for s in a.spines.values(): s.set_color('#2c4363')
    a.grid(color='#22344e', alpha=0.6)

# 1. best vs base
a1 = fig.add_subplot(gs[0, 0])
x = [0, 1, 2]; bw = 0.38
base_v = [row[0] for row in base]; base_e = [row[1] for row in base]
best_v = [row['final_return_mean'] for row in best]; best_e = [row['final_return_std'] for row in best]
a1.bar([i-bw/2 for i in x], base_v, yerr=base_e, capsize=4, width=bw, color='#8aa0bd', alpha=0.85, label='base (lr 3e-4)')
a1.bar([i+bw/2 for i in x], best_v, yerr=best_e, capsize=4, width=bw, color='#4f9cf7', alpha=0.9, label='best (lr 1e-3)')
a1.set_xticks(x); a1.set_xticklabels(['run 1', 'run 2', 'run 3'], fontsize=9)
a1.set_title('Best config vs base (1М)', fontsize=13)
a1.set_ylabel('награда')
a1.legend(fontsize=8)
for i in x:
    a1.annotate(f'{base_v[i]:.0f}', (i-bw/2, base_v[i]), textcoords='offset points', xytext=(0, 5), ha='center', color='#eef4ff', fontsize=8)
    a1.annotate(f'{best_v[i]:.0f}', (i+bw/2, best_v[i]), textcoords='offset points', xytext=(0, 5), ha='center', color='#eef4ff', fontsize=8)
ax_style(a1)

# 2. convergence
a2 = fig.add_subplot(gs[0, 1])
xs = [step / 1e6 for step in sorted(checkpoints)]
ys = [checkpoints[step]['return_mean'] for step in sorted(checkpoints)]
a2.plot(xs, ys, 'o-', color='#7fe0b5', lw=2)
a2.axvline(1.0, color='#f59e0b', ls='--', lw=1.5)
a2.text(1.02, 3400, '1М бюджет', color='#f59e0b', fontsize=9)
a2.set_title('Сходимость (seed 43)', fontsize=13)
a2.set_xlabel('шаги (М)'); a2.set_ylabel('награда')
ax_style(a2)

# 3. ablation
a3 = fig.add_subplot(gs[1, 0])
abl = ['net64', 'net128', 'lr1e-4', 'lr1e-3', 'bat128', 'bat512']
ablation = [summary(xp_id) for xp_id in ('net64', 'net128', 'lr1e4', 'lr1e3', 'batch128', 'batch512')]
av = [row['final_return_mean'] for row in ablation]
ae = [row['final_return_std'] for row in ablation]
col = ['#e06666' if v < 1500 else ('#7fe0b5' if v > 3500 else '#4f9cf7') for v in av]
b3 = a3.bar(abl, av, yerr=ae, capsize=4, color=col, alpha=0.9)
a3.set_title('Абляция (500k, по одному фактору)', fontsize=13)
a3.set_ylabel('награда')
a3.tick_params(axis='x', labelsize=8, rotation=20)
for x, v in zip(b3, av):
    a3.annotate(f'{v:.0f}', (x.get_x()+x.get_width()/2, v), textcoords='offset points',
                xytext=(0, 5), ha='center', color='#eef4ff', fontsize=8)
ax_style(a3)

# 4. baselines
a4 = fig.add_subplot(gs[1, 1])
baselines = [summary('td3_1m'), summary('ppo_1m')]
bn = ['SAC', 'TD3', 'PPO']; bv = [base_v[0]] + [row['final_return_mean'] for row in baselines]
be = [base_e[0]] + [row['final_return_std'] for row in baselines]
bc = ['#7fe0b5', '#f59e0b', '#e06666']
b4 = a4.bar(bn, bv, yerr=be, capsize=5, color=bc, alpha=0.9)
a4.set_title('Сравнение алгоритмов (1М)', fontsize=13)
a4.set_ylabel('награда')
for x, v in zip(b4, bv):
    a4.annotate(f'{v:.0f}', (x.get_x()+x.get_width()/2, v), textcoords='offset points',
                xytext=(0, 6), ha='center', color='#eef4ff', fontsize=10)
ax_style(a4)

# bottom line
fig.text(0.05, 0.018,
         'Три SAC seed на конфигурацию; разные протоколы. PPO/TD3: по одному seed, TD3 без action noise.\n'
         'Наблюдения не доказывают превосходство алгоритма или конфигурации.',
         color='#c3d6ee', fontsize=9)

# Ensure the slide-level titles and footnote stay inside the canvas.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for text in fig.texts:
    bbox = text.get_window_extent(renderer=renderer)
    if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > fig.bbox.width or bbox.y1 > fig.bbox.height:
        raise ValueError(f'Slide text exceeds canvas: {text.get_text()!r}')
fig.savefig(OUT)
print('slide written:', OUT, OUT.stat().st_size, 'bytes')
