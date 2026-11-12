#!/usr/bin/env python3
"""Generate a single results slide (16:9) for the presentation deck."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

OUT = Path('artifacts/ant_sac_full_20261021T194131399885Z/hardening/results_slide.png')
BASE = OUT.parent

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

# 1. seeds
a1 = fig.add_subplot(gs[0, 0])
seeds = ['42', '43', '44']; vals = [5701, 3286, 2571]; errs = [157, 904, 1108]
b = a1.bar(seeds, vals, yerr=errs, capsize=5, color='#4f9cf7', alpha=0.9)
a1.set_title('Разброс по сидам (1М)', fontsize=13)
a1.set_ylabel('награда')
for x, v in zip(b, vals):
    a1.annotate(f'{v:.0f}', (x.get_x()+x.get_width()/2, v), textcoords='offset points',
                xytext=(0, 6), ha='center', color='#eef4ff', fontsize=9)
ax_style(a1)

# 2. convergence
a2 = fig.add_subplot(gs[0, 1])
xs = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
ys = [1068, 2324, 3348, 3286, 2011, 2974, 2315, 3394]
a2.plot(xs, ys, 'o-', color='#7fe0b5', lw=2)
a2.axvline(1.0, color='#f59e0b', ls='--', lw=1.5)
a2.text(1.02, 3400, '1М бюджет', color='#f59e0b', fontsize=9)
a2.set_title('Сходимость (seed 43)', fontsize=13)
a2.set_xlabel('шаги (М)'); a2.set_ylabel('награда')
ax_style(a2)

# 3. ablation
a3 = fig.add_subplot(gs[1, 0])
abl = ['net64', 'net128', 'lr1e-4', 'lr1e-3', 'bat128', 'bat512']
av = [657, 2074, 1803, 4503, 2389, 1707]
ae = [579, 743, 700, 1241, 1266, 1124]
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
bn = ['SAC', 'TD3', 'PPO']; bv = [5701, 3117, 1199]; be = [157, 415, 332]
bc = ['#7fe0b5', '#f59e0b', '#e06666']
b4 = a4.bar(bn, bv, yerr=be, capsize=5, color=bc, alpha=0.9)
a4.set_title('Сравнение алгоритмов (1М)', fontsize=13)
a4.set_ylabel('награда')
for x, v in zip(b4, bv):
    a4.annotate(f'{v:.0f}', (x.get_x()+x.get_width()/2, v), textcoords='offset points',
                xytext=(0, 6), ha='center', color='#eef4ff', fontsize=10)
ax_style(a4)

# bottom line
fig.text(0.05, 0.015,
         'Вывод: SAC лучший из 3 алгоритмов · 5701 = верх сид-диапазона 2571–5701 (ср. ≈3853) · 1М шагов — плато · '
         'помогают lr=1e-3 и сеть ≥128',
         color='#c3d6ee', fontsize=11)

fig.savefig(OUT)
print('slide written:', OUT, OUT.stat().st_size, 'bytes')
