#!/usr/bin/env python3
"""Professional statistics on FINAL 30-episode evals (evaluation_all.csv).
Honest caveat: base seeds (42/44) != best seeds (50/51/52) -> NOT a controlled
same-seed comparison. Reported as such."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from math import erf, sqrt

ART = Path('artifacts')
RUN = ART / 'ant_sac_full_20261021T194131399885Z'
rng = np.random.default_rng(0)

def final_returns(csv):
    df = pd.read_csv(csv)
    df = df[df['policy'].str.endswith('_final')]
    return df['return_'].to_numpy(dtype=float), df

def has_csv(pattern):
    ds = [d for d in ART.glob(pattern) if (d / 'evaluation_all.csv').is_file()]
    return sorted(ds)[0]

base42, _ = final_returns(RUN / 'evaluation_all.csv')
base44, _ = final_returns(has_csv('xp_sac_44_*seed*') / 'evaluation_all.csv')
best50, _ = final_returns(has_csv('xp_sac_50_*best*') / 'evaluation_all.csv')
best51, _ = final_returns(has_csv('xp_sac_51_*best*') / 'evaluation_all.csv')
best52, _ = final_returns(has_csv('xp_sac_52_*best*') / 'evaluation_all.csv')

base = np.concatenate([base42, base44])      # 60 eps, seeds 42/44
best = np.concatenate([best50, best51, best52])  # 90 eps, seeds 50/51/52

def boot_ci(x, n=20000, ci=95):
    m = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(m, (100-ci)/2)), float(np.percentile(m, 100-(100-ci)/2))

def mwu(a, b):
    a = np.asarray(a); b = np.asarray(b)
    comb = np.concatenate([a, b])
    ranks = np.argsort(np.argsort(comb)) + 1
    ra, rb = ranks[:len(a)], ranks[len(a):]
    n1, n2 = len(a), len(b)
    u1 = ra.sum() - n1*(n1+1)/2
    u = min(u1, n1*n2 - u1)
    mu = n1*n2/2
    _, counts = np.unique(comb, return_counts=True)
    tie = (counts**3 - counts).sum()
    sigma = np.sqrt(n1*n2*(n1+n2+1)/12 - n1*n2*tie/(12*(n1+n2)))
    z = (u - mu + 0.5)/sigma
    p = 2*(1 - 0.5*(1+erf(abs(z)/sqrt(2))))
    return float(u), float(z), float(p)

print('Final 30-episode evals (per seed):')
for name, arr in [('base42', base42), ('base44', base44),
                  ('best50', best50), ('best51', best51), ('best52', best52)]:
    print(f'  {name:8} n={len(arr):2d}  mean={arr.mean():7.1f}  sd={arr.std(ddof=1):7.1f}')

print('\nPooled FINAL (base 60 eps [42/44]  vs  best 90 eps [50/51/52]):')
bm, blo, bhi = boot_ci(base)
sm, slo, shi = boot_ci(best)
print(f'  base  mean={bm:7.1f}  95% CI [{blo:.1f}, {bhi:.1f}]')
print(f'  best  mean={sm:7.1f}  95% CI [{slo:.1f}, {shi:.1f}]')
u, z, p = mwu(base, best)
sp = np.sqrt((base.std(ddof=1)**2 + best.std(ddof=1)**2)/2)
d = (sm - bm)/sp
print(f'\nMann-Whitney U={u:.0f}, z={z:.2f}, p={p:.4f}')
print(f'Mean diff best-base = {sm-bm:+.1f}, Cohen d = {d:.2f}')
print('\nCAVEAT: base seeds (42/44) != best seeds (50/51/52); seed43@1M has no per-episode')
print('final data. This is NOT a controlled same-seed comparison. A same-seed head-to-head')
print('(base vs best config on seeds 50/51/52) is the remaining gap for a airtight claim.')

json.dump({
    'base_mean': bm, 'base_ci95': [blo, bhi], 'best_mean': sm, 'best_ci95': [slo, shi],
    'mwu': {'U': u, 'z': z, 'p': p}, 'cohens_d': float(d),
    'per_seed': {k: float(v.mean()) for k, v in
                 [('base42',base42),('base44',base44),('best50',best50),
                  ('best51',best51),('best52',best52)]},
    'caveat': 'base seeds (42/44) != best seeds (50/51/52); not a controlled same-seed comparison',
}, open('artifacts/campaign/best_vs_base_stats.json', 'w'), indent=2)
print('\nwritten artifacts/campaign/best_vs_base_stats.json')
