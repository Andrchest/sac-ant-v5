#!/usr/bin/env python3
"""Descriptive per-training-seed results, not an independent-episode test.

The older pooled episode bootstrap and p-value were withdrawn: episodes within a
training run are not independent training seeds, and evaluation sizes differ.
"""
import json
from pathlib import Path

ART = Path('artifacts')
CAMPAIGN = ART / 'campaign'


def main():
    index = CAMPAIGN / 'campaign_summary.json'
    if not index.is_file():
        index = Path('research/campaign/derived_summary.json')
    meta = json.loads(index.read_text(encoding='utf-8'))
    by_id = {e['id']: e for e in meta['experiments']}
    run = Path(by_id['conv_s43_2m']['run_dir'])
    s43 = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    cp = next(c for c in s43['checkpoints'] if c['step'] == 1_000_000)
    base = [meta['reference']['final_return_mean'], cp['return_mean'],
            by_id['seed44_1m']['final_return_mean']]
    bundled = index.parent / 'best'
    best = []
    for seed in (50, 51, 52):
        paths = ([bundled / f'seed_{seed}' / 'summary.json'] if bundled.is_dir() else
                 sorted(ART.glob(f'xp_sac_{seed}_*best{seed}/summary.json')))
        if len(paths) != 1 or not paths[0].is_file():
            raise FileNotFoundError(f'Expected one summary for seed {seed}: {paths}')
        best.append(json.loads(paths[0].read_text(encoding='utf-8'))['final_return_mean'])
    for name, values in [('base 42/43/44', base), ('selected 50/51/52', best)]:
        print(f'{name}: means={values}; mean={sum(values)/len(values):.1f}; '
              f'range={min(values):.1f}–{max(values):.1f}')
    print('Descriptive only: 3 training seeds per configuration, different seeds and evaluation protocols.')


if __name__ == '__main__':
    main()
