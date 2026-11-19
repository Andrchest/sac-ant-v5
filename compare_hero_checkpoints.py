#!/usr/bin/env python3
"""Evaluate the two saved hero policies on identical held-out reset seeds.

Inference only; each checkpoint is loaded with a one-slot replay buffer.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from stable_baselines3 import SAC

from run_experiment import evaluate_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--final', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--seed', type=int, default=920000)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    torch.set_num_threads(2)
    seeds = list(range(args.seed, args.seed + args.episodes))
    args.output.mkdir(parents=True, exist_ok=True)
    frames, summary = [], {}
    for name, path in [('checkpoint', args.checkpoint), ('final', args.final)]:
        model = SAC.load(path, device='cpu', buffer_size=1)
        steps, updates = model.num_timesteps, model._n_updates
        frame = evaluate_agent(model, seeds, name)
        assert model.num_timesteps == steps and model._n_updates == updates
        frames.append(frame)
        summary[name] = {'training_steps': int(steps), 'gradient_updates': int(updates),
                         'episodes': len(frame), 'return_mean': float(frame.return_.mean()),
                         'return_std': float(frame.return_.std(ddof=1)) if len(frame) > 1 else 0.0,
                         'falls': int(frame.terminated.sum()),
                         'x_progress_mean': float(frame.x_progress.mean())}
    summary['protocol'] = {'deterministic': True, 'reset_seeds': seeds,
                           'note': 'Paired held-out resets; one training seed, not independent training runs.'}
    pd.concat(frames, ignore_index=True).to_csv(args.output / 'hero_checkpoint_comparison.csv', index=False)
    (args.output / 'hero_checkpoint_comparison.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
