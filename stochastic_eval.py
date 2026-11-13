#!/usr/bin/env python3
"""Stochastic (temperature) evaluation of a SAC checkpoint: 30 episodes,
deterministic=False. Shows robustness beyond the deterministic policy."""
from pathlib import Path
import argparse
import json
import gymnasium as gym
from stable_baselines3 import SAC


def make_env():
    return gym.make('Ant-v5', max_episode_steps=1000,
        include_cfrc_ext_in_observation=True, exclude_current_positions_from_observation=True,
        healthy_reward=1.0, forward_reward_weight=1.0, ctrl_cost_weight=0.5,
        contact_cost_weight=0.0005, healthy_z_range=(0.2, 1.0), terminate_when_unhealthy=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--episodes', type=int, default=30)
    p.add_argument('--base-seed', type=int, default=950000)
    p.add_argument('--out', default='')
    a = p.parse_args()

    model = SAC.load(Path(a.checkpoint), device='cpu', buffer_size=1)
    env = make_env()
    rewards, xs, falls, lens = [], [], 0, []
    for i in range(a.episodes):
        obs, _ = env.reset(seed=a.base_seed + i)
        total, x, steps = 0.0, 0.0, 0
        for step in range(1000):
            action, _ = model.predict(obs, deterministic=False)
            obs, r, terminated, truncated, info = env.step(action)
            total += float(r)
            x = info['x_position']
            steps = step + 1
            if terminated or truncated:
                break
        rewards.append(total); xs.append(x); lens.append(steps)
        falls += int(terminated)
    env.close()
    import numpy as np
    r = np.array(rewards); x = np.array(xs)
    res = dict(episodes=a.episodes, stochastic=True,
        return_mean=float(r.mean()), return_std=float(r.std(ddof=1)),
        x_mean=float(x.mean()), fall_rate=falls / a.episodes,
        min_return=float(r.min()), max_return=float(r.max()))
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
        print('written', a.out)


if __name__ == '__main__':
    main()
