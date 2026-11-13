#!/usr/bin/env python3
"""Record a deterministic rollout MP4 for a given SAC checkpoint (inference only)."""
from pathlib import Path
import argparse
import gymnasium as gym
import imageio
from stable_baselines3 import SAC


def make_env():
    return gym.make('Ant-v5', max_episode_steps=1000, render_mode='rgb_array',
        include_cfrc_ext_in_observation=True, exclude_current_positions_from_observation=True,
        healthy_reward=1.0, forward_reward_weight=1.0, ctrl_cost_weight=0.5,
        contact_cost_weight=0.0005, healthy_z_range=(0.2, 1.0), terminate_when_unhealthy=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=900050)
    p.add_argument('--steps', type=int, default=1000)
    a = p.parse_args()

    model = SAC.load(Path(a.checkpoint), device='cpu', buffer_size=1)
    n0 = model.num_timesteps
    env = make_env()
    frames = []
    obs, _ = env.reset(seed=a.seed)
    total = 0.0
    for _ in range(a.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, _ = env.step(action)
        total += float(r)
        frames.append(env.render())
        if terminated or truncated:
            break
    env.close()
    assert model.num_timesteps == n0, 'no training performed'
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=24, macro_block_size=1)
    print(f'wrote {out} ({len(frames)} frames), reward={total:.2f}')


if __name__ == '__main__':
    main()
