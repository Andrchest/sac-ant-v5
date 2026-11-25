#!/usr/bin/env python3
"""Record a clearer selected-policy rollout; no training occurs.

Ant's MuJoCo plane is physically infinite, but its default visible extent is
small relative to this policy's progress. Enlarge the plane's *rendered* grid;
verify the resulting episode against the saved evaluation before publishing.
"""
import argparse
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import SAC, TD3, PPO
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--algo', choices=('sac', 'td3', 'ppo'), default='sac')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--seed', type=int, default=100050)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--label', default='SAC seed 50 | final 1M steps')
    args = p.parse_args()
    torch.set_num_threads(2)
    cls = {'sac': SAC, 'td3': TD3, 'ppo': PPO}[args.algo]
    model = cls.load(args.checkpoint, device='cpu', **({'buffer_size': 1} if args.algo != 'ppo' else {}))
    initial_timesteps = model.num_timesteps
    env = gym.make('Ant-v5', max_episode_steps=1000, render_mode='rgb_array',
                   include_cfrc_ext_in_observation=True, exclude_current_positions_from_observation=True,
                   healthy_reward=1.0, forward_reward_weight=1.0, ctrl_cost_weight=0.5,
                   contact_cost_weight=0.0005, healthy_z_range=(0.2, 1.0),
                   terminate_when_unhealthy=True)
    floor = mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    if floor < 0 or env.unwrapped.model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE:
        raise RuntimeError('Expected MuJoCo plane named floor')
    # For a MuJoCo plane the first two geom_size values control visualization;
    # its collision plane is infinite. Compare the outcome to the saved CSV too.
    env.unwrapped.model.geom_size[floor, :2] = 5000
    obs, _ = env.reset(seed=args.seed)
    x0 = float(env.unwrapped.data.qpos[0])
    reward_sum = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames = 0
    font = ImageFont.load_default(size=15)
    terminated = truncated = False
    try:
        with imageio.get_writer(args.out, fps=env.metadata['render_fps'], macro_block_size=1,
                                codec='libx264', quality=4) as writer:
            for _ in range(args.steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                reward_sum += float(reward)
                progress = float(env.unwrapped.data.qpos[0]) - x0
                image = Image.fromarray(env.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((6, 6, 450, 52), fill=(0, 0, 0))
                draw.text((12, 8), args.label, fill='white', font=font)
                draw.text((12, 28), f'Forward progress: {progress:.1f} m | Step: {frames + 1}', fill='white', font=font)
                writer.append_data(np.asarray(image))
                frames += 1
                if terminated or truncated:
                    break
    finally:
        env.close()
    if model.num_timesteps != initial_timesteps:
        raise RuntimeError('Inference unexpectedly changed training timestep count')
    print(f'frames={frames} return={reward_sum:.12f} x_progress={progress:.12f} '
          f'terminated={terminated} truncated={truncated} train_steps={initial_timesteps}', flush=True)


if __name__ == '__main__':
    main()
