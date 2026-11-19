"""Run a single, reproducible Ant-v5 experiment (SAC / PPO / TD3).

Self-contained and parallel-safe: each invocation writes to its own timestamped
directory and prints a single JSON summary line on stdout. The environment, reward
shaping and evaluation match ``sac_ant_v5.ipynb`` exactly, so results are comparable
to the original full run.

Example:
    python run_experiment.py --algo sac --seed 43 --steps 2000000 \
        --net 256,256 --lr 3e-4 --batch 256 --threads 4 \
        --checkpoint-every 250000 --eval-episodes 30 --tag conv
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
import json
import random
import sys
import time
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList

ENV_ID = 'Ant-v5'
MAX_STEPS = 1000
OBS_DIM = 105
ACT_DIM = 8


def parse_net(text):
    return [int(x) for x in text.split(',') if x.strip()]


def make_env(render_mode=None):
    return gym.make(
        ENV_ID, render_mode=render_mode, max_episode_steps=MAX_STEPS,
        include_cfrc_ext_in_observation=True, exclude_current_positions_from_observation=True,
        healthy_reward=1.0, forward_reward_weight=1.0, ctrl_cost_weight=0.5,
        contact_cost_weight=0.0005, healthy_z_range=(0.2, 1.0), terminate_when_unhealthy=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_agent(model, episode_seeds, policy_name):
    env = make_env()
    rows = []
    try:
        for episode, episode_seed in enumerate(episode_seeds, 1):
            obs, _ = env.reset(seed=int(episode_seed))
            env.action_space.seed(int(episode_seed) + 1_000_000)
            x0 = float(env.unwrapped.data.qpos[0])
            total_reward = 0.0
            components = {}
            length = MAX_STEPS
            for length in range(1, MAX_STEPS + 1):
                action = (env.action_space.sample() if model is None
                          else model.predict(obs, deterministic=True)[0])
                assert env.action_space.contains(np.asarray(action, dtype=np.float32))
                obs, reward, terminated, truncated, info = env.step(action)
                assert np.isfinite(obs).all() and np.isfinite(reward)
                total_reward += float(reward)
                for key in ('reward_forward', 'reward_survive', 'reward_ctrl', 'reward_contact'):
                    if key in info:
                        components[key] = components.get(key, 0.0) + float(info[key])
                if terminated or truncated:
                    break
            rows.append(dict(policy=policy_name, episode=episode, episode_seed=int(episode_seed),
                return_=total_reward, length=length,
                x_progress=float(env.unwrapped.data.qpos[0]) - x0,
                terminated=bool(terminated), truncated=bool(truncated), **components))
    finally:
        env.close()
    return pd.DataFrame(rows)


class StepCallback(BaseCallback):
    """Record losses, optionally save checkpoints and stop on an episode budget."""
    def __init__(self, checkpoint_every=None, checkpoint_dir=None, loss_every=100):
        super().__init__()
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir
        self.loss_every = loss_every
        self.losses = []
        self.episodes = []
        self.saved = []

    def _on_step(self):
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done:
                ep = info['episode']
                self.episodes.append(dict(episode=len(self.episodes) + 1,
                    timestep=int(self.num_timesteps), return_=float(ep['r']), length=int(ep['l'])))
        if self.num_timesteps % self.loss_every == 0:
            values = self.model.logger.name_to_value
            self.losses.append(dict(timestep=int(self.num_timesteps),
                **{k: float(v) for k, v in values.items() if k in
                   ('train/actor_loss', 'train/critic_loss', 'train/ent_coef', 'train/ent_coef_loss',
                    'loss/n_actor_loss', 'loss/n_critic_loss')}))
        if self.checkpoint_every and self.checkpoint_dir:
            step = int(self.num_timesteps)
            if step > 0 and step % self.checkpoint_every == 0:
                path = Path(self.checkpoint_dir) / f'step_{step:09d}'
                path.mkdir(parents=True, exist_ok=True)
                self.model.save(path / 'model.zip')
                self.saved.append(step)
        return True


def build_policy(algo, env, a):
    common = dict(seed=a.seed, device='cpu', verbose=0,
                  learning_rate=a.lr, gamma=a.gamma)
    net = dict(net_arch=parse_net(a.net))
    if algo == 'sac':
        return SAC('MlpPolicy', env, buffer_size=min(1_000_000, a.steps),
                   learning_starts=a.learning_starts, batch_size=a.batch, tau=a.tau,
                   ent_coef=a.ent_coef, train_freq=1, gradient_steps=1,
                   policy_kwargs=net, **common)
    if algo == 'td3':
        noise_std = getattr(a, 'action_noise_std', 0.0)
        noise = (NormalActionNoise(mean=np.zeros(env.action_space.shape),
                                   sigma=np.full(env.action_space.shape, noise_std))
                 if noise_std > 0 else None)
        return TD3('MlpPolicy', env, buffer_size=min(1_000_000, a.steps),
                   learning_starts=a.learning_starts, batch_size=a.batch, tau=a.tau,
                   action_noise=noise, policy_kwargs=net, **common)
    if algo == 'ppo':
        return PPO('MlpPolicy', env, batch_size=a.batch, n_steps=a.ppo_n_steps,
                   ent_coef=0.0, policy_kwargs=net, **common)
    raise SystemExit(f'unknown algo {algo}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--algo', choices=['sac', 'ppo', 'td3'], default='sac')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--net', default='256,256')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--tau', type=float, default=0.005)
    p.add_argument('--ent-coef', default='auto',
                   help="SAC entropy coefficient: 'auto' or a float, e.g. 0.2")
    p.add_argument('--learning-starts', type=int, default=10_000)
    p.add_argument('--action-noise-std', type=float, default=0.0,
                   help='TD3 normal exploration noise standard deviation (0 disables)')
    p.add_argument('--ppo-n-steps', type=int, default=2048)
    p.add_argument('--eval-episodes', type=int, default=10)
    p.add_argument('--checkpoint-every', type=int, default=0,
                   help='save + evaluate a checkpoint every N steps (0 = off)')
    p.add_argument('--checkpoint-eval-episodes', type=int, default=10)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--output', default='artifacts')
    p.add_argument('--tag', default='')
    p.add_argument('--save-replay', action='store_true',
                   help='also save the (large) replay buffer')
    args = p.parse_args()

    if not np.isfinite(args.action_noise_std) or args.action_noise_std < 0:
        p.error('--action-noise-std must be finite and non-negative')
    if args.algo != 'td3' and args.action_noise_std != 0:
        p.error('--action-noise-std applies only to TD3')
    torch.set_num_threads(args.threads)
    assert args.steps > args.learning_starts, 'steps must exceed learning_starts'
    ent_coef = 'auto' if args.ent_coef == 'auto' else float(args.ent_coef)

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    tag = f'_{args.tag}' if args.tag else ''
    run_dir = Path(args.output) / f'xp_{args.algo}_{args.seed}_{run_id}{tag}'
    run_dir.mkdir(parents=True, exist_ok=False)
    seed_dir = run_dir / f'seed_{args.seed}'
    seed_dir.mkdir(exist_ok=True)
    checkpoint_dir = seed_dir / 'checkpoints'

    versions = {pk: version(pk) for pk in ['gymnasium', 'mujoco', 'stable-baselines3',
                                           'torch', 'numpy', 'pandas']}
    config = dict(algo=args.algo, seed=args.seed, steps=args.steps,
                  net=parse_net(args.net), lr=args.lr, batch=args.batch, gamma=args.gamma,
                  tau=args.tau, ent_coef=ent_coef, learning_starts=args.learning_starts,
                  action_noise_std=args.action_noise_std if args.algo == 'td3' else None,
                  ppo_n_steps=args.ppo_n_steps, eval_episodes=args.eval_episodes,
                  checkpoint_every=args.checkpoint_every, threads=args.threads,
                  versions=versions, python=sys.version, executable=sys.executable,
                  platform=platform.platform(), run_id=run_id)
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2))

    seed_everything(args.seed)
    eval_seeds = [args.seed + 100_000 + i for i in range(args.eval_episodes)]
    random_eval = evaluate_agent(None, eval_seeds[:10], 'random')

    train_env = Monitor(make_env(), filename=str(seed_dir / 'train.monitor.csv'))
    train_env.reset(seed=args.seed)
    train_env.action_space.seed(args.seed)
    periodic_env = Monitor(make_env())
    periodic_env.reset(seed=args.seed + 200_000)

    model = build_policy(args.algo, train_env, args)
    initial_steps = int(model.num_timesteps)
    initial = evaluate_agent(model, eval_seeds, f'{args.algo}_initial')

    step_cb = StepCallback(checkpoint_every=args.checkpoint_every or None,
                           checkpoint_dir=checkpoint_dir if args.checkpoint_every else None)
    eval_freq = max(50_000, args.steps // 20)
    periodic = EvalCallback(eval_env=periodic_env,
                            best_model_save_path=str(seed_dir / 'best'),
                            log_path=str(seed_dir / 'eval'), eval_freq=eval_freq,
                            n_eval_episodes=min(5, args.eval_episodes),
                            deterministic=True, verbose=0)
    start = time.perf_counter()
    model.learn(total_timesteps=args.steps, reset_num_timesteps=True,
                callback=CallbackList([periodic, step_cb]), log_interval=1)
    train_seconds = time.perf_counter() - start
    model.save(seed_dir / f'{args.algo}_final.zip')
    if args.save_replay and hasattr(model, 'save_replay_buffer'):
        model.save_replay_buffer(seed_dir / 'replay_buffer.pkl')

    final = evaluate_agent(model, eval_seeds, f'{args.algo}_final')
    pd.DataFrame(step_cb.episodes).to_csv(seed_dir / 'train_episodes.csv', index=False)
    pd.DataFrame(step_cb.losses).to_csv(seed_dir / 'losses.csv', index=False)

    # Evaluate saved checkpoints (convergence ablation).
    checkpoint_rows = []
    for ckpt in sorted(checkpoint_dir.glob('step_*/model.zip')):
        if args.checkpoint_every == 0:
            break
        step = int(ckpt.parent.name.split('_')[1])
        try:
            loaded = type(model).load(ckpt, device='cpu')
            seeds = [args.seed + 100_000 + i for i in range(args.checkpoint_eval_episodes)]
            df = evaluate_agent(loaded, seeds, f'{args.algo}_ckpt_{step}')
            checkpoint_rows.append(dict(step=step,
                return_mean=float(df.return_.mean()),
                return_std=float(df.return_.std(ddof=1) if len(df) > 1 else 0.0),
                x_progress_mean=float(df.x_progress.mean()),
                fall_rate=float(df.terminated.mean())))
            print(json.dumps({'event': 'checkpoint', **checkpoint_rows[-1]}), flush=True)
        except Exception as exc:  # keep going if one checkpoint fails
            print(json.dumps({'event': 'checkpoint_error', 'step': step, 'error': str(exc)}),
                  flush=True)

    for frame in (random_eval, initial, final):
        frame['training_seed'] = args.seed
    evaluations = pd.concat([random_eval, initial, final], ignore_index=True)
    evaluations.to_csv(run_dir / 'evaluation_all.csv', index=False)

    summary = dict(
        status='done', algo=args.algo, seed=args.seed, steps=int(model.num_timesteps),
        gradient_updates=int(getattr(model, '_n_updates', 0)), train_seconds=round(train_seconds, 1),
        budget_completed=bool(model.num_timesteps >= args.steps),
        random_return_mean=float(random_eval.return_.mean()),
        initial_return_mean=float(initial.return_.mean()),
        final_return_mean=float(final.return_.mean()),
        final_return_std=float(final.return_.std(ddof=1) if len(final) > 1 else 0.0),
        final_x_progress_mean=float(final.x_progress.mean()),
        final_length_mean=float(final.length.mean()),
        final_fall_rate=float(final.terminated.mean()),
        eval_episodes=args.eval_episodes, checkpoints=checkpoint_rows,
        run_dir=str(run_dir.resolve()), config=config)
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'event': 'summary', 'summary': summary}), flush=True)


if __name__ == '__main__':
    main()
