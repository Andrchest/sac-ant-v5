from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise

from followup_campaign import experiment_jobs
from run_experiment import build_policy


def test_protocol_jobs_are_seed_paired_and_noise_is_explicit():
    jobs = dict(experiment_jobs())
    assert len(jobs) == 10
    for algo in ('sac', 'td3', 'ppo'):
        for seed in (60, 61, 62):
            args = jobs[f'{algo}_{seed}']
            assert args[args.index('--steps') + 1] == '1000000'
            assert args[args.index('--eval-episodes') + 1] == '30'
            if algo == 'td3':
                assert args[args.index('--action-noise-std') + 1] == '0.1'
            else:
                assert '--action-noise-std' not in args
    assert jobs['sac_control_42_500k'][jobs['sac_control_42_500k'].index('--steps') + 1] == '500000'


def test_td3_exploration_noise_is_present_and_configured():
    env = gym.make('Pendulum-v1')
    try:
        args = SimpleNamespace(seed=1, lr=3e-4, gamma=.99, net='8,8', steps=100,
                               learning_starts=10, batch=8, tau=.005, action_noise_std=.1)
        model = build_policy('td3', env, args)
        assert isinstance(model.action_noise, NormalActionNoise)
        np.testing.assert_allclose(model.action_noise._sigma, np.full(env.action_space.shape, .1))
    finally:
        env.close()
