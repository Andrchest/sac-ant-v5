import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from deprecated.overnight_sac import assess_rows, extension_budget
from support.sac_monitor import RunDashboard
from support.training_control import TrainingControl, parse_stop_at, validate_resume
from train_sac import parse_args


class FakeModel:
    num_timesteps = 100
    _n_updates = 90

    def save(self, path):
        Path(path).write_bytes(b'model')

    def save_replay_buffer(self, path):
        Path(path).write_bytes(b'replay')


def test_deadline_stops_without_starting_another_update(tmp_path):
    control = TrainingControl(tmp_path, stop_at=10, clock=lambda: 11)
    assert control.on_step(FakeModel()) is False
    assert control.stop_reason == 'deadline'


def test_checkpoints_are_committed_as_complete_bundles_and_rotated(tmp_path):
    control = TrainingControl(tmp_path, checkpoint_every=100, clock=lambda: 0)
    model = FakeModel()
    for step in [100, 200, 300]:
        model.num_timesteps = step
        assert control.on_step(model)
    root = tmp_path / 'checkpoints'
    bundle = Path(json.loads((root / 'latest.json').read_text())['directory'])
    assert bundle.name == 'step_000000300'
    assert (bundle / 'model.zip').read_bytes() == b'model'
    assert (bundle / 'replay_buffer.pkl').read_bytes() == b'replay'
    assert len(list(root.glob('step_*'))) == 2
    assert not list(root.glob('.tmp*'))


def test_resume_rejects_smoke_as_full_and_missing_replay(tmp_path):
    run = tmp_path / 'run'
    seed = run / 'seed_42'
    seed.mkdir(parents=True)
    model = seed / 'sac_final.zip'
    model.write_bytes(b'model')
    (run / 'config.json').write_text(json.dumps({'mode': 'smoke', 'seeds': [42]}))
    with pytest.raises(ValueError, match='full'):
        validate_resume(model, mode='full', seeds=[42])
    (run / 'config.json').write_text(json.dumps({'mode': 'full', 'seeds': [42]}))
    with pytest.raises(FileNotFoundError):
        validate_resume(model, mode='full', seeds=[42])
    (seed / 'replay_buffer.pkl').write_bytes(b'replay')
    assert validate_resume(model, mode='full', seeds=[42]) == model.resolve()


def test_resume_selects_seed_from_multiseed_run(tmp_path):
    run = tmp_path / 'run'
    seed = run / 'seed_42'
    seed.mkdir(parents=True)
    model = seed / 'sac_final.zip'
    model.write_bytes(b'model')
    (seed / 'replay_buffer.pkl').write_bytes(b'replay')
    (run / 'config.json').write_text(json.dumps({'mode': 'full', 'seeds': [42, 43, 44]}))
    assert validate_resume(model, 'full', [42]) == model.resolve()
    with pytest.raises(ValueError, match='same training seed'):
        validate_resume(model, 'full', [43])


def test_real_multiseed_checkpoint_restores_model_optimizer_replay_and_continues(tmp_path):
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3 import SAC

    torch.set_num_threads(1)
    run = tmp_path / 'run'
    seed = run / 'seed_42'
    seed.mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps({'mode': 'full', 'seeds': [42, 43]}))
    env = gym.make('Pendulum-v1')
    try:
        model = SAC('MlpPolicy', env, seed=42, device='cpu', buffer_size=128,
                    learning_starts=2, batch_size=4, policy_kwargs={'net_arch': [8, 8]})
        model.learn(total_timesteps=12)
        path = seed / 'sac_final.zip'
        model.save(path)
        model.save_replay_buffer(seed / 'replay_buffer.pkl')
        validate_resume(path, 'full', [42])
        restored = SAC.load(path, env=env, device='cpu')
        restored.load_replay_buffer(seed / 'replay_buffer.pkl')
        assert restored.num_timesteps == model.num_timesteps == 12
        assert restored._n_updates == model._n_updates
        assert restored.replay_buffer.size() == model.replay_buffer.size()
        np.testing.assert_array_equal(restored.replay_buffer.observations, model.replay_buffer.observations)
        for old, new in zip(model.actor.parameters(), restored.actor.parameters()):
            assert torch.equal(old, new)
        old_opt, new_opt = model.actor.optimizer.state_dict(), restored.actor.optimizer.state_dict()
        assert old_opt['param_groups'] == new_opt['param_groups']
        for key in old_opt['state']:
            for name in old_opt['state'][key]:
                a, b = old_opt['state'][key][name], new_opt['state'][key][name]
                assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
        restored.learn(total_timesteps=4, reset_num_timesteps=False)
        assert restored.num_timesteps == 16 and restored._n_updates > model._n_updates
    finally:
        env.close()


def test_deadline_requires_an_explicit_timezone():
    assert parse_stop_at('2026-10-22T07:00:00+03:00') == datetime(2026, 10, 22, 4, tzinfo=timezone.utc).timestamp()
    with pytest.raises(ValueError):
        parse_stop_at('2026-10-22T07:00:00')


def test_standing_is_not_a_success_even_with_survival_reward():
    rows = [dict(return_='1000', x_progress='0.1', length='1000', terminated='False')] * 10
    assert not assess_rows(rows)['sufficient']
    rows = [dict(return_='3500', x_progress='100', length='1000', terminated='False')] * 10
    assert assess_rows(rows)['sufficient']
    assert '30 episodes' in assess_rows(rows * 3)['note']


def test_short_or_nonfinite_evaluation_is_rejected():
    with pytest.raises(ValueError):
        assess_rows([])
    rows = [dict(return_='nan', x_progress='0', length='1000', terminated='False')] * 10
    with pytest.raises(ValueError):
        assess_rows(rows)


def test_extension_respects_available_time_and_budget():
    assert extension_budget(1_000_000, 3600, 100) == 250_000
    assert extension_budget(1_000_000, 60, 100) == 0
    assert extension_budget(3_000_000, 3600, 100) == 0
    assert extension_budget(1_000_000, 1000, 100) == 50_000


def test_deadline_and_resume_flags_parse_without_changing_safe_defaults():
    args = parse_args(['--stop-at', '2026-10-22T07:00:00+03:00'])
    assert args.mode == 'smoke'
    assert args.resume is None
    assert args.checkpoint_every == 100000


def test_resumed_progress_and_speed_do_not_count_previous_steps(tmp_path):
    monitor = RunDashboard(tmp_path / 'run', 'full', 250000, tensorboard=False, interactive=False)
    monitor.begin_seed(42, 1, 1)
    monitor.start_training(initial_steps=1000000)
    monitor.update_training(1050000, 1040000, [], {})
    state = monitor.snapshot()
    assert state['progress'] == 0.2
    assert state['session_steps'] == 50000
    assert state['step_budget'] == 1250000
    monitor.finish('completed')
