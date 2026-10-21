import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from training_control import TrainingControl, parse_stop_at, validate_resume
from overnight_sac import assess_rows, extension_budget
from train_sac import parse_args
from sac_monitor import RunDashboard


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
