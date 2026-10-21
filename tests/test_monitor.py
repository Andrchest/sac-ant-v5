import json
from pathlib import Path

import pytest
from rich.console import Console

from sac_monitor import RunDashboard, atomic_json, estimate_progress, sparkline
from train_sac import parse_args
from watch_sac import resolve_run


def test_step_eta_is_step_based_and_zero_safe():
    assert estimate_progress(0, 100, 0, None, 5) == (0.0, None)
    progress, remaining = estimate_progress(25, 100, 7, None, 10)
    assert progress == 0.25
    assert remaining == 30


def test_smoke_eta_uses_episodes_not_maximum_step_cap():
    progress, remaining = estimate_progress(100, 20000, 5, 20, 30)
    assert progress == 0.25
    assert remaining == 90
    assert estimate_progress(200, 20000, 20, 20, 30) == (1.0, 0.0)


def test_sparkline_is_finite_and_handles_empty_flat_and_nan():
    assert sparkline([]) == 'waiting for data'
    assert len(sparkline([2, 2, 2])) == 3
    assert len(sparkline([float('nan'), 1, float('inf'), 2])) == 2
    assert len(sparkline(list(range(100)), width=10)) == 10
    assert len(sparkline([1, 2, 3], width=1)) == 1


def test_defaults_never_start_full_training():
    args = parse_args([])
    assert args.mode == 'smoke'
    assert args.episodes == 20
    assert args.steps is None
    with pytest.raises(SystemExit):
        parse_args(['--episodes', '0'])
    with pytest.raises(SystemExit):
        parse_args(['--steps', '1000000'])


def test_snapshot_png_terminal_and_final_state(tmp_path):
    run = tmp_path / 'run'
    dashboard = RunDashboard(run, mode='smoke', step_budget=20000,
                             max_episodes=20, tensorboard=False,
                             interactive=False, plot_seconds=0)
    dashboard.begin_seed(42, 1, 1)
    dashboard.start_training()
    dashboard.update_training(steps=500, updates=399,
        episodes=[dict(episode=1, timestep=100, return_=1.5, length=100)],
        losses={'train/actor_loss': -2, 'train/critic_loss': 0.5, 'train/ent_coef': 0.9})
    dashboard.record_evaluation(500, [10, 20])
    dashboard.publish(force=True)
    data = json.loads((run / 'progress.json').read_text())
    assert data['steps'] == 500 and data['updates'] == 399
    assert data['episodes'] == 1
    assert data['elapsed_seconds'] >= 0 and data['eta_seconds'] is not None
    assert (run / 'live_curves.png').read_bytes().startswith(b'\x89PNG')
    assert (run / 'seed_42' / 'live_episodes.csv').is_file()
    assert len((run / 'seed_42' / 'live_evaluations.csv').read_text().splitlines()) == 2
    console = Console(record=True, width=120, force_terminal=False)
    console.print(dashboard.render())
    rendered = console.export_text()
    for label in ['Elapsed', 'ETA', 'Steps', 'Episodes', 'Evaluation', 'Reward', 'Critic']:
        assert label in rendered
    dashboard.finish('completed')
    data = json.loads((run / 'progress.json').read_text())
    assert data['status'] == 'completed' and data['eta_seconds'] == 0
    assert resolve_run(None, tmp_path) == run.resolve()


def test_failed_run_is_not_success_and_is_viewable(tmp_path):
    run = tmp_path / 'failed'
    dashboard = RunDashboard(run, 'smoke', 20000, 20,
                             tensorboard=False, interactive=False)
    dashboard.finish('failed', error='test error')
    data = json.loads((run / 'progress.json').read_text())
    assert data['status'] == 'failed' and data['error'] == 'test error'
    assert data['eta_seconds'] is None
    assert resolve_run(str(run), tmp_path) == run.resolve()


def test_explicit_run_never_silently_selects_another(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_run(str(tmp_path / 'missing'), tmp_path)


def test_concurrent_latest_pointer_is_atomic(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / 'latest.json'
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: atomic_json(path, {'id': i}), range(20)))
    assert json.loads(path.read_text())['id'] in range(20)
    assert not list(tmp_path.glob('*.tmp'))


def test_json_heartbeat_cannot_starve_live_png(tmp_path, monkeypatch):
    import time
    dashboard = RunDashboard(tmp_path / 'run', 'smoke', 20000, 20,
                             tensorboard=False, interactive=False)
    calls = []
    monkeypatch.setattr(dashboard, '_plot', lambda state: calls.append(state))
    dashboard.last_publish = time.monotonic()  # heartbeat just wrote JSON
    dashboard.last_plot = 0
    dashboard.publish()
    assert len(calls) == 1
    dashboard.finish('completed')
