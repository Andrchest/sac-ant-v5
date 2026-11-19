import csv
import json

import pytest

from followup_analysis import validate


def fixture_run(tmp_path):
    args = ['--algo', 'td3', '--seed', '60', '--steps', '1000000', '--eval-episodes', '2']
    summary = {'status': 'done', 'algo': 'td3', 'seed': 60, 'steps': 1000000,
               'eval_episodes': 2, 'budget_completed': True, 'final_return_mean': 2.,
               'final_return_std': 2 ** .5, 'final_x_progress_mean': 2., 'final_fall_rate': .5,
               'config': {'net': [256, 256], 'lr': .0003, 'batch': 256,
                          'learning_starts': 10000, 'gamma': .99, 'tau': .005, 'action_noise_std': .1}}
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    with (tmp_path / 'evaluation_all.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['policy', 'episode_seed', 'return_', 'x_progress', 'terminated'])
        writer.writeheader()
        writer.writerows([{'policy': 'td3_final', 'episode_seed': 100060, 'return_': 1,
                           'x_progress': 1, 'terminated': 'True'},
                          {'policy': 'td3_final', 'episode_seed': 100061, 'return_': 3,
                           'x_progress': 3, 'terminated': 'False'}])
    return {'status': 'done', 'exit_code': 0, 'run_dir': str(tmp_path), 'args': args}


def test_validate_checks_raw_csv_against_summary(tmp_path):
    job = fixture_run(tmp_path)
    summary, rows, _, _ = validate('td3_60', job)
    assert summary['final_return_mean'] == 2. and len(rows) == 2
    path = tmp_path / 'evaluation_all.csv'
    path.write_text(path.read_text().replace('td3_final,100061,3,', 'td3_final,100061,9,'))
    with pytest.raises(ValueError, match='does not match per-episode'):
        validate('td3_60', job)


def test_validate_rejects_td3_without_noise(tmp_path):
    job = fixture_run(tmp_path)
    path = tmp_path / 'summary.json'
    summary = json.loads(path.read_text())
    summary['config']['action_noise_std'] = 0
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='configuration violates'):
        validate('td3_60', job)
