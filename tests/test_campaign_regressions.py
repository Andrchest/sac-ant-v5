from importlib import import_module
import json
from pathlib import Path

import pytest

from deprecated import ablation_campaign as campaign
from deprecated.ablation_campaign import write_markdown

campaign_analysis = import_module('additional analysis.campaign_analysis')
_load_campaign = campaign_analysis._load_campaign
_validate_complete = campaign_analysis._validate_complete
REQUIRED_EXPERIMENT_IDS = campaign_analysis.REQUIRED_EXPERIMENT_IDS


def test_missing_campaign_index_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='campaign index'):
        _load_campaign(tmp_path)


def test_missing_completed_run_summary_is_an_error(tmp_path):
    (tmp_path / 'campaign_summary.json').write_text(json.dumps({'experiments': [
        {'id': 'example', 'status': 'done', 'run_dir': str(tmp_path / 'missing')}
    ]}))
    with pytest.raises(FileNotFoundError, match='experiment summary'):
        _load_campaign(tmp_path)


def complete_summaries():
    metrics = {'final_return_mean': 1., 'final_return_std': 1.,
               'final_x_progress_mean': 1., 'final_fall_rate': 0., 'eval_episodes': 10}
    records = [dict(metrics, xp_id=x, xp_status='done', status='done') for x in REQUIRED_EXPERIMENT_IDS]
    next(r for r in records if r['xp_id'] == 'conv_s43_2m')['checkpoints'] = [
        {'step': step, 'return_mean': 1., 'return_std': 1., 'x_progress_mean': 1., 'fall_rate': 0.}
        for step in range(250000, 2000001, 250000)]
    return records


def test_full_analysis_rejects_missing_experiment():
    records = complete_summaries()
    records = [r for r in records if r['xp_id'] != 'td3_1m']
    with pytest.raises(ValueError, match='td3_1m'):
        _validate_complete(records)


def test_full_analysis_rejects_failed_duplicate_and_invalid_metrics():
    records = complete_summaries()
    records[0]['xp_status'] = 'running'
    with pytest.raises(ValueError, match='not completed'):
        _validate_complete(records)
    records = complete_summaries()
    records.append(records[0])
    with pytest.raises(ValueError, match='Duplicate'):
        _validate_complete(records)
    records = complete_summaries()
    records[0]['final_return_mean'] = float('nan')
    with pytest.raises(ValueError, match='final_return_mean'):
        _validate_complete(records)


def test_full_analysis_rejects_missing_checkpoint():
    records = complete_summaries()
    next(r for r in records if r['xp_id'] == 'conv_s43_2m')['checkpoints'].pop()
    with pytest.raises(ValueError, match='checkpoints'):
        _validate_complete(records)


def test_full_analysis_accepts_complete_data():
    _validate_complete(complete_summaries())


def test_interrupt_waits_for_active_worker_and_serializes_checkpoints(tmp_path, monkeypatch):
    import sys
    import threading
    import types

    release = threading.Event()
    handlers = {}
    experiment = {'id': 'conv', 'priority': 1, 'algo': 'sac', 'seed': 43,
                  'steps': 100, 'net': '256,256', 'lr': .0003, 'batch': 256,
                  'eval_episodes': 1}
    monkeypatch.setattr(campaign, 'EXPERIMENTS', [experiment])
    monkeypatch.setattr(sys, 'argv', ['deprecated.ablation_campaign', '--workers', '1', '--threads', '1',
                                    '--deadline', '2099-01-01T00:00:00+00:00',
                                    '--output', str(tmp_path)])
    monkeypatch.setattr(campaign.signal, 'signal', lambda sig, fn: handlers.update({sig: fn}))
    def fake_run(cmd, stdout, stop, grace_seconds):
        assert release.wait(timeout=5)
        summary = {'run_dir': '/test', 'checkpoints': [{'step': 100, 'return_mean': 123.,
                    'return_std': 1., 'x_progress_mean': 2., 'fall_rate': 0.}]}
        stdout.write(json.dumps({'event': 'summary', 'summary': summary}) + '\n')
        stdout.flush()
        return 0
    monkeypatch.setattr(campaign, 'run_process', fake_run)
    def interrupt(_):
        handlers[campaign.signal.SIGINT](campaign.signal.SIGINT, None)
        release.set()
    monkeypatch.setattr(campaign.time, 'sleep', interrupt)
    campaign.main()
    rows = json.loads((tmp_path / 'campaign_summary.json').read_text())['experiments']
    assert rows[0]['status'] == 'done'
    assert rows[0]['checkpoints'][0]['return_mean'] == 123.
    assert '| 100 | 123.0 |' in (tmp_path / 'campaign_report.md').read_text(encoding='utf-8')


def test_real_sigint_terminates_active_process_after_grace(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time

    code = '''import sys
from deprecated import ablation_campaign as c
c.EXPERIMENTS = [{'id':'probe','priority':1,'algo':'sac','seed':1,'steps':100,
                  'net':'256,256','lr':0.0003,'batch':256,'eval_episodes':1}]
orig = c.run_process
def run(cmd, output, stop, grace):
    return orig([sys.executable, '-u', '-c', 'import os,time; print(os.getpid(),flush=True); time.sleep(60)'], output, stop, grace)
c.run_process = run
c.main()
'''
    cmd = [sys.executable, '-c', code, '--workers', '1', '--threads', '1', '--shutdown-grace', '0.1',
           '--deadline', '2099-01-01T00:00:00+00:00', '--output', str(tmp_path)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        log = tmp_path / 'probe.log'
        limit = time.monotonic() + 10
        while (not log.exists() or not log.read_text().strip()) and time.monotonic() < limit:
            time.sleep(.05)
        assert log.exists() and log.read_text().strip(), 'child did not start'
        child_pid = int(log.read_text().splitlines()[0])
        os.kill(proc.pid, signal.SIGINT)
        output, _ = proc.communicate(timeout=10)
        assert proc.returncode == 0, output
        state = json.loads((tmp_path / 'campaign_summary.json').read_text())
        assert state['experiments'][0]['status'] == 'error'
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_checkpoint_evaluations_render_in_report(tmp_path):
    ref = {'seed': 42, 'steps': 1000000, 'net': [256, 256],
           'final_return_mean': 1., 'final_return_std': 0.,
           'eval_episodes': 10, 'final_x_progress_mean': 1., 'final_fall_rate': 0.}
    row = {'id': 'conv', 'priority': 1, 'algo': 'sac', 'seed': 43,
           'steps': 2000000, 'net': '256,256', 'lr': .0003, 'batch': 256,
           'status': 'done', 'final_return_mean': 1., 'final_return_std': 0.,
           'final_x_progress_mean': 1., 'final_fall_rate': 0., 'train_seconds': 1,
           'checkpoints': [{'step': 250000, 'return_mean': 123., 'return_std': 5.,
                            'x_progress_mean': 3., 'fall_rate': 0.}]}
    report = tmp_path / 'report.md'
    write_markdown(report, {'reference': ref, 'deadline': 'deadline', 'workers': 1,
                            'threads': 1, 'experiments': [row]})
    assert '| 250,000 | 123.0 |' in report.read_text(encoding='utf-8')
