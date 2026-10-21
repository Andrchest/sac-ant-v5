#!/usr/bin/env python3
"""Bounded overnight supervisor: 1M steps, evaluate, optionally resume in chunks."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from sac_monitor import atomic_json
from training_control import parse_stop_at

ROOT = Path(__file__).resolve().parent


def assess_rows(rows):
    if len(rows) < 10:
        raise ValueError('At least ten final evaluation episodes required')
    values = {key: [float(row[key]) for row in rows]
              for key in ('return_', 'x_progress', 'length')}
    if not all(math.isfinite(v) for column in values.values() for v in column):
        raise ValueError('Nonfinite evaluation metric')
    means = {key: sum(column) / len(column) for key, column in values.items()}
    fall_rate = sum(str(row['terminated']).lower() == 'true' for row in rows) / len(rows)
    sufficient = (means['return_'] >= 2500 and means['x_progress'] >= 20
                  and means['length'] >= 900 and fall_rate <= 0.2)
    return dict(mean_return=means['return_'], mean_x_progress=means['x_progress'],
                mean_length=means['length'], fall_rate=fall_rate, sufficient=sufficient,
                note=f'Operational gait heuristic on {len(rows)} episodes, not a statistical benchmark claim.')


def extension_budget(total_steps, available_seconds, steps_per_second):
    remaining_cap = 3_000_000 - total_steps
    if remaining_cap < 50000 or available_seconds <= 300 or steps_per_second <= 0:
        return 0
    affordable = int((available_seconds - 180) * steps_per_second * 0.8)
    return max(0, min(250000, remaining_cap, affordable) // 50000 * 50000)


def stop_group(process, sig):
    if process.poll() is None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass


def write_report(directory, state):
    atomic_json(directory / 'status.json', state)
    lines = ['# Overnight SAC report', '', f"Status: {state['status']}",
             f"Deadline: {state['deadline']}", f"Reason: {state.get('reason', '')}", '']
    for row in state['rounds']:
        lines.extend([f"## Round {row['round']}",
            f"Requested additional steps: {row['requested_steps']}",
            f"Run: {row.get('run_dir', 'starting')}",
            f"Exit code: {row.get('exit_code', 'running')}",
            '```json', json.dumps(row.get('assessment', {}), indent=2), '```', ''])
    lines.append('No claim of statistically significant improvement; inspect x-progress and rollout videos.')
    temporary = directory / 'report.md.tmp'
    temporary.write_text('\n'.join(lines), encoding='utf-8')
    os.replace(temporary, directory / 'report.md')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--until', required=True, help='hard deadline, ISO timestamp including timezone')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    deadline = parse_stop_at(args.until)
    if deadline - time.time() < 600:
        parser.error('At least ten minutes are needed for a safe first run')
    cutoff = deadline - 180  # reserve time for final evaluation, replay, plots, video
    directory = ROOT / 'artifacts' / datetime.now().astimezone().strftime('night_%Y%m%dT%H%M%S%z')
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(ROOT / 'artifacts/night_latest.json', {'directory': str(directory)})
    state = dict(status='running', deadline=args.until,
                 training_cutoff=datetime.fromtimestamp(cutoff).astimezone().isoformat(),
                 rounds=[], reason=None, seed=args.seed)
    write_report(directory, state)
    resume = None
    budget = 1_000_000
    process = None
    try:
        while time.time() < cutoff:
            number = len(state['rounds']) + 1
            manifest = directory / f'round_{number:03d}.json'
            log_path = directory / f'round_{number:03d}.log'
            command = [sys.executable, '-u', str(ROOT / 'train_sac.py'), '--mode', 'full',
                       '--steps', str(budget), '--seeds', str(args.seed), '--no-live',
                       '--stop-at', datetime.fromtimestamp(cutoff).astimezone().isoformat(),
                       '--manifest', str(manifest)]
            if resume:
                command.extend(['--resume', str(resume)])
            row = dict(round=number, requested_steps=budget, resume=str(resume) if resume else None,
                       log=str(log_path), started_at=datetime.now().astimezone().isoformat())
            state['rounds'].append(row)
            write_report(directory, state)
            print(f'Round {number}: +{budget:,} steps; log {log_path}', flush=True)
            with log_path.open('w', buffering=1) as output:
                process = subprocess.Popen(command, cwd=ROOT, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                row['pid'] = process.pid
                sent = set()
                while process.poll() is None:
                    now = time.time()
                    if manifest.exists() and 'run_dir' not in row:
                        row['run_dir'] = json.loads(manifest.read_text())['run_dir']
                    # The normal callback stops at cutoff. These are fallbacks for
                    # an unusually long evaluation, save, or a wedged process.
                    for offset, sig in [(90, signal.SIGINT), (30, signal.SIGTERM), (0, signal.SIGKILL)]:
                        if now >= deadline - offset and sig not in sent:
                            stop_group(process, sig)
                            sent.add(sig)
                    write_report(directory, state)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                row['exit_code'] = process.returncode
            row['finished_at'] = datetime.now().astimezone().isoformat()
            if manifest.exists():
                row['run_dir'] = json.loads(manifest.read_text())['run_dir']
            if process.returncode != 0:
                state.update(status='stopped', reason='child_failed_or_hard_deadline; inspect round log/checkpoints')
                break
            run = Path(row['run_dir'])
            summary = json.loads((run / 'summary.json').read_text())
            details = summary['runs'][0]
            with (run / 'evaluation_all.csv').open() as stream:
                evaluations = [r for r in csv.DictReader(stream) if r['policy'] == 'sac_final']
            assessment = assess_rows(evaluations)
            row.update(assessment=assessment, total_steps=details['train_timesteps'],
                       gradient_updates=details['gradient_updates'], summary=str(run / 'summary.json'))
            if not summary.get('budget_completed', True):
                state.update(status='completed', reason='training_cutoff; partial budget saved and evaluated')
                break
            if assessment['sufficient']:
                state.update(status='completed', reason='gait_quality_heuristic_met')
                break
            completed_steps = details['train_timesteps']
            rate = details['session_timesteps'] / max(1.0, details['train_seconds'])
            budget = extension_budget(completed_steps, cutoff - time.time(), rate)
            if not budget:
                state.update(status='completed', reason='insufficient_time_for_safe_extension_or_3M_cap')
                break
            resume = run / f'seed_{args.seed}' / 'sac_final.zip'
            row['decision'] = f'Quality target not yet reached; continue same model/replay for +{budget} steps'
            write_report(directory, state)
            print(row['decision'], flush=True)
        else:
            state.update(status='completed', reason='training_cutoff')
    except KeyboardInterrupt:
        if process is not None:
            stop_group(process, signal.SIGINT)
        state.update(status='interrupted', reason='supervisor_interrupted; child asked to save')
    except Exception as error:
        if process is not None:
            stop_group(process, signal.SIGINT)
        state.update(status='failed', reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_report(directory, state)
        print(f"Night status: {state['status']}; report: {directory / 'report.md'}", flush=True)
    return 0 if state['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
