#!/usr/bin/env python3
"""Run the follow-up experiments defined in EXPERIMENT_PROTOCOL.md."""
import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parent


def experiment_jobs():
    jobs = []
    for algo in ('sac', 'td3', 'ppo'):
        for seed in (60, 61, 62):
            args = ['--algo', algo, '--seed', str(seed), '--steps', '1000000',
                    '--net', '256,256', '--lr', '0.0003', '--batch', '256',
                    '--learning-starts', '10000', '--eval-episodes', '30',
                    '--ppo-n-steps', '2000', '--threads', '4', '--tag', f'followup_{algo}_{seed}']
            if algo == 'td3':
                args += ['--action-noise-std', '0.1']
            jobs.append((f'{algo}_{seed}', args))
    jobs.append(('sac_control_42_500k', ['--algo', 'sac', '--seed', '42', '--steps', '500000',
        '--net', '256,256', '--lr', '0.0003', '--batch', '256', '--learning-starts', '10000',
        '--eval-episodes', '10', '--threads', '4', '--tag', 'followup_control_42_500k']))
    return jobs


def main():
    os.chdir(ROOT)
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    if available < 9 * 1024**3:
        raise SystemExit(f'Refusing three workers: only {available / 1024**3:.1f} GiB available')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'artifacts' / f'followup_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    jobs = experiment_jobs()
    state = {'protocol': 'EXPERIMENT_PROTOCOL.md', 'started': stamp,
             'jobs': {name: {'status': 'pending', 'args': args} for name, args in jobs}}
    lock = threading.Lock()
    stop = threading.Event()

    def publish():
        temp = out / 'status.json.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        os.replace(temp, out / 'status.json')

    def on_signal(signum, frame):
        stop.set()
        print('Signal received: no new runs will be launched; waiting for active runs.', flush=True)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    def execute(name, args):
        if stop.is_set():
            return
        cmd = [sys.executable, '-u', str(ROOT / 'run_experiment.py'), *args]
        log = out / f'{name}.log'
        with lock:
            state['jobs'][name].update(status='running', command=cmd)
            publish()
        with log.open('w', encoding='utf-8') as stream:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        summary = None
        for line in reversed(log.read_text(encoding='utf-8').splitlines()):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get('event') == 'summary':
                summary = obj.get('summary')
                break
        ok = proc.returncode == 0 and summary and summary.get('budget_completed')
        with lock:
            state['jobs'][name].update(status='done' if ok else 'failed', exit_code=proc.returncode,
                                       run_dir=summary.get('run_dir') if summary else None)
            publish()
        print(f'{name}: {state["jobs"][name]["status"]} exit={proc.returncode}', flush=True)
        if not ok:
            stop.set()

    with lock:
        publish()
    print(f'Follow-up output: {out}', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(execute, name, args) for name, args in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print('Follow-up finished; pending/failed runs remain visible in status.json.', flush=True)
    return 0 if all(v['status'] == 'done' for v in state['jobs'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
