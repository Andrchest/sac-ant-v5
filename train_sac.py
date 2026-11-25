#!/usr/bin/env python3
"""Run the trusted local notebook's experiment cells as a plain background process."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke',
                        help='default: smoke (20 episodes), NOT one million steps')
    parser.add_argument('--episodes', type=positive_int, default=20,
                        help='training episode budget in smoke mode')
    parser.add_argument('--steps', type=positive_int,
                        help='override full-mode budget (at least 50000); requires --mode full')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts')
    parser.add_argument('--no-live', action='store_true',
                        help='plain logs instead of animated terminal panel; disk charts still update')
    parser.add_argument('--no-tensorboard', action='store_true')
    from support.training_control import parse_stop_at
    parser.add_argument('--stop-at', type=parse_stop_at, help='training cutoff ISO timestamp with timezone')
    parser.add_argument('--resume', type=Path, help='full-mode sac_final.zip with sibling replay_buffer.pkl')
    parser.add_argument('--checkpoint-every', type=positive_int, default=100000)
    parser.add_argument('--manifest', type=Path, help='write the exact new run path for an external supervisor')
    args = parser.parse_args(argv)
    if args.steps is not None and args.mode != 'full':
        parser.error('--steps requires explicit --mode full; use --episodes for smoke')
    if args.steps is not None and args.steps < 50000:
        parser.error('full budget must allow warmup and evaluation: --steps >= 50000')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 - 1_000_000 for s in args.seeds):
        parser.error('seeds must be unique integers in [0, 2**32 - 1000000)')
    if args.resume is not None and (args.mode != 'full' or len(args.seeds) != 1):
        parser.error('--resume requires --mode full and one seed')
    return args


def save_emergency(namespace):
    model, seed_dir = namespace.get('model'), namespace.get('seed_dir')
    if model is not None and seed_dir is not None:
        try:
            from support.training_control import TrainingControl
            path = TrainingControl(seed_dir).save_checkpoint(model)
            print(f'Emergency checkpoint + replay saved: {path}', flush=True)
        except Exception as error:
            print(f'Emergency save failed: {type(error).__name__}: {error}', flush=True)


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    os.environ['ANT_SAC_MODE'] = args.mode
    os.environ['ANT_SAC_SEEDS'] = json.dumps(args.seeds)
    os.environ['ANT_SAC_EPISODES'] = str(args.episodes)
    os.environ['ANT_SAC_OUTPUT'] = str(args.output.resolve())
    from support.training_control import TrainingControl, validate_resume
    if args.resume is not None:
        args.resume = validate_resume(args.resume, args.mode, args.seeds)
        os.environ['ANT_SAC_RESUME'] = str(args.resume)
    else:
        os.environ.pop('ANT_SAC_RESUME', None)
    if args.steps is not None:
        os.environ['ANT_SAC_STEPS'] = str(args.steps)
    else:
        os.environ.pop('ANT_SAC_STEPS', None)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    notebook_path = ROOT / 'sac_ant_v5.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    namespace = {'__name__': '__sac_experiment__', 'DASHBOARD': None,
        'DEADLINE_REACHED': lambda: args.stop_at is not None and time.time() >= args.stop_at,
        'TRAINING_CONTROL_FACTORY': lambda model, seed_dir: TrainingControl(
            seed_dir, args.stop_at, args.checkpoint_every)}
    dashboard = None
    try:
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code' or cell['id'] in ('install', 'background-viewer', 'toy-sac-update'):
                continue
            source = cell['source']
            if isinstance(source, list):
                source = ''.join(source)
            print(f"Executing {cell['id']}…", flush=True)
            if dashboard and cell['id'] in ('plots', 'video', 'summary'):
                dashboard.set_phase(cell['id'])
            # Only the repository's trusted notebook is executed. No downloaded code.
            exec(compile(source, f'{notebook_path}:cell-{index}', 'exec'), namespace)
            if cell['id'] == 'config':
                from support.sac_monitor import RunDashboard
                cfg = namespace['CFG']
                if args.manifest:
                    from support.sac_monitor import atomic_json
                    args.manifest.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(args.manifest, {'run_dir': str(namespace['RUN_DIR'].resolve())})
                dashboard = RunDashboard(namespace['RUN_DIR'], args.mode,
                    cfg['total_timesteps'], cfg['train_episodes'],
                    tensorboard=not args.no_tensorboard,
                    interactive=False if args.no_live else None).start()
                namespace['DASHBOARD'] = dashboard
                namespace['display'] = lambda value: print(
                    value.to_string() if hasattr(value, 'to_string') else str(value), flush=True)
                namespace['plt'].show = lambda: namespace['plt'].close('all')
        complete = namespace.get('SUMMARY', {}).get('budget_completed', True)
        if dashboard:
            dashboard.finish('completed' if complete else 'partial')
        print('PASS — background experiment completed.' if complete else
              'PARTIAL — deadline stopped training; partial artifacts saved.', flush=True)
        return 0
    except KeyboardInterrupt:
        save_emergency(namespace)
        if dashboard:
            dashboard.finish('interrupted', error='KeyboardInterrupt; completed snapshots retained')
        print('Interrupted. Existing snapshots/checkpoints are retained.', flush=True)
        return 130
    except Exception as error:
        save_emergency(namespace)
        if dashboard:
            dashboard.finish('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
