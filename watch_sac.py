#!/usr/bin/env python3
"""Attach a read-only dashboard to an independent background SAC process."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def resolve_run(value=None, root=Path(__file__).resolve().parent / 'artifacts'):
    root = Path(root)
    if value:
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f'Run directory does not exist: {path}')
        return path
    latest = root / 'latest_run.json'
    if not latest.is_file():
        raise FileNotFoundError(f'No background run yet: {latest}')
    path = Path(json.loads(latest.read_text())['run_dir'])
    if not path.is_dir():
        raise FileNotFoundError(f'Latest run directory is missing: {path}')
    return path.resolve()


def read_snapshot(run_dir):
    path = Path(run_dir) / 'progress.json'
    state = json.loads(path.read_text(encoding='utf-8'))
    state['snapshot_age_seconds'] = max(0.0, time.time() - state['updated_at'])
    return state


def show_notebook(run=None, *, auto_refresh=False, seconds=2):
    """Read-only: rerunning this function never starts or restarts training."""
    from IPython.display import display, clear_output, Image, Markdown
    path = resolve_run(run)
    try:
        while True:
            state = read_snapshot(path)
            from sac_monitor import duration
            if auto_refresh:
                clear_output(wait=True)
            display(Markdown(f"### SAC × Ant-v5 — {state['phase']}\n"
                f"**Steps:** {state['steps']:,} · **Episodes:** {state['episodes']} · "
                f"**Elapsed:** {duration(state['elapsed_seconds'])} · "
                f"**ETA (training):** {duration(state['eta_seconds'])} · "
                f"**Snapshot age:** {state['snapshot_age_seconds']:.0f}s\n\n`{path}`"))
            image = path / 'live_curves.png'
            if image.is_file():
                display(Image(data=image.read_bytes()))
            if not auto_refresh or state['status'] != 'running':
                break
            time.sleep(seconds)
    except KeyboardInterrupt:
        print('Viewer stopped; the background training process is unaffected.')
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', nargs='?', help='run directory; default: latest background run')
    parser.add_argument('--once', action='store_true', help='print one snapshot and exit')
    parser.add_argument('--interval', type=float, default=1)
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error('--interval must be positive')
    from rich.console import Console, Group
    from rich.live import Live
    from rich.text import Text
    from sac_monitor import render_snapshot
    path = resolve_run(args.run)
    console = Console()

    def render():
        state = read_snapshot(path)
        age = state['snapshot_age_seconds']
        warning = f'Snapshot age: {age:.0f}s'
        if state['status'] == 'running' and age > 10:
            warning += ' — stale! Training may have stopped; inspect events.log/process.'
        return state, Group(render_snapshot(state), Text(warning, style='yellow' if age > 10 else 'dim'))

    state, panel = render()
    if args.once or not console.is_terminal:
        console.print(panel)
        return
    try:
        with Live(panel, console=console, refresh_per_second=2) as live:
            while state['status'] == 'running':
                time.sleep(args.interval)
                state, panel = render()
                live.update(panel)
    except KeyboardInterrupt:
        console.print('Detached. Training continues independently.')


if __name__ == '__main__':
    main()
