"""Live terminal, durable snapshots and TensorBoard for the notebook experiment.

No RL algorithm lives here. This module only observes training.
"""
from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
import tempfile
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text


def estimate_progress(steps, budget, episodes, episode_budget, elapsed):
    """Return bounded fraction and estimated TRAINING time left, not whole-run ETA."""
    fraction = (episodes / episode_budget if episode_budget else steps / budget)
    fraction = min(1.0, max(0.0, fraction))
    return fraction, elapsed * (1 - fraction) / fraction if fraction else None


def sparkline(values, width=65):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return 'waiting for data'
    width = max(1, width)
    if width == 1:
        values = values[-1:]
    if len(values) > width:
        indices = [round(i * (len(values) - 1) / (width - 1)) for i in range(width)]
        values = [values[i] for i in indices]
    low, high = min(values), max(values)
    levels = '▁▂▃▄▅▆▇█'
    return ''.join(levels[round((v - low) / (high - low) * 7)]
                   if high > low else levels[3] for v in values)


def duration(seconds):
    if seconds is None:
        return 'estimating…'
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def atomic_json(path, value):
    path = Path(path)
    # Unique sibling temp files also make the shared latest-run pointer safe
    # when independent processes launch at the same time.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def append_csv(path, fields, row):
    path = Path(path)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class RunDashboard:
    def __init__(self, run_dir, mode, step_budget, max_episodes=None, *,
                 tensorboard=True, interactive=None, plot_seconds=5):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.step_budget = step_budget
        self.max_episodes = max_episodes
        self.tensorboard = tensorboard
        self.console = Console()
        self.interactive = self.console.is_terminal if interactive is None else interactive
        self.plot_seconds = plot_seconds
        self.lock = threading.RLock()
        self.write_lock = threading.Lock()
        self.created = time.monotonic()
        self.training_started = None
        self.training_stopped = None
        self.initial_steps = 0
        self.last_publish = 0.0
        self.last_plot = 0.0
        self.last_log = 0.0
        self.stop_event = threading.Event()
        self.thread = None
        self.live = None
        self.writer = None
        self.episodes = []
        self.metrics = []
        self.evaluations = []
        self.seen_episodes = 0
        self.state = dict(status='running', phase='initializing', mode=mode,
                          run_dir=str(self.run_dir), seed=None, seed_index=0,
                          seed_count=0, steps=0, episodes=0, updates=0,
                          step_budget=step_budget, episode_budget=max_episodes,
                          phase_current=None, phase_total=None, error=None)
        atomic_json(self.run_dir.parent / 'latest_run.json', {'run_dir': str(self.run_dir)})
        self.publish(force=True, plot=False)

    def start(self):
        if self.interactive:
            from rich.live import Live
            self.live = Live(get_renderable=self.render, console=self.console,
                             refresh_per_second=2, transient=False)
            self.live.start()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        self.event('Dashboard started')
        return self

    def _heartbeat(self):
        while not self.stop_event.wait(1):
            # Plotting stays on the training thread; JSON remains live during evaluations.
            self.publish(plot=False)

    def event(self, message):
        with (self.run_dir / 'events.log').open('a', encoding='utf-8') as stream:
            stream.write(time.strftime('%Y-%m-%d %H:%M:%S ') + message + '\n')
        if not self.interactive:
            self.console.print(message, markup=False)

    def begin_seed(self, seed, index, total):
        if self.writer:
            self.writer.close()
        self.writer = None
        if self.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.run_dir / 'tensorboard' / f'seed_{seed}'),
                                        flush_secs=3)
        with self.lock:
            self.state.update(seed=seed, seed_index=index, seed_count=total,
                              steps=0, episodes=0, updates=0, phase='baseline',
                              phase_current=0, phase_total=None)
            self.episodes, self.metrics, self.evaluations = [], [], []
            self.seen_episodes = 0
            self.training_started = self.training_stopped = None
            self.seed_dir = self.run_dir / f'seed_{seed}'
            self.seed_dir.mkdir(parents=True, exist_ok=True)
        self.publish(force=True)

    def set_phase(self, phase, current=None, total=None):
        with self.lock:
            self.state.update(phase=phase, phase_current=current, phase_total=total)
        self.publish(force=True, plot=False)

    def start_training(self, initial_steps=0):
        with self.lock:
            self.initial_steps = int(initial_steps)
            self.state.update(steps=self.initial_steps,
                              step_budget=self.step_budget + self.initial_steps,
                              initial_steps=self.initial_steps)
            self.training_started = time.monotonic()
            self.training_stopped = None
        self.set_phase('training')
        self.event(f"Training seed {self.state['seed']}; target: "
                   f"{self.max_episodes or self.step_budget} "
                   f"{'episodes' if self.max_episodes else 'steps'}")

    def end_training(self, steps, updates):
        with self.lock:
            self.state.update(steps=int(steps), updates=int(updates))
            self.training_stopped = time.monotonic()
        self.set_phase('final evaluation')
        self.publish(force=True)

    def update_training(self, steps, updates, episodes, losses):
        with self.lock:
            self.state.update(steps=int(steps), updates=int(updates), episodes=len(episodes))
            for row in episodes[self.seen_episodes:]:
                row = dict(row)
                self.episodes.append(row)
                append_csv(self.seed_dir / 'live_episodes.csv',
                           ['episode', 'timestep', 'return_', 'length'], row)
                if self.writer:
                    self.writer.add_scalar('rollout/episode_return', row['return_'], steps)
                    self.writer.add_scalar('rollout/episode_length', row['length'], steps)
            self.seen_episodes = len(episodes)
            if steps % 100 == 0 and losses:
                row = {'timestep': int(steps)}
                for key in ('train/actor_loss', 'train/critic_loss', 'train/ent_coef'):
                    if key in losses and math.isfinite(float(losses[key])):
                        row[key] = float(losses[key])
                        if self.writer:
                            self.writer.add_scalar(key, row[key], steps)
                self.metrics.append(row)
                append_csv(self.seed_dir / 'live_metrics.csv',
                           ['timestep', 'train/actor_loss', 'train/critic_loss', 'train/ent_coef'], row)
                if self.writer:
                    self.writer.add_scalar('time/gradient_updates', updates, steps)
        self.publish()

    def record_evaluation(self, steps, returns):
        values = list(map(float, returns))
        row = dict(timestep=int(steps), mean_return=sum(values) / len(values))
        with self.lock:
            self.evaluations.append(row)
            append_csv(self.seed_dir / 'live_evaluations.csv', list(row), row)
            if self.writer:
                self.writer.add_scalar('eval/mean_return', row['mean_return'], steps)
                self.writer.flush()
        self.publish(force=True)

    def evaluation_callback(self, **kwargs):
        from stable_baselines3.common.callbacks import EvalCallback
        dashboard = self

        class LiveEvalCallback(EvalCallback):
            def _on_step(self):
                evaluating = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
                if evaluating:
                    dashboard.set_phase('periodic evaluation')
                try:
                    result = super()._on_step()
                    if evaluating:
                        dashboard.record_evaluation(self.num_timesteps, self.evaluations_results[-1])
                    return result
                finally:
                    if evaluating:
                        dashboard.set_phase('training')

        return LiveEvalCallback(**kwargs)

    def snapshot(self):
        with self.lock:
            state = dict(self.state)
            state['elapsed_seconds'] = time.monotonic() - self.created
            elapsed = (self.training_stopped or time.monotonic()) - self.training_started if self.training_started else 0.0
            state['training_elapsed_seconds'] = elapsed
            session_steps = max(0, state['steps'] - self.initial_steps)
            fraction, remaining = estimate_progress(session_steps, self.step_budget,
                                                    state['episodes'], self.max_episodes, elapsed)
            state['progress'] = fraction
            state['eta_seconds'] = remaining if self.training_started and not self.training_stopped else None
            if state['status'] == 'completed':
                state['eta_seconds'] = 0.0
            elif state['status'] != 'running':
                state['eta_seconds'] = None
            state['session_steps'] = session_steps
            state['steps_per_second'] = session_steps / elapsed if elapsed else 0.0
            state['reward_mean_10'] = (sum(x['return_'] for x in self.episodes[-10:]) /
                                       len(self.episodes[-10:])) if self.episodes else None
            state['latest_evaluation'] = self.evaluations[-1]['mean_return'] if self.evaluations else None
            state['updated_at'] = time.time()
            state['curves'] = dict(episodes=list(self.episodes[-200:]),
                                   metrics=list(self.metrics[-200:]),
                                   evaluations=list(self.evaluations[-200:]))
            return state

    def publish(self, force=False, plot=True):
        now = time.monotonic()
        snapshot_due = force or now - self.last_publish >= 1
        plot_due = plot and (force or now - self.last_plot >= self.plot_seconds)
        if not snapshot_due and not plot_due:
            return
        with self.write_lock:
            state = self.snapshot()
            if snapshot_due:
                atomic_json(self.run_dir / 'progress.json', state)
                self.last_publish = now
            if plot_due:
                self._plot(state)
                self.last_plot = now
            if not self.interactive and now - self.last_log >= 10:
                self.console.print(f"[{state['phase']}] seed={state['seed']} "
                    f"steps={state['steps']:,} episodes={state['episodes']} "
                    f"elapsed={duration(state['elapsed_seconds'])} "
                    f"ETA(train)={duration(state['eta_seconds'])} "
                    f"speed={state['steps_per_second']:.0f} steps/s", markup=False)
                self.last_log = now

    def _plot(self, state):
        # No pyplot/global figure registry: safe alongside the notebook's final graphs.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(12, 7), layout='constrained', facecolor='#101827')
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 2)
        curves = state['curves']
        series = [
            (curves['episodes'], 'timestep', 'return_', 'Training return', '#38bdf8'),
            (curves['evaluations'], 'timestep', 'mean_return', 'Evaluation return', '#34d399'),
            (curves['metrics'], 'timestep', 'train/critic_loss', 'Critic loss', '#fbbf24'),
            (curves['metrics'], 'timestep', 'train/ent_coef', 'Entropy coefficient', '#c084fc'),
        ]
        for ax, (rows, xkey, ykey, title, color) in zip(axes.flat, series):
            rows = [r for r in rows if ykey in r]
            ax.set_facecolor('#172338')
            ax.plot([r[xkey] for r in rows], [r[ykey] for r in rows], color=color, linewidth=1.8)
            ax.set_title(title, color='white')
            ax.set_xlabel('Environment steps', color='#a8b4c8')
            ax.tick_params(colors='#a8b4c8')
            ax.grid(alpha=0.15)
            for spine in ax.spines.values():
                spine.set_color('#344256')
        fig.suptitle(f"SAC × Ant-v5 • {self.mode} • seed {state['seed']} • {state['phase']}\n"
                     f"Elapsed {duration(state['elapsed_seconds'])}  |  "
                     f"ETA (training) {duration(state['eta_seconds'])}", color='white')
        temporary = self.run_dir / 'live_curves.tmp.png'
        fig.savefig(temporary, dpi=110, facecolor=fig.get_facecolor())
        os.replace(temporary, self.run_dir / 'live_curves.png')
        fig.clear()

    def render(self):
        return render_snapshot(self.snapshot())

    def finish(self, status, error=None):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        with self.lock:
            self.state.update(status=status, phase=status, error=error)
            if self.training_started and self.training_stopped is None:
                self.training_stopped = time.monotonic()
        if self.writer:
            self.writer.flush()
            self.writer.close()
            self.writer = None
        self.publish(force=True)
        if self.live:
            self.live.update(self.render(), refresh=True)
            self.live.stop()
        self.event(f'Run {status}: {self.run_dir}')


def render_snapshot(state):
    table = Table.grid(expand=True, padding=(0, 2))
    table.add_column(style='bold cyan', ratio=1)
    table.add_column(ratio=2)
    table.add_column(style='bold cyan', ratio=1)
    table.add_column(ratio=2)
    episode_target = state.get('episode_budget') or '—'
    table.add_row('Phase', str(state['phase']), 'Seed',
                  f"{state['seed']} ({state['seed_index']}/{state['seed_count']})")
    table.add_row('Elapsed', duration(state['elapsed_seconds']), 'ETA (train)',
                  duration(state['eta_seconds']))
    table.add_row('Steps', f"{state['steps']:,} / {state['step_budget']:,}", 'Episodes',
                  f"{state['episodes']} / {episode_target}")
    table.add_row('Speed', f"{state['steps_per_second']:.1f} steps/s", 'Updates',
                  f"{state['updates']:,}")
    for label, key in [('Reward (last 10)', 'reward_mean_10'), ('Evaluation', 'latest_evaluation')]:
        value = state.get(key)
        table.add_row(label, '—' if value is None else f'{value:.2f}', '', '')
    if state.get('phase_total'):
        table.add_row('Phase episodes', f"{state.get('phase_current', 0)} / {state['phase_total']}", '', '')
    curves = state['curves']
    graph = Table.grid(expand=True, padding=(0, 1))
    graph.add_column(style='bold', width=15)
    graph.add_column()
    for label, rows, key, color in [
        ('Reward', curves['episodes'], 'return_', 'cyan'),
        ('Evaluation', curves['evaluations'], 'mean_return', 'green'),
        ('Critic loss', curves['metrics'], 'train/critic_loss', 'yellow'),
        ('Actor loss', curves['metrics'], 'train/actor_loss', 'magenta'),
    ]:
        values = [r[key] for r in rows if key in r]
        extent = f'  [{min(values):.1f} … {max(values):.1f}]' if values else ''
        graph.add_row(label, Text(sparkline(values) + extent, style=color))
    progress = Table.grid(expand=True, padding=(0, 1))
    progress.add_column(ratio=1)
    progress.add_column(width=7)
    progress.add_row(ProgressBar(total=100, completed=state['progress'] * 100),
                     Text(f"{state['progress']:.1%}", style='bold green'))
    content = Group(table, progress,
                    Text('ETA estimates training only; evaluations included in speed', style='dim'),
                    Panel(graph, title='Live trends · steps →', border_style='blue'),
                    Text(state['run_dir'], style='dim'))
    return Panel(content, title=f"SAC × Ant-v5  |  {state['mode']}  |  {state['status']}",
                 border_style='green' if state['status'] == 'completed' else 'cyan')
