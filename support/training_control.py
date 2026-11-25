"""Deadline and checkpoint operations, independent of the learning algorithm."""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

from .sac_monitor import atomic_json


def parse_stop_at(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('Deadline must include timezone, e.g. 2026-10-22T07:00:00+03:00')
    return parsed.timestamp()


def validate_resume(path, mode, seeds):
    path = Path(path).expanduser().resolve()
    if mode != 'full' or len(seeds) != 1:
        raise ValueError('Resume requires full mode and exactly one seed')
    if not path.is_file():
        raise FileNotFoundError(path)
    seed_dir = path.parent
    # Final checkpoint lives directly under seed_N; a periodic complete bundle
    # lives two levels deeper. Both must retain their original run config.
    config_path = next((parent / 'config.json' for parent in list(path.parents)[:5]
                        if (parent / 'config.json').is_file()), None)
    if config_path is None:
        raise FileNotFoundError(seed_dir.parent / 'config.json')
    config = json.loads(config_path.read_text())
    # Check the actual seed directory, including periodic bundles nested below it.
    seed_parents = [parent for parent in path.parents if parent.name.startswith('seed_')]
    if not seed_parents:
        raise ValueError('Resume checkpoint is not inside a seed directory')
    try:
        checkpoint_seed = int(seed_parents[0].name.removeprefix('seed_'))
    except ValueError as exc:
        raise ValueError('Invalid checkpoint seed directory') from exc
    if config['mode'] != 'full' or checkpoint_seed != seeds[0] or checkpoint_seed not in config['seeds']:
        raise ValueError('Resume requires a full checkpoint of the same training seed, not smoke')
    replay = seed_dir / 'replay_buffer.pkl'
    if not replay.is_file():
        raise FileNotFoundError(replay)
    return path


class TrainingControl:
    def __init__(self, seed_dir, stop_at=None, checkpoint_every=100000, clock=time.time):
        self.seed_dir = Path(seed_dir)
        self.stop_at = stop_at
        self.checkpoint_every = checkpoint_every
        self.clock = clock
        self.next_checkpoint = None
        self.stop_reason = None
        self.last_checkpoint = None

    def on_step(self, model):
        if self.stop_at is not None and self.clock() >= self.stop_at:
            self.stop_reason = 'deadline'
            return False
        step = int(model.num_timesteps)
        if self.next_checkpoint is None:
            # New sessions at zero save at the first boundary; resumed sessions
            # schedule from their current boundary without rewriting an old run.
            self.next_checkpoint = ((max(0, step - 1) // self.checkpoint_every) + 1) * self.checkpoint_every
        if step >= self.next_checkpoint:
            self.save_checkpoint(model)
            self.next_checkpoint = (step // self.checkpoint_every + 1) * self.checkpoint_every
        return True

    def callback(self):
        from stable_baselines3.common.callbacks import BaseCallback
        control = self

        class DeadlineCheckpointCallback(BaseCallback):
            def _on_step(self):
                return control.on_step(self.model)

        return DeadlineCheckpointCallback()

    def save_checkpoint(self, model):
        root = self.seed_dir / 'checkpoints'
        root.mkdir(parents=True, exist_ok=True)
        step = int(model.num_timesteps)
        destination = root / f'step_{step:09d}'
        if destination.exists():
            return destination
        temporary = root / ('.tmp-' + uuid.uuid4().hex)
        temporary.mkdir()
        try:
            model.save(temporary / 'model.zip')
            model.save_replay_buffer(temporary / 'replay_buffer.pkl')
            atomic_json(temporary / 'metadata.json', dict(
                steps=step, gradient_updates=int(model._n_updates),
                saved_at=datetime.now().astimezone().isoformat()))
            os.replace(temporary, destination)
            atomic_json(root / 'latest.json', {'directory': str(destination.resolve()), 'steps': step})
            self.last_checkpoint = str(destination)
            # Only owned, generated checkpoint bundles are rotated; final/best
            # checkpoints and all previous experiments are never removed.
            owned = sorted(p for p in root.glob('step_[0-9]*')
                           if p.is_dir() and not p.is_symlink())
            for stale in owned[:-2]:
                shutil.rmtree(stale)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return destination
