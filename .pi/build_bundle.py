"""Build and verify an explicit allowlist of downloadable experiment files."""
from pathlib import Path
import hashlib
from importlib.metadata import version
import json
import os
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(json.loads((ROOT / 'artifacts/latest_run.json').read_text())['run_dir'])
assert RUN.name == 'ant_sac_full_20261021T194131399885Z'
PREFIX = 'sac-ant-v5/'
files = {}


def include(path, relative):
    path = Path(path)
    assert path.is_file() and not path.is_symlink(), path
    assert not any(part in {'.env', '.git', '.venv', '.pi', '__pycache__'} for part in Path(relative).parts)
    assert 'replay_buffer' not in relative
    files[relative] = path.read_bytes()


for name in ['.gitignore', 'README.md', 'requirements-monitor.txt', 'sac_ant_v5.ipynb',
             'sac_monitor.py', 'train_sac.py', 'watch_sac.py', 'training_control.py',
             'overnight_sac.py', 'tests/test_monitor.py', 'tests/test_overnight.py',
             'tests/test_campaign_regressions.py', 'tests/test_followup_campaign.py',
             'EXPERIMENT_PROTOCOL.md', 'followup_campaign.py',
             'run_experiment.py', 'campaign_experiments.py', 'ablation_campaign.py',
             'campaign_analysis.py', 'campaign_plots.py', 'best_config_plots.py',
             'best_vs_base_stats.py', 'stochastic_eval.py', 'compare_hero_checkpoints.py', 'make_slide.py']:
    include(ROOT / name, name)

result_names = ['summary.json', 'config.json', 'versions.txt', 'learning_curves.png',
    'live_curves.png', 'evaluation_all.csv', 'paired_returns.csv',
    'unseen_reset_validation.json', 'unseen_reset_validation.csv',
    'seed_42/train_episodes.csv', 'seed_42/losses.csv', 'seed_42/evaluation.csv',
    'seed_42/train.monitor.csv', 'seed_42/eval/evaluations.npz']
for name in result_names:
    include(RUN / name, f'artifacts/{RUN.name}/{name}')
# Hero model = best config (seed 50); base-config model/video stay on the website only.
for name in ['sac_final.zip', 'rollout.mp4', 'train_episodes.csv', 'losses.csv', 'hero_summary.json']:
    include(RUN / 'best_config/seed_50' / name, f'artifacts/{RUN.name}/best_config/seed_50/{name}')
include(RUN / 'best_config/seed_50/eval/evaluations.npz', f'artifacts/{RUN.name}/best_config/seed_50/eval/evaluations.npz')
for path in sorted((RUN / 'tensorboard').rglob('events.out.tfevents.*')):
    include(path, f'artifacts/{RUN.name}/{path.relative_to(RUN).as_posix()}')
for name in ['seeds.png', 'convergence.png', 'ablation.png', 'baselines.png',
             'best_config.png', 'best_vs_base.png', 'results_slide.png']:
    include(RUN / 'hardening' / name, f'artifacts/{RUN.name}/hardening/{name}')
# Include small primary data, not checkpoints or replay buffers. Rewrite the
# campaign index so analysis works from the extracted bundle's project root.
meta = json.loads((ROOT / 'artifacts/campaign/campaign_summary.json').read_text())
for exp in meta['experiments']:
    if exp.get('status') != 'done':
        continue
    source = Path(exp['run_dir'])
    relative = Path('artifacts/campaign/runs') / exp['id']
    for name in ('summary.json', 'evaluation_all.csv'):
        include(source / name, (relative / name).as_posix())
    exp['run_dir'] = relative.as_posix()
files['artifacts/campaign/campaign_summary.json'] = (json.dumps(meta, indent=2) + '\n').encode()
for seed in (50, 51, 52):
    matches = sorted((ROOT / 'artifacts').glob(f'xp_sac_{seed}_*best{seed}/evaluation_all.csv'))
    assert len(matches) == 1, f'ambiguous best-config source for seed {seed}: {matches}'
    include(matches[0], f'artifacts/campaign/best/seed_{seed}/evaluation_all.csv')
    include(matches[0].parent / 'summary.json', f'artifacts/campaign/best/seed_{seed}/summary.json')
for name in ('hero_checkpoint_comparison.csv', 'hero_checkpoint_comparison.json'):
    include(ROOT / 'artifacts/campaign' / name, f'artifacts/campaign/{name}')
include(ROOT / 'artifacts/campaign/analysis.md', 'HARDENING_ANALYSIS.md')
night = Path(json.loads((ROOT / 'artifacts/night_latest.json').read_text())['directory'])
include(night / 'report.md', 'OVERNIGHT_REPORT.md')

packages = ['gymnasium', 'stable-baselines3', 'mujoco', 'torch', 'numpy', 'pandas',
            'matplotlib', 'imageio', 'imageio-ffmpeg', 'rich', 'tensorboard',
            'nbformat', 'nbclient', 'ipykernel', 'pytest']
files['requirements.txt'] = ('# Exact versions recorded on Linux / Python 3.10.19.\n' +
    '# For CPU-only PyTorch, install the matching torch wheel first (see BUNDLE_README.md).\n' +
    '\n'.join(f'{name}=={version(name)}' for name in packages) + '\n').encode()

files['BUNDLE_README.md'] = f'''# SAC × Ant-v5 — code, model and results

This bundle contains the experiment source, notebook, final trained model, plots,
CSV tables, rollout video and TensorBoard event data. Training finished at
**1,000,000 steps / 990,000 gradient updates** with training seed 42.

## View existing results without running training

- Open `artifacts/{RUN.name}/learning_curves.png`.
- Open `artifacts/{RUN.name}/best_config/seed_50/rollout.mp4` in a video player.
- Read `OVERNIGHT_REPORT.md` for the 10-episode final evaluation, 30 fresh-reset checks,
  and the full hardening campaign (multi-seed, convergence, ablations, baselines).
- `HARDENING_ANALYSIS.md` and `artifacts/{RUN.name}/hardening/*.png` (seeds, convergence,
  ablation, baselines, and a ready-to-paste `results_slide.png`) summarize the campaign.
- The notebook's saved outputs are from the initial smoke test, not the full run.
  The complete full-run results are in the artifacts directory above.

## Install (tested platform: Linux, Python 3.10.19)

1. Create a virtual environment: `python -m venv .venv`.
2. Activate it: `source .venv/bin/activate` (use the equivalent activation command on Windows).
3. For CPU-only PyTorch install: `python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu`.
4. Install the remaining pinned packages: `python -m pip install -r requirements.txt`.

Dependencies or graphics drivers may need platform-specific adjustments. No `.venv`,
credentials, caches or Git history are included. TensorBoard can view the saved events:
`python -m tensorboard.main --logdir artifacts --host 127.0.0.1 --port 6006`.

## Run the trained model (no learning)

`python demo_policy.py --episodes 3`

This loads the **best-config hero checkpoint** (seed 50, lr=1e-3, batch=128, 1M steps)
on CPU and prints episode reward and forward progress. It does not call learn() or need
the replay buffer. Loading uses a one-slot empty replay buffer to avoid allocating
training-sized storage during inference. The original base-config model (seed 42) is
available on the results website if you want to compare.

## Training is explicit

`python train_sac.py` starts a NEW 20-episode smoke experiment. It does not silently
continue the downloaded full model. Full training requires explicit `--mode full`.
The separate live viewer expects a running experiment or an explicit run directory.

## What is intentionally not included

The ~1.7 GB replay buffer, duplicate best/periodic checkpoints, and local environments
are omitted to keep this download small. The final model alone is sufficient for inference.
For exact continuation use the separately downloadable replay buffer beside the original
checkpoint; do not claim that a fresh empty replay is the original continued experiment.

Checkpoints/pickle files must come from a trusted source. Local paths in config.json are
recorded provenance from the original machine, not required paths on your computer.

## Integrity

`MANIFEST.json` contains SHA-256 digests for all bundled content files. A separate
`.zip.sha256` file on the download page verifies the archive itself. Critic loss is the
combined loss of both Q critics; separate historical Q1/Q2 losses were not recorded.

The original base-config policy (training seed 42) was tested on 30 fresh reset seeds:
three of those 30 episodes ended in falls. These are not 30 independent trainings and
are not the hero seed 50 result.

## Hardening campaign (added 2026-11-12) and best-config agent (2026-11-13)

A 10-experiment campaign (seeds 42/43/44 at 1M, a 2M convergence run, six hyperparameter
ablations at 500k, and PPO/TD3 baselines at 1M) is included. Key honest findings:
the original 5701 is the top of the observed 2571-5701 range (mean ~3853). These are
three SAC training seeds with mixed evaluation protocols, not a general performance range.
PPO and TD3 use one seed; TD3 has no action exploration noise. No algorithm superiority is
established. Reproduce the analysis with `campaign_analysis.py` and plots with `campaign_plots.py`.

Applying those levers, a best-config agent (SAC 256 / lr 1e-3 / batch 128 / 1M) was trained
on seeds 50/51/52: mean 5401, range 5023-5818, spread 794, falls 0-3% in these runs.
The base config used different seeds and mixed evaluation protocols. Three training runs
per configuration do not establish a general performance or reliability advantage.
The hero final model is seed 50: 5818 ± 148, 0/30 observed falls. A 1M-step
checkpoint saved before the last gradient update scored 5279 ± 1601, 2/10 falls;
the final model is after one additional update, so these are different weights. The first
10 reset seeds overlap. The old checkpoint and its original per-episode results are not
included; their aggregate is in `hero_summary.json`. A later paired evaluation of both
models on 30 new reset seeds is included in `artifacts/campaign/hero_checkpoint_comparison.csv`.
Zero observed falls does not establish zero fall probability. See `HARDENING_ANALYSIS.md`,
`OVERNIGHT_REPORT.md` (section 5), and `hardening/best_*.png`.
'''.encode()

files['demo_policy.py'] = f'''"""Run the downloaded SAC checkpoint without training or replay-buffer allocation."""
from pathlib import Path
import argparse
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import SAC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=3)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    torch.set_num_threads(2)
    checkpoint = Path(__file__).resolve().parent / 'artifacts/{RUN.name}/best_config/seed_50/sac_final.zip'
    model = SAC.load(checkpoint, device='cpu', buffer_size=1)
    original_steps = model.num_timesteps
    env = gym.make('Ant-v5', max_episode_steps=1000,
        include_cfrc_ext_in_observation=True, exclude_current_positions_from_observation=True,
        healthy_reward=1.0, forward_reward_weight=1.0, ctrl_cost_weight=0.5,
        contact_cost_weight=0.0005, healthy_z_range=(0.2, 1.0), terminate_when_unhealthy=True)
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=900042 + episode)
            x0, total = float(env.unwrapped.data.qpos[0]), 0.0
            for length in range(1, 1001):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                assert np.isfinite(obs).all() and np.isfinite(reward)
                total += float(reward)
                if terminated or truncated:
                    break
            progress = float(env.unwrapped.data.qpos[0]) - x0
            print(f'Episode {{episode + 1}}: reward={{total:.2f}}, steps={{length}}, '
                  f'x_progress={{progress:.2f}} m, fell={{terminated}}')
    finally:
        env.close()
    assert model.num_timesteps == original_steps == 1000000
    print('Inference only (best-config hero, seed 50): no training steps performed.')


if __name__ == '__main__':
    main()
'''.encode()
compile(files['demo_policy.py'], 'demo_policy.py', 'exec')
manifest = dict(run=RUN.name, training_steps=1000000,
    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    excluded=['replay buffer', 'duplicate best/periodic models', 'local environments', 'credentials'],
    sha256={name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())})
files['MANIFEST.json'] = json.dumps(manifest, indent=2).encode()
archive = RUN / 'sac_ant_v5_bundle.zip'
temporary = archive.with_suffix('.zip.tmp')
with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
    for name, data in sorted(files.items()):
        output.writestr(PREFIX + name, data)
assert temporary.stat().st_size < 10_000_000, 'Bundle exceeded 10 MB safety limit'
with zipfile.ZipFile(temporary) as check:
    assert check.testzip() is None
    assert len(check.namelist()) == len(files)
    assert PREFIX + 'sac_ant_v5.ipynb' in check.namelist()
    assert PREFIX + f'artifacts/{RUN.name}/best_config/seed_50/sac_final.zip' in check.namelist()
    assert PREFIX + f'artifacts/{RUN.name}/learning_curves.png' in check.namelist()
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256(check.read(PREFIX + name)).hexdigest() == digest
os.replace(temporary, archive)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
checksum = archive.with_suffix('.zip.sha256')
checksum.write_text(f'{digest}  {archive.name}\n')
print('BUNDLE VERIFIED:', archive)
print('Files:', len(files), 'Bytes:', archive.stat().st_size,
      'MiB:', round(archive.stat().st_size / 1024**2, 2))
print('SHA256:', digest)
