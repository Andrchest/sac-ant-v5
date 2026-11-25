"""Build and verify an explicit allowlist of downloadable experiment files."""
from pathlib import Path
import argparse
import hashlib
from importlib.metadata import version
import json
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--submission', action='store_true', help='write the Moodle submission archive')
parser.add_argument('--compact', action='store_true',
                    help='omit intermediate plots, TensorBoard events and duplicate loss logs')
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(ROOT))
from deprecated.ablation_campaign import write_markdown

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


for name in ['.gitignore', 'README.md', 'REPORT.md', 'requirements-monitor.txt', 'sac_ant_v5.ipynb',
             'train_sac.py', 'support/__init__.py', 'support/sac_monitor.py',
             'support/training_control.py', 'support/watch_sac.py',
             'tests/test_monitor.py', 'tests/test_overnight.py',
             'tests/test_campaign_regressions.py', 'tests/test_followup_campaign.py',
             'tests/test_followup_analysis.py', 'EXPERIMENT_PROTOCOL.md', 'followup_campaign.py',
             'run_experiment.py', 'tests/test_submission.py', 'tests/test_research_export.py']:
    include(ROOT / name, name)

for name in ['__init__.py', 'overnight_sac.py', 'campaign_experiments.py', 'ablation_campaign.py',
             'best_config_campaign.sh', 'make_rollout.py']:
    relative = f'deprecated/{name}'
    include(ROOT / relative, relative)

for name in ['followup_analysis.py', 'campaign_analysis.py', 'campaign_plots.py', 'best_config_plots.py',
             'best_vs_base_stats.py', 'compare_hero_checkpoints.py', 'stochastic_eval.py',
             'export_research_data.py']:
    relative = f'additional analysis/{name}'
    include(ROOT / relative, relative)

result_names = ['summary.json', 'config.json', 'versions.txt', 'learning_curves.png',
    'evaluation_all.csv', 'paired_returns.csv',
    'unseen_reset_validation.json', 'unseen_reset_validation.csv',
    'seed_42/train_episodes.csv', 'seed_42/evaluation.csv',
    'seed_42/train.monitor.csv', 'seed_42/eval/evaluations.npz']
if not args.compact:
    result_names += ['live_curves.png', 'seed_42/losses.csv']
for name in result_names:
    include(RUN / name, f'artifacts/{RUN.name}/{name}')
# Hero model = best config (seed 50); base-config model/video stay on the website only.
hero_names = ['sac_final.zip', 'rollout.mp4', 'train_episodes.csv', 'hero_summary.json']
if not args.compact:
    hero_names += ['losses.csv']
for name in hero_names:
    include(RUN / 'best_config/seed_50' / name, f'artifacts/{RUN.name}/best_config/seed_50/{name}')
include(RUN / 'best_config/seed_50/eval/evaluations.npz', f'artifacts/{RUN.name}/best_config/seed_50/eval/evaluations.npz')
if not args.compact:
    for path in sorted((RUN / 'tensorboard').rglob('events.out.tfevents.*')):
        include(path, f'artifacts/{RUN.name}/{path.relative_to(RUN).as_posix()}')
hardening_names = ['seeds.png', 'convergence.png', 'ablation.png', 'baselines.png',
                   'best_config.png', 'best_vs_base.png', 'followup.png']
for name in hardening_names:
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
for directory in ('research/followup', 'research/campaign'):
    for path in sorted((ROOT / directory).rglob('*')):
        if path.is_file():
            include(path, path.relative_to(ROOT).as_posix())
# Rebuild the reports with the English renderers and recorded data.
with TemporaryDirectory(prefix='sac-bundle-reports-') as temporary_reports:
    report_dir = Path(temporary_reports)
    campaign_report = report_dir / 'campaign_report.md'
    write_markdown(campaign_report, meta)
    include(campaign_report, 'artifacts/campaign/campaign_report.md')

    analysis = report_dir / 'analysis.md'
    analysis_dir = ROOT / 'additional analysis'
    subprocess.run([sys.executable, '-X', 'utf8', str(analysis_dir / 'campaign_analysis.py'),
                    '--campaign', 'artifacts/campaign', '--out', str(analysis)],
                   cwd=ROOT, stdout=subprocess.DEVNULL, check=True)
    include(analysis, 'artifacts/campaign/analysis.md')
    include(analysis, 'HARDENING_ANALYSIS.md')

include(ROOT / 'REPORT.md', 'OVERNIGHT_REPORT.md')

packages = ['gymnasium', 'stable-baselines3', 'mujoco', 'torch', 'numpy', 'pandas',
            'matplotlib', 'imageio', 'imageio-ffmpeg', 'pillow', 'rich', 'tensorboard',
            'nbformat', 'nbclient', 'ipykernel', 'pytest']
files['requirements.txt'] = ('# Exact versions recorded on Linux / Python 3.10.19.\n' +
    '# For CPU-only PyTorch, install the matching torch wheel first (see BUNDLE_README.md).\n' +
    '\n'.join(f'{name}=={version(name)}' for name in packages) + '\n').encode()

contents = ('rollout video and TensorBoard event data' if not args.compact else
            'recorded demo video')
tb_note = ('TensorBoard can view the saved events: '
           '`python -m tensorboard.main --logdir artifacts --host 127.0.0.1 --port 6006`.'
           if not args.compact else
           'TensorBoard event logs are not included in this compact archive; the plotted '
           'curves are.')
files['BUNDLE_README.md'] = (f'''# SAC × Ant-v5 — code, model and results

This bundle contains the experiment source, notebook, final trained model, plots,
CSV tables and {contents}. Training finished at
**1,000,000 steps / 990,000 gradient updates** with training seed 42.

## View existing results without running training

- Open `artifacts/{RUN.name}/learning_curves.png`.
- Open `artifacts/{RUN.name}/best_config/seed_50/rollout.mp4` in a video player.
- Read `OVERNIGHT_REPORT.md` for the 10-episode final evaluation, 30 fresh-reset checks,
  and the full hardening campaign (multi-seed, convergence, ablations, baselines).
- `HARDENING_ANALYSIS.md` and `artifacts/{RUN.name}/hardening/*.png` (seeds, convergence,
  ablation, baselines, follow-up) summarize the campaign.
- The notebook's saved outputs are from the initial smoke test, not the full run.
  The complete full-run results are in the artifacts directory above.''' + ('' if not args.compact else '''

The compact archive includes the final evaluation outputs, trained model, rollout
video and campaign plots. The full archive also includes intermediate `live_curves.png`,
duplicate loss CSVs and TensorBoard event logs.''') + f'''

## Install (tested platform: Linux, Python 3.10.19)

1. Create a virtual environment: `python -m venv .venv`.
2. Activate it: `source .venv/bin/activate` (use the equivalent activation command on Windows).
3. For CPU-only PyTorch install: `python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu`.
4. Install the remaining pinned packages: `python -m pip install -r requirements.txt`.

Dependencies or graphics drivers may need platform-specific adjustments. No `.venv`,
credentials, caches or Git history are included. {tb_note}

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
Run `python -m support.watch_sac` to view a live experiment or pass an explicit run directory.

## What is intentionally not included

The ~1.7 GB replay buffer, duplicate best/periodic checkpoints, and local environments
are omitted to keep this download small. The final model alone is sufficient for inference.
Exact continuation requires the original checkpoint and its matching replay buffer.

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

The ten campaign jobs include a 2M-step SAC seed-43 run, a 1M-step SAC seed-44
run, six 500k-step ablations, and two baselines. The original SAC seed-42 run is
the reference. At 1M steps, the three SAC evaluations ranged from 2571 to 5701
with a mean of about 3853. Seed 43 was evaluated at a checkpoint and episode
counts varied. TD3 and PPO each used one training seed. TD3 used no action
exploration noise. PPO completed 1,001,472 steps in 2,048-step rollouts on a
requested 1M-step budget. Run `python "additional analysis/campaign_analysis.py"`
for the analysis and `python "additional analysis/campaign_plots.py"` for the plots.

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

## Fixed-protocol follow-up (2026-11-19/20)

On three paired training seeds (60/61/62) at 1M environment steps, mean evaluation
returns across trained seeds were SAC 4012, TD3 3309 and PPO 948. TD3 used normal
exploration noise σ=0.1. These are descriptive results for specific configurations;
three training seeds and one environment do not establish general superiority.
The single-seed 500k SAC base control scored 1584 ± 1183 (7/10 falls). Full protocol,
per-episode CSVs, provenance and training-seed analysis are in `EXPERIMENT_PROTOCOL.md`
and `research/followup/`; see `OVERNIGHT_REPORT.md` section 7.
''').encode()

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
    source_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
    excluded=['replay buffer', 'duplicate best/periodic models', 'local environments', 'credentials'],
    sha256={name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())})
files['MANIFEST.json'] = json.dumps(manifest, indent=2).encode()
archive_name = 'sac_ant_v5_submission.zip' if args.submission else 'sac_ant_v5_bundle.zip'
if args.compact:
    archive_name = archive_name.replace('.zip', '_compact.zip')
archive = RUN / archive_name
temporary = archive.with_suffix('.zip.tmp')
with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
    for name, data in sorted(files.items()):
        info = zipfile.ZipInfo(PREFIX + name)
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        output.writestr(info, data)
assert temporary.stat().st_size < 10_000_000, 'Bundle exceeded 10 MB safety limit'
with zipfile.ZipFile(temporary) as check:
    assert check.testzip() is None
    assert len(check.namelist()) == len(files)
    assert PREFIX + 'sac_ant_v5.ipynb' in check.namelist()
    assert PREFIX + f'artifacts/{RUN.name}/best_config/seed_50/sac_final.zip' in check.namelist()
    assert PREFIX + f'artifacts/{RUN.name}/best_config/seed_50/rollout.mp4' in check.namelist()
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
