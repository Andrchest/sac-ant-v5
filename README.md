# SAC × Ant-v5

## Start here

Team 01 studied SAC locomotion in Ant-v5, ran exploratory ablations, and
compared SAC with TD3 and PPO under a fixed follow-up protocol.

- **Notebook:** [`sac_ant_v5.ipynb`](sac_ant_v5.ipynb), including a standalone toy
  critic update near the top. Older training outputs are from the smoke run.
- The project report covers the full SAC run, exploratory ablations, and the
  follow-up comparison. [`REPORT.md`](REPORT.md)
- **Follow-up comparison:** [protocol](EXPERIMENT_PROTOCOL.md),
  [analysis](research/followup/analysis.md), and [per-run data](research/followup/runs/).
- The repository's `videos/` folder contains six recorded rollouts and notes on their policies and reset seeds.
- Model checkpoints and replay buffers are excluded from this Git checkout.
  The release page is the download location for models and archives. [GitHub Releases](https://github.com/Andrchest/sac-ant-v5/releases)
- **Training entry points:** the notebook and `train_sac.py` for the original SAC
  experiment. Use `run_experiment.py` for individual SAC/TD3/PPO runs and
  `followup_campaign.py` for the matched comparison.

Post-training evaluation, analyses, and plots are under
[`additional analysis/`](additional%20analysis/). Earlier training campaigns,
video tools, and release builders are under [`deprecated/`](deprecated/).

The **Worked example: one SAC critic update** cell requires only Python. Running
the complete notebook starts smoke training. The release bundle's
`BUNDLE_README.md` documents running its `demo_policy.py` with the selected checkpoint.

The follow-up uses three independent training seeds per algorithm. Its results
are descriptive for fixed configurations, not proof of a general algorithm ranking.
The selected seed-50 hero model is separate from that comparison.

## Setup

The recorded training environment used Linux and Python 3.10.19. The commands
below assume Linux and run from the repository root. Training and rendering
default to the MuJoCo EGL backend.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

`requirements-monitor.txt` lists add-ons for an existing experiment environment.
The notebook runner executes the experiment cells without a notebook kernel. Monitoring,
checkpoint controls, and the live viewer are in [`support/`](support/).

## Background execution with a live dashboard

1. Start a short check in a separate tmux session:

   ```bash
   # From the repository root:
   tmux new-session -d -s ant-sac '.venv/bin/python -u train_sac.py'
   ```

   The default smoke run trains for **20 episodes**, with a cap of 20,000 steps.

2. Open the live dashboard:

   ```bash
   tmux attach -t ant-sac
   ```

   The dashboard shows elapsed time, approximate training ETA, steps, episodes, steps/s, gradient updates, mean reward and miniature reward/evaluation/actor/critic-loss plots. `Ctrl+B`, then `D` detaches the terminal **without stopping training**. `Ctrl+C` during training stops it; snapshots and saved checkpoints remain. There is no automatic resumption of an unfinished episode.

3. Alternatively, connect an independent read-only viewer:

   ```bash
   .venv/bin/python -m support.watch_sac
   ```

   Run the viewer from the repository root. `Ctrl+C` closes the viewer while training continues. `--once` prints one snapshot. To select a run, pass its directory with `.venv/bin/python -m support.watch_sac artifacts/ant_sac_smoke_...`.

ETA applies to **the current training seed**. It includes intermediate evaluation and plotting overhead, but excludes the baseline, final evaluation, saving and video. Smoke estimates use completed episodes; full estimates use steps. Episode lengths are random, so ETA can fluctuate considerably. Until the first episode finishes, it shows `estimating…`. During baseline/evaluation, the dashboard continues counting elapsed time and shows the current phase. The viewer warns about snapshots older than 10 seconds: the process may have crashed.

## Detailed interactive plots

1. Start TensorBoard as a separate process:

   ```bash
   tmux new-session -d -s ant-tensorboard '.venv/bin/python -m tensorboard.main --logdir artifacts --host 127.0.0.1 --port 6006'
   ```

2. Open **http://127.0.0.1:6006** in a browser.

The server binds to localhost only. For training on another machine, use SSH forwarding (`ssh -L 6006:127.0.0.1:6006 USER@HOST`), rather than exposing the port on all interfaces. TensorBoard provides reward, episode length, evaluation return, actor/critic losses, entropy coefficient and gradient-update counts; new data is flushed roughly every 3 seconds.

## Notebook plots without launching training

1. Open `sac_ant_v5.ipynb`.
2. In the upper background-training/live-plots cell, set `VIEW_BACKGROUND = True`.
3. Execute **only that cell**.

`RUN_TO_VIEW = None` selects the latest background run; an exact path can be supplied. `AUTO_REFRESH = True` refreshes the PNG and counters every 2 seconds. Interrupting the viewer cell does not affect the external process. Do not click Run All just to view results: the remaining cells start separate training.

## Outputs from `train_sac.py`

Each run gets a separate `artifacts/ant_sac_MODE_TIMESTAMP/` directory:

- `progress.json` — atomic live snapshot, status/phase, elapsed/ETA/speed and recent curve points;
- `live_curves.png` — refreshed plot, approximately every 5 seconds and during evaluation;
- `seed_N/live_episodes.csv`, `live_metrics.csv`, `live_evaluations.csv` — metrics as they arrive;
- `tensorboard/seed_N/events...` — interactive plots;
- `events.log` — monitoring events;
- standard experiment artifacts: config, best/final checkpoint, replay buffer, CSV, final PNG, MP4 and summary.

`artifacts/latest_run.json` points to the latest **started**, not necessarily successfully completed, process. For parallel runs, give the viewer an exact directory. Earlier experiments are not overwritten. PNG/JSON files are replaced atomically so the viewer cannot read a partially written file. Without a TTY, periodic text lines replace ANSI animation: execution through `nohup` and log reading are supported; all PNG/CSV/TensorBoard outputs continue updating.

## Training budgets

```bash
# Still a short check, but with 40 episodes:
.venv/bin/python train_sac.py --episodes 40

# Full network configuration limited to 50,000 steps:
.venv/bin/python train_sac.py --mode full --steps 50000

# New notebook runs with 1 million steps per seed
.venv/bin/python train_sac.py --mode full --seeds 42 43 44

# Single run through the comparison script
.venv/bin/python run_experiment.py --algo sac --seed 42 --steps 50000
```

Full requires an explicit `--mode full`; its warm-up is 10k steps and evaluation occurs every 50k. Smoke and full use different network sizes and are not a single continuing experiment. A high return from standing still does not mean Ant learned forward locomotion: inspect x-progress and video too.

`run_experiment.py` writes each run to `artifacts/xp_*`. Add `--save-replay` to save its replay buffer.

## Overnight execution on Linux

```bash
deadline="$(date -Iseconds -d '+12 hours')"
.venv/bin/python -u -m deprecated.overnight_sac --until "$deadline"
```

The supervisor first runs **1 million steps**. After saving the model, replay, tables and video, it checks 10 final episodes. The operational adequate-gait heuristic is mean return ≥2500, mean x-progress ≥20 m, mean length ≥900, falls ≤20%. This is neither statistical proof nor an algorithm comparison. If the goal is not reached, it continues **the same model, optimizer and replay** in chunks of up to 250k steps. The next chunk is reduced according to measured speed and time remaining; the overall safety cap is 3 million steps per night.

The training cutoff is three minutes before the deadline to allow final evaluation, saving, and video. If the process is still running, the supervisor escalates through SIGINT, SIGTERM, and SIGKILL near the deadline. A run that saves a partial budget records `budget_completed: false` in `summary.json` and `stop_reason: "deadline"` in its per-seed record.

Every 100k steps a complete model+replay+metadata bundle is saved atomically; the two latest bundles are retained. Final, best and earlier experiments are not deleted. `Ctrl+C` also attempts to save an emergency bundle. A full replay takes about **1.7 GB**, so sufficient RAM and disk space are needed.

The supervisor writes its decisions and reports to `artifacts/night_TIMESTAMP/`, including `status.json`, `report.md`, and per-round logs. `artifacts/night_latest.json` records the latest session path.

Manually continue a full checkpoint for another 250k steps:

```bash
.venv/bin/python train_sac.py --mode full --steps 250000 --resume artifacts/ant_sac_full_TIMESTAMP/seed_42/sac_final.zip
```

You can also supply `.../checkpoints/step_N/model.zip` if a complete `replay_buffer.pkl` is beside it. A smoke model is not accepted as full. Continuation does not reset the global step counter; ETA and speed describe only the new chunk.

## Hardening campaign: multiple seeds, ablations and baselines

The archived scripts use the notebook's environment and reward settings. Training runs in separate processes, and evaluation budgets vary by experiment.

- `run_experiment.py` — one experiment: SAC/PPO/TD3 with chosen hyperparameters, steps and seed.
- `deprecated/campaign_experiments.py` — experiment definitions: base, ablations and baselines.
- `deprecated/ablation_campaign.py` — priority-based orchestrator with N workers and a deadline.
- `additional analysis/campaign_analysis.py` — campaign results summarized as Markdown tables.
- `additional analysis/campaign_plots.py` — plots of seeds, convergence, ablations and baselines.
- `deprecated/best_config_campaign.sh` — selected-configuration training: three seeds in parallel.
- `additional analysis/best_config_plots.py` — selected-configuration versus base plots.
- `deprecated/make_rollout.py` — record a rollout MP4 from a checkpoint; inference only.
- `additional analysis/best_vs_base_stats.py` — descriptive training-seed means.
- `additional analysis/stochastic_eval.py` — stochastic evaluation of model robustness.
- `additional analysis/compare_hero_checkpoints.py` — paired evaluation of two saved models.

Launch a new campaign on Linux with a fresh output directory and a future deadline.

```bash
deadline="$(date -Iseconds -d '+24 hours')"
campaign_dir="artifacts/campaign_$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python -u -m deprecated.ablation_campaign --workers 3 --threads 4 \
  --deadline "$deadline" --baseline-seconds-per-million 6500 --output "$campaign_dir"
```

The launcher writes `campaign_summary.json` and `campaign_report.md` to the chosen directory. The analysis script generates `analysis.md` separately. The historical results and their evaluation protocols are documented in the [project report](REPORT.md).

Generate the analysis from the research data included in this repository:

```bash
.venv/bin/python "additional analysis/campaign_analysis.py" --campaign research/campaign --out artifacts/campaign/analysis.md
```

## Best-config agent (2026-11-13)

The ablations informed the selected configuration of **two 256-unit layers, lr=1e-3, batch size 128, and 1M steps**. `deprecated/best_config_campaign.sh` trained seeds 50, 51, and 52.

| seed | return | std | falls |
|---|---|---|---|
| **50 (hero)** | **5818** | 148 | 0% |
| 51 | 5362 | 701 | 3% |
| 52 | 5023 | 1256 | 3% |

The selected configuration averaged **5401** across these seeds. The base runs averaged **3853** and used different training seeds and evaluation protocols. The release bundle uses seed 50 for its inference demo. [Per-run summaries](research/campaign/best/)

## Follow-up: matched algorithm comparison

`followup_campaign.py` launches the matched runs on Linux with at least 9 GiB of available memory. It uses up to three CPU workers with four Torch threads each. Every launch creates a new status directory and separate run directories. [Experiment protocol](EXPERIMENT_PROTOCOL.md)

```bash
.venv/bin/python followup_campaign.py
```

Mean returns across three training seeds were **4012.4 for SAC, 3308.7 for TD3, and 948.3 for PPO**. TD3 used Gaussian exploration noise with standard deviation 0.1. These are descriptive results for the tested configurations in Ant-v5. [Follow-up results](research/followup/analysis.md)

Reproduce the report and plot from the recorded follow-up data:

```bash
.venv/bin/python "additional analysis/followup_analysis.py" --status research/followup/status.json --out artifacts/followup_analysis
```

## Checks

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python train_sac.py --help
.venv/bin/python -m support.watch_sac --help
.venv/bin/python -m py_compile run_experiment.py
.venv/bin/python -m compileall -q deprecated "additional analysis"
```
