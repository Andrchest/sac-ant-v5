# SAC × Ant-v5

## Start here

Team 01 reinforcement-learning project: SAC locomotion in Ant-v5, exploratory
ablations, and a preregistered comparison with TD3 and PPO.

- **Notebook:** [`sac_ant_v5.ipynb`](sac_ant_v5.ipynb), including a standalone toy
  critic update near the top. Older training outputs are from the smoke run.
- **Presentation content:** [`submission/PRESENTATION_DRAFT.md`](submission/PRESENTATION_DRAFT.md).
  This is a draft, not a finished slide deck.
- **Comparison and primary data:** [`research/followup/analysis.md`](research/followup/analysis.md).
- **Downloadable model, videos and submission ZIP:** [GitHub Releases](https://github.com/Andrchest/sac-ant-v5/releases).
  These binary assets are intentionally excluded from Git; replay buffers are not published.
- **Training entry points:** the notebook and `train_sac.py` for the original SAC
  experiment; `run_experiment.py` for later SAC/TD3/PPO experiments.

Use Python 3.10 and a virtual environment; install `requirements.txt`. The toy cell
requires only Python. Running the complete notebook starts smoke training, not
just inference. For model inference, download and extract the English submission
ZIP from Releases and follow its `BUNDLE_README.md`.

The follow-up uses three independent training seeds per algorithm. Its results
are descriptive for fixed configurations, not proof of a general algorithm ranking.
The selected seed-50 hero model is separate from that comparison.

The algorithm and the complete experiment are in **`sac_ant_v5.ipynb`**. `train_sac.py` executes its code cells as a regular Python process, without Jupyter or a second copy of the algorithm. `sac_monitor.py` handles observation and visualization only. Older notebook outputs come from the first smoke test; the viewer shows the current background run.

## Background execution with a live dashboard

1. Start a short check in a separate tmux session:

   ```bash
   cd /home/andreipc/Inno-RL
   tmux new-session -d -s ant-sac '.venv/bin/python -u train_sac.py'
   ```

   Default: **20 training episodes**, not one million steps. No notebook kernel is required. Additional monitoring dependencies are listed in `requirements-monitor.txt`; they are already available in the current environment.

2. Open the live dashboard:

   ```bash
   tmux attach -t ant-sac
   ```

   The dashboard shows elapsed time, approximate training ETA, steps, episodes, steps/s, gradient updates, mean reward and miniature reward/evaluation/actor/critic-loss plots. `Ctrl+B`, then `D` detaches the terminal **without stopping training**. `Ctrl+C` during training stops it; snapshots and saved checkpoints remain. There is no automatic resumption of an unfinished episode.

3. Alternatively, connect an independent read-only viewer:

   ```bash
   .venv/bin/python watch_sac.py
   ```

   Closing the viewer with `Ctrl+C` does not stop training. `--once` prints one snapshot. For a specific run: `.venv/bin/python watch_sac.py artifacts/ant_sac_smoke_...`.

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

## What is saved during execution

Each run gets a separate `artifacts/ant_sac_MODE_TIMESTAMP/` directory:

- `progress.json` — atomic live snapshot, status/phase, elapsed/ETA/speed and recent curve points;
- `live_curves.png` — refreshed plot, approximately every 5 seconds and during evaluation;
- `seed_N/live_episodes.csv`, `live_metrics.csv`, `live_evaluations.csv` — metrics as they arrive;
- `tensorboard/seed_N/events...` — interactive plots;
- `events.log` — monitoring events;
- standard experiment artifacts: config, best/final checkpoint, replay buffer, CSV, final PNG, MP4 and summary.

`artifacts/latest_run.json` points to the latest **started**, not necessarily successfully completed, process. For parallel runs, give the viewer an exact directory. Earlier experiments are not overwritten. PNG/JSON files are replaced atomically so the viewer cannot read a partially written file. Without a TTY, periodic text lines replace ANSI animation: execution through `nohup` and log reading are supported; all PNG/CSV/TensorBoard outputs continue updating.

## Other budgets — only on explicit request

```bash
# Still a short check, but with 40 episodes:
.venv/bin/python train_sac.py --episodes 40

# Full network configuration limited to 50,000 steps:
.venv/bin/python train_sac.py --mode full --steps 50000

# Presentation plan: 1 million steps for each of three seeds (long-running):
.venv/bin/python train_sac.py --mode full --seeds 42 43 44
```

Full requires an explicit `--mode full`; its warm-up is 10k steps and evaluation occurs every 50k. Smoke and full use different network sizes and are not a single continuing experiment. A high return from standing still does not mean Ant learned forward locomotion: inspect x-progress and video too.

## Overnight execution with a deadline and continuation

```bash
# Date and time zone are required; example: an overnight run until 07:00 MSK:
.venv/bin/python -u overnight_sac.py --until 2026-10-22T07:00:00+03:00
```

The supervisor first runs **1 million steps**. After saving the model, replay, tables and video, it checks 10 final episodes. The operational adequate-gait heuristic is mean return ≥2500, mean x-progress ≥20 m, mean length ≥900, falls ≤20%. This is neither statistical proof nor an algorithm comparison. If the goal is not reached, it continues **the same model, optimizer and replay** in chunks of up to 250k steps. The next chunk is reduced according to measured speed and time remaining; the overall safety cap is 3 million steps per night.

Training steps stop 3 minutes before the deadline to allow final evaluation, saving and video. If the process hangs, the supervisor additionally sends SIGINT 90 seconds before the deadline, SIGTERM 30 seconds before it, and SIGKILL at the deadline. With a partial budget, `summary.json` explicitly contains `budget_completed=false` and `stop_reason=deadline`; this is not presented as a completed million.

Every 100k steps a complete model+replay+metadata bundle is saved atomically; the two latest bundles are retained. Final, best and earlier experiments are not deleted. `Ctrl+C` also attempts to save an emergency bundle. A full replay takes about **1.7 GB**, so sufficient RAM and disk space are needed.

Decisions and reports: `artifacts/night_TIMESTAMP/status.json`, `report.md`, `round_001.log`, and subsequent logs. `artifacts/night_latest.json` contains the overnight-session path. These files update independently of the chat; questions or agent wake-ups are unnecessary. On error, the supervisor records the cause and stops instead of hiding it behind endless restarts.

Manually continue a full checkpoint for another 250k steps:

```bash
.venv/bin/python train_sac.py --mode full --steps 250000 --resume artifacts/ant_sac_full_TIMESTAMP/seed_42/sac_final.zip
```

You can also supply `.../checkpoints/step_N/model.zip` if a complete `replay_buffer.pkl` is beside it. A smoke model is not accepted as full. Continuation does not reset the global step counter; ETA and speed describe only the new chunk.

## Hardening campaign: multiple seeds, ablations and baselines

Separate scripts support reproducible experiments (environment/reward/evaluation match the notebook; training runs in separate, parallel processes):

- `run_experiment.py` — one experiment: SAC/PPO/TD3 with chosen hyperparameters, steps and seed.
- `campaign_experiments.py` — experiment definitions: base, ablations and baselines.
- `ablation_campaign.py` — priority-based orchestrator with N workers and a deadline.
- `campaign_analysis.py` — campaign results summarized as Markdown tables.
- `campaign_plots.py` — plots of seeds, convergence, ablations and baselines.
- `best_config_campaign.sh` — selected-configuration training: three seeds in parallel.
- `best_config_plots.py` — selected-configuration versus base plots.
- `make_rollout.py` — record a rollout MP4 from a checkpoint; inference only.
- `best_vs_base_stats.py` — descriptive training-seed means, without invalid pooled-episode inference.
- `stochastic_eval.py` — stochastic (temperature) evaluation of model robustness.
- `make_slide.py` — a presentation-ready 16:9 results slide.

Launch the campaign (example: approximately 16 hours on 12 cores for all 10 experiments):

```bash
.venv/bin/python -u ablation_campaign.py --workers 3 --threads 4 \
  --deadline 2026-11-13T07:00:00+03:00 --baseline-seconds-per-million 6500 \
  --output artifacts/campaign
```

Results: `artifacts/campaign/campaign_summary.json`, `campaign_report.md`, `analysis.md`. Main findings are in `artifacts/night_*/report.md` (Hardening section) and on the results page (`#hardening`). The GPU does not accelerate this task: MuJoCo physics is the bottleneck, and CPU/CUDA speeds are approximately equal. In the observed base runs, 5701 is the top of the 2571–5701 range (mean ≈3853; evaluation protocols differ). One seed each for PPO and TD3 does not establish general SAC superiority; that TD3 run had no action noise.

## Best-config agent (2026-11-13)

Using the levers from the ablations — **network 256, lr=1e-3, batch=128, 1M steps** — three new seeds (50/51/52) were trained through `best_config_campaign.sh`:

| seed | return | std | falls |
|---|---|---|---|
| **50 (hero)** | **5818** | 148 | 0% |
| 51 | 5362 | 701 | 3% |
| 52 | 5023 | 1256 | 3% |

Mean **5401**, range **5023–5818**, spread **794**, versus 3853 / 2571–5701 / 3130 for the original configuration. The weakest seed of the new configuration (5023) exceeds the original mean (3853); falls were 0–3% versus 0–50% in these runs. Training seeds and evaluation protocols differ, with only three training runs per configuration: superiority is not established. **The hero model is seed 50**; it is the default in `demo_policy.py` and in the bundle. Plots: `best_config_plots.py`; video: `make_rollout.py`.

## Follow-up: matched algorithm comparison

The protocol preregistering three shared training seeds (60–62), a 1M budget and TD3 exploration noise is in `EXPERIMENT_PROTOCOL.md`. Running `followup_campaign.py` creates a separate directory with logs and statuses; reruns do not overwrite history. The resulting metrics describe specific configurations, not general algorithm superiority. Earlier campaign data is exported into `research/campaign/` by `export_research_data.py`; the derived index includes SHA-256 source hashes, while the original summary remains unchanged. The `make_slide.py` slide reads these data.

Completed campaign results: `research/followup/analysis.md` and primary CSV/JSON. Means across three training seeds: SAC 4012, TD3 3309, PPO 948. TD3 uses σ=0.1; these are fixed configurations in one environment, not proof of general superiority. SAC seed-42 control at 500k: 1584 ± 1183, 7/10 falls; this is a single-seed observation.

## Checks

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python train_sac.py --help
.venv/bin/python watch_sac.py --help
.venv/bin/python -m py_compile run_experiment.py campaign_experiments.py \
  ablation_campaign.py campaign_analysis.py campaign_plots.py best_config_plots.py \
  make_rollout.py best_vs_base_stats.py stochastic_eval.py make_slide.py
```
