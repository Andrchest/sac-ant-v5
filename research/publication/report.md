# Overnight SAC report

Status: completed
Deadline: 2026-10-22T07:00:00+03:00
Reason: gait_quality_heuristic_met

## Round 1
Requested additional steps: 1000000
Run: /home/andreipc/Inno-RL/artifacts/ant_sac_full_20261021T194131399885Z
Exit code: 0
```json
{
  "mean_return": 5701.037002504314,
  "mean_x_progress": 286.3874996492947,
  "mean_length": 1000.0,
  "fall_rate": 0.0,
  "sufficient": true,
  "note": "Operational gait heuristic on 10 episodes, not a statistical benchmark claim."
}
```

## Independent validation on 30 fresh reset seeds

The same final checkpoint was evaluated without additional training on seeds 900042–900071 (not used for training, periodic evaluation or the original final evaluation).

- Mean return: **5303.18**, sample standard deviation **1156.93**; range **1895.88–5903.54**.
- Mean forward progress: **266.57 m**.
- Mean episode length: **944.13 / 1000**.
- Falls: **3 / 30 (10%)**. This is not a perfectly reliable gait.
- The predefined operational gait threshold is still met. Training remained at **1,000,000 steps**; no unnecessary extension was started.
- Full metrics: `../ant_sac_full_20261021T194131399885Z/unseen_reset_validation.json` and `.csv`.

Artifact checks passed: final/best model ZIP integrity, full replay, CSV counts, PNG, TensorBoard events, and 1001-frame MP4 decoding. The learning curves and a rollout contact sheet were visually inspected.

Only one training seed was run; fresh reset seeds are not independent training seeds. No claim of statistically significant algorithm superiority or perfect convergence.

---

## Hardening campaign (2026-11-12, multi-seed + ablations + baselines)

To address the single-seed limitation above, a 10-experiment campaign was run on 12 CPU cores (the GPU does not accelerate this workload — the MuJoCo physics step is the bottleneck, measured ~equal speed on CPU and CUDA). Full data: `../campaign/analysis.md`, `../campaign/campaign_summary.json`.

### 1. Inter-seed variance (SAC, 1M steps, base hyperparameters)

| seed | final return | std | x-progress (m) | fall rate | n |
|---|---|---|---|---|---|
| 42 | 5701 | 157 | 286 | 0% | 10 |
| 43 | 3286 | 904 | 171 | 50% | 10 |
| 44 | 2571 | 1108 | 133 | 43% | 30 |

Observed cross-seed mean **3853**, range **2571–5701** (spread 3130). The previously reported 5701 is the upper end of these three runs, not a typical-value estimate. These runs mix checkpoint and final evaluation protocols and do not establish a population range. Observed fall rates span 0–50%.

### 2. Convergence (seed 43, checkpoint eval every 250k)

| steps | 250k | 500k | 750k | 1M | 1.25M | 1.5M | 1.75M | 2M |
|---|---|---|---|---|---|---|---|---|
| return | 1068 | 2324 | 3348 | 3286 | 2011 | 2974 | 2315 | 3394 |

For this one training seed, checkpoint returns fluctuate after 750k (1M→2M: 3286→3394). This single trajectory does not establish general convergence or an optimal training budget. (`convergence.png`)

### 3. Hyperparameter ablation (SAC, seed 42, 500k steps, one factor at a time)

| factor | setting | return | std | fall |
|---|---|---|---|---|
| network | 64,64 | 657 | 579 | 80% |
| network | 128,128 | 2074 | 743 | 30% |
| network | 256,256 (base) | — (see 1M ref) | | |
| learning rate | 1e-4 | 1803 | 700 | 30% |
| learning rate | 3e-4 (base) | — (see 1M ref) | | |
| learning rate | 1e-3 | **4503** | 1241 | 10% |
| batch size | 128 | 2389 | 1266 | 60% |
| batch size | 256 (base) | — (see 1M ref) | | |
| batch size | 512 | 1707 | 1124 | 70% |

These are single-training-seed observations. The 1e-3 arm exceeded 1e-4 and batch 128 exceeded 512 under this protocol. No matched 500k final-evaluation control for the base configuration was run; a causal comparison to base is not established. (`ablation.png`)

### 4. Algorithm baselines (1M steps, seed 42)

| algo | return | std | x (m) | fall |
|---|---|---|---|---|
| **SAC** | **5701** | 157 | 286 | 0% |
| TD3 | 3117 | 415 | 137 | 17% |
| PPO | 1199 | 332 | 35 | 10% |

These are results of specific configurations at one training seed, not evidence of general algorithm superiority. TD3 was run without action exploration noise; PPO and TD3 each used one seed. SAC seed 42 was favorable among the observed SAC seeds. (`baselines.png`, `seeds.png`)

### Bottom line
- SAC, TD3 and PPO figures describe these individual configurations; no general ranking is established.
- The observed base 1M range is 2571–5701 (mean ≈ 3853), mixing checkpoint and final evaluation protocols.
- Seed 43's checkpoint curve fluctuates after 750k; one trajectory cannot establish general convergence.
- Ablations are exploratory and lack a matched final-evaluation base arm.

### 5. Best-config agent (applied the ablation findings)

Using the levers above, a final configuration — **SAC, network 256, lr=1e-3, batch=128, 1M steps** — was trained on three fresh seeds (50/51/52, 30 eval episodes each):

| seed | return | std | x (m) | fall | n |
|---|---|---|---|---|---|
| **50 (hero)** | **5818** | 148 | 283 | 0% | 30 |
| 51 | 5362 | 701 | 263 | 3% | 30 |
| 52 | 5023 | 1256 | 237 | 3% | 30 |

Observed best-config seed means average **5401**, range **5023–5818**; base-config means average 3853, range 2571–5701. The training seeds and some evaluation protocols differ; three runs per configuration cannot establish a general advantage. Seed 50 is the selected hero (`best_config/seed_50/sac_final.zip`). At 500k, base seed 43 checkpoint return was 2324; base seed 42 periodic evaluation was 3411 ± 1476 (n=10), using a different evaluation protocol. Neither establishes a matched causal comparison.

At 1M, the hero checkpoint saved before the final gradient update scored **5279 ± 1601, 2/10 falls**; after one further update the final model scored **5818 ± 148, 0/30 falls**. The first ten reset seeds overlap; these are different weights. Zero observed falls does not prove a zero fall probability.

Paired held-out inference on reset seeds 920000–920029: checkpoint **5854.15 ± 152.58, 0/30 falls**, final **5810.11 ± 99.82, 0/30 falls**. Both are the same pair of training runs; this does not establish population-level reliability. Per-episode CSV and summary: `artifacts/campaign/hero_checkpoint_comparison.csv` and `.json`.

### 6. Statistical methods and limitations

**Evaluation protocol.** Final evaluations use deterministic actions and seeded resets. The hero was additionally evaluated stochastically on 30 resets: **5685 ± 129, 0/30 falls, x = 277 m**. This is a finite-sample observation, not proof of robustness or absence of overfitting.

**Statistical correction.** An earlier version reported pooled episode-level bootstrap intervals, Mann–Whitney p and Cohen's d as evidence of a configuration effect. That inference was invalid: episodes share training runs, seed 43 was omitted, and unequal episode counts weighted training seeds unequally. Those numbers are withdrawn. Independent training runs are the relevant units for claims across training seeds.

**Limitations (stated plainly):**
1. **Not a controlled same-seed comparison.** Base-config evals use seeds 42/44 while best-config uses 50/51/52; seed 43 at 1M has no per-episode final data. The improvement is therefore not isolated from seed luck. A paired same-seed head-to-head would reduce some variability, but three seeds would still be insufficient for a conclusive general claim.
2. **Small n per configuration (3 seeds).** Underpowered for population-level claims; ranges are reported, not single points.
3. **Selection bias.** The best config (lr=1e-3, batch=128) was chosen because it performed best in the 500k ablations — selection on the same metric. This is standard practice but means the best-config numbers are optimistic relative to an unbiased search.
4. **Atypical learning rate.** lr=1e-3 is high for SAC and may be less stable than 3e-4; it is a measured finding for this task, not a general best practice.
5. **Asymmetric eval sizes.** The original seed-42 run used 10 final episodes; later runs used 30.
6. **Single environment.** All results are Ant-v5 only; no cross-environment robustness is claimed.
7. **Operational, not inferential.** With these sample sizes the statistics describe the observed runs; they are not population-level significance tests.

### 7. Preregistered follow-up (2026-11-19/20)

Nine independent 1M-step training runs used paired seeds 60/61/62, 30 deterministic reset episodes per run, fixed configurations and TD3 Gaussian exploration noise σ=0.1. Mean episode return by training seed:

| Algorithm | seed 60 | seed 61 | seed 62 | Mean of 3 |
|---|---:|---:|---:|---:|
| SAC | 3366.9 | 4243.5 | 4426.9 | 4012.4 |
| TD3 | 3367.8 | 3514.2 | 3044.1 | 3308.7 |
| PPO | 1087.8 | 948.0 | 809.2 | 948.3 |

SAC–TD3 paired differences are approximately −1.0, +729.3 and +1382.8. These are descriptive observations for three training seeds and one environment; no general superiority claim, pooled episode p-value or equal-compute claim is made. Equal budget means environment steps. Reset sets overlap across training seeds. See `research/followup/analysis.md`, per-run CSV and `EXPERIMENT_PROTOCOL.md`.

The separate SAC base control on seed 42 / 500k under the 500k ablations' final-evaluation protocol yielded **1584.4 ± 1183.2**, 7/10 falls. It is a single-seed exploratory comparison, not a general causal conclusion.