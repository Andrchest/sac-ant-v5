# SAC × Ant-v5 experiment report

This report covers the original SAC run, the exploratory campaign, and the
matched SAC, TD3, and PPO follow-up. Evaluation standard deviations are across
episodes of a trained policy. Comparisons across training seeds use the trained
run as the unit of analysis.

## Original SAC run (seed 42)

The October 2026 run `ant_sac_full_20261021T194131399885Z` completed 1,000,000
environment steps and 990,000 gradient updates. In 10 final evaluation episodes,
the policy averaged **5701.04 ± 157.17** in return, **286.39 m** of forward
progress, and **1000 steps** per episode, with **0/10 falls**. The run met the
supervisor's operational gait heuristic and finished after its first round.
[Run summary](research/campaign/reference/summary.json) · [Per-episode data](research/campaign/reference/evaluation_all.csv)

### Evaluation on 30 previously unused reset seeds

The same final checkpoint was evaluated without additional training on seeds 900042–900071 (not used for training, periodic evaluation or the original final evaluation).

- Mean return: **5303.18**, sample standard deviation **1156.93**; range **1895.88–5903.54**.
- Mean forward progress: **266.57 m**.
- Mean episode length: **944.13 / 1000**.
- Falls: **3 / 30 (10%)**.
- The policy met the operational gait threshold in the fresh-reset evaluation. Its training budget remained at **1,000,000 steps**.

The evaluation describes the performance of a single trained policy across reset
conditions. [Per-episode data](research/campaign/reference/unseen_reset_validation.csv) · [Evaluation summary](research/campaign/reference/unseen_reset_validation.json)

---

## Hardening campaign (2026-11-12, multi-seed + ablations + baselines)

The exploratory campaign completed 10 jobs with three workers and four Torch
threads per worker. It used the original seed-42 run as its full-budget SAC
reference. [Campaign index](research/campaign/derived_summary.json) · [Run data](research/campaign/runs/)

### 1. Inter-seed variance (SAC, 1M steps, base hyperparameters)

| Seed | Evaluation | Mean return | SD | Forward progress (m) | Fall rate | Episodes |
|---|---|---:|---:|---:|---:|---:|
| 42 | Final | 5701 | 157 | 286 | 0% | 10 |
| 43 | 1M-step checkpoint | 3286 | 904 | 171 | 50% | 10 |
| 44 | Final | 2571 | 1108 | 133 | 43% | 30 |

The mean of the three seed means was **3853**, and observed returns ranged from
**2571 to 5701**. The evaluations cover different episode counts and include a
checkpoint from the longer seed-43 run. Observed fall rates ranged from 0% to 50%.

### 2. Convergence (seed 43, checkpoint eval every 250k)

| steps | 250k | 500k | 750k | 1M | 1.25M | 1.5M | 1.75M | 2M |
|---|---|---|---|---|---|---|---|---|
| return | 1068 | 2324 | 3348 | 3286 | 2011 | 2974 | 2315 | 3394 |

Checkpoint returns fluctuated after 750k for seed 43, including a low of 2011 at
1.25M and a return of 3394 at 2M. This is one training trajectory.
[Checkpoint summaries](research/campaign/runs/conv_s43_2m/summary.json)

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

The 1e-3 arm exceeded 1e-4 and batch 128 exceeded 512 under this protocol. The
original campaign compared one training seed and used a 1M-step base reference.
The later fixed-protocol campaign added a 500k-step base control on seed 42 (§7).

### 4. Algorithm baselines (nominal 1M-step budget, seed 42)

| Algorithm | Actual steps | Mean return | SD | Forward progress (m) | Fall rate | Episodes |
|---|---:|---:|---:|---:|---:|---:|
| SAC | 1,000,000 | 5701 | 157 | 286 | 0% | 10 |
| TD3 | 1,000,000 | 3117 | 415 | 137 | 17% | 30 |
| PPO | 1,001,472 | 1199 | 332 | 35 | 10% | 30 |

PPO completed 2,048-step rollouts, which took it past the requested 1M steps.
TD3 used no action exploration noise. All three results are single-seed baselines.
[PPO run summary](research/campaign/runs/ppo_1m/summary.json) · [TD3 run summary](research/campaign/runs/td3_1m/summary.json)

### 5. Best-config agent (applied the ablation findings)

The ablations informed a configuration with **two 256-unit layers, lr=1e-3, and
batch size 128**. Seeds 50, 51, and 52 each trained for 1M steps and used 30
final evaluation episodes.

| Seed | Mean return | SD | Forward progress (m) | Fall rate | Episodes |
|---|---|---|---|---|---|
| **50 (selected)** | **5818** | 148 | 282.5 | 0% | 30 |
| 51 | 5362 | 701 | 263.4 | 3% | 30 |
| 52 | 5023 | 1256 | 237.3 | 3% | 30 |

The selected configuration had an observed mean of **5401** and a range of
**5023–5818** across the three seed means. The base configuration averaged 3853
with a range of 2571–5701. The groups used different training seeds and some
different evaluation protocols. Seed 50 supplies the recorded demo.
[Selected-configuration run data](research/campaign/best/)

The archived seed-50 evaluations cover two snapshots of one training run. The
1M-step checkpoint at 989,999 gradient updates scored **5279 ± 1601** with
**2/10 falls**. The final model at 990,000 updates scored **5818 ± 148** with
**0/30 falls**. The first ten reset seeds overlapped in those evaluations.

In an archived paired evaluation on reset seeds 920000–920029, the checkpoint
scored **5854.15 ± 152.58** and the final model scored **5810.11 ± 99.82**. Each
had **0/30 observed falls**. Their paired evidence is stored in the archived
release bundle under `artifacts/campaign/hero_checkpoint_comparison.csv` and
`artifacts/campaign/hero_checkpoint_comparison.json`.

### 6. Statistical methods and limitations

**Evaluation protocol.** Final evaluations use deterministic actions and seeded resets. A previously quoted stochastic evaluation is withdrawn from this report because its per-episode source was not preserved; rerunning an unseeded stochastic policy would not reproduce the original sample.

**Statistical correction.** An earlier version reported pooled episode-level bootstrap intervals, Mann–Whitney p and Cohen's d as evidence of a configuration effect. That inference was invalid: episodes share training runs, seed 43 was omitted, and unequal episode counts weighted training seeds unequally. Those numbers are withdrawn. Independent training runs are the relevant units for claims across training seeds.

**Limitations**

1. The base results use seeds 42, 43, and 44. The selected configuration uses seeds 50, 51, and 52. Performance differences can reflect training-seed variation as well as configuration.
2. Three runs per configuration give a small sample. The reported means and ranges describe the observed seeds.
3. The 500k exploratory results informed the learning rate and batch size separately. The joint setting was evaluated later on three fresh seeds.
4. The 1M-step base evaluations used 10 episodes for seeds 42 and 43 and 30 episodes for seed 44. The seed-43 result came from a checkpoint.
5. The study evaluated one environment, Ant-v5.

### 7. Fixed-protocol follow-up (2026-11-19/20)

Nine independent 1M-step training runs used paired seeds 60/61/62, 30 deterministic reset episodes per run, fixed configurations and TD3 Gaussian exploration noise σ=0.1. Mean episode return by training seed:

| Algorithm | seed 60 | seed 61 | seed 62 | Mean of 3 |
|---|---:|---:|---:|---:|
| SAC | 3366.9 | 4243.5 | 4426.9 | 4012.4 |
| TD3 | 3367.8 | 3514.2 | 3044.1 | 3308.7 |
| PPO | 1087.8 | 948.0 | 809.2 | 948.3 |

SAC–TD3 paired differences were approximately −1.0, +729.3, and +1382.8 by
training seed. These are descriptive comparisons of the tested configurations.
Reset sets overlap across training seeds, and matching environment steps leaves
compute budgets free to differ. [Protocol](EXPERIMENT_PROTOCOL.md) · [Analysis](research/followup/analysis.md) · [Evaluation plot](research/followup/followup.png) · [Per-run data](research/followup/runs/)

The separate SAC base control on seed 42 / 500k under the 500k ablations' final-evaluation protocol yielded **1584.4 ± 1183.2**, 7/10 falls. It is a single-seed exploratory comparison, not a general causal conclusion.
