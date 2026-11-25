# Fixed-protocol Ant-v5 follow-up

The protocol is defined in `EXPERIMENT_PROTOCOL.md`. Each algorithm was trained
on seeds 60, 61, and 62. Within each training seed, the algorithms used the same
30 deterministic evaluation resets. Results describe these fixed configurations
in Ant-v5. The falls column counts unhealthy environment terminations.

| algorithm | training seed | steps | return mean ± episode SD | x progress (m) | falls |
|---|---:|---:|---:|---:|---:|
| SAC | 60 | 1,000,000 | 3366.9 ± 1051.1 | 171.4 | 7/30 |
| SAC | 61 | 1,000,000 | 4243.5 ± 1215.0 | 218.2 | 6/30 |
| SAC | 62 | 1,000,000 | 4426.9 ± 992.7 | 229.2 | 2/30 |
| TD3 | 60 | 1,000,000 | 3367.8 ± 564.5 | 146.7 | 6/30 |
| TD3 | 61 | 1,000,000 | 3514.2 ± 581.0 | 152.7 | 2/30 |
| TD3 | 62 | 1,000,000 | 3044.1 ± 865.9 | 133.9 | 9/30 |
| PPO | 60 | 1,000,000 | 1087.8 ± 652.1 | 39.0 | 14/30 |
| PPO | 61 | 1,000,000 | 948.0 ± 455.2 | 34.1 | 18/30 |
| PPO | 62 | 1,000,000 | 809.2 ± 516.5 | 29.4 | 22/30 |

## Descriptive training-seed aggregates

- SAC: mean 4012.4; range 3366.9–4426.9 (n=3 training runs).
- TD3: mean 3308.7; range 3044.1–3514.2 (n=3 training runs).
- PPO: mean 948.3; range 809.2–1087.8 (n=3 training runs).

Paired SAC − TD3 differences by training seed: 60: -1.0, 61: +729.3, 62: +1382.8.
These are descriptive; no episode-pooled p-value or population superiority claim.

## Single-seed 500k SAC base control

Seed 42, same final reset protocol as 500k ablations: 1584.4 ± 1183.2, 7/10 falls.
Compare to existing 500k arms only as single-seed exploratory observations; no independent replication.

Evaluation episode standard deviations describe reset variability, not training-seed uncertainty.
TD3 uses Gaussian exploration noise σ=0.1. Equal budgets are environment steps, not compute time.
Three seeds and one environment limit generalization. Reset sets overlap across seeds.
