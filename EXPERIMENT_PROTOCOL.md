# Ant-v5 follow-up protocol

This protocol defines the settings for the comparison of fixed configurations.
The launcher recorded the job arguments at campaign start. The protocol file
first appeared in Git about six minutes after launch. [Recorded job arguments](research/followup/source_status.json)

## Training arms

- Environment: Ant-v5, reward settings and wrappers in `run_experiment.make_env`.
- Training seeds: 60, 61, 62 for every algorithm; exactly 1,000,000 environment steps per run.
- SAC: MlpPolicy [256,256], learning rate 3e-4, batch 256, auto entropy,
  gamma .99, tau .005, warmup 10,000, one gradient update per step.
- TD3: same architecture, learning rate, batch, gamma, tau and warmup;
  Gaussian exploration action noise with standard deviation 0.1 for each action dimension.
  SB3 defaults are used for target policy noise and delayed updates.
- PPO: [256,256], learning rate 3e-4, batch 256, gamma .99, rollout 2,000 steps,
  other SB3 defaults, no manually added entropy bonus. 2,000 divides 1M exactly.
- Evaluation: deterministic actions, same 30 episode reset seeds within each training
  seed across algorithms (100060–100089, 100061–100090, or 100062–100091).
  Cross-training-seed reset sets overlap; independent experimental units are training runs,
  not episodes. Report each seed and return, x-progress, falls; compare paired seed means
  descriptively, no pooled episode p-value. Three training seeds cannot justify definitive
  algorithm superiority or reliably quantify rare falls. Evaluation uses seeds distinct
  from training and periodic evaluation.
- A separate SAC base control: training seed 42, 500,000 steps, base SAC
  hyperparameters, and the same ten final-evaluation reset seeds as the earlier
  seed-42 ablations.
  This is a single-seed baseline, not a general causal claim.

The launcher fixed these settings at the start of the follow-up. The separate
selected-configuration runs came from exploratory ablations.
Budget parity is in environment steps, not wall clock or gradient update count. These are
single fixed hyperparameter configurations, not tuned algorithm optima. Models and replay
buffers are excluded from Git and compact publication artifacts unless explicitly selected.

## Running the follow-up

Run `python followup_campaign.py` from the repository root on Linux with at least
9 GiB of available memory. The launcher uses up to three CPU workers and four
Torch threads per worker. It records commands, exit status, and logs under a new
`artifacts/followup_*` directory and writes each run under `artifacts/xp_*`.
[Recorded results](research/followup/analysis.md)
