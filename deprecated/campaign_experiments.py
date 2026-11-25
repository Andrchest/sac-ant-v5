"""Experiment list for the SAC × Ant-v5 ablation campaign.

Ordered by priority. The reference baseline (SAC seed 42, 1M steps) is the
pre-existing full run and is not re-executed here; it is merged into the
campaign report by ablation_campaign.py.
"""

# Baseline config (the "default" SAC): net [256,256], lr 3e-4, batch 256.
# Each ablation changes exactly ONE factor from this baseline (one-factor-at-a-time).

BASE = dict(algo='sac', net='256,256', lr=3e-4, batch=256,
            gamma=0.99, tau=0.005, ent_coef='auto', learning_starts=10_000)


def exp(overrides):
    d = dict(BASE)
    d.update(overrides)
    return d


EXPERIMENTS = [
    # ---- Tier 1: core (seeds + convergence) ------------------------------
    # 2M run with checkpoints every 250k -> convergence curve AND the seed-43
    # 1M data point (the 1M checkpoint is evaluated like the other seeds).
    exp(dict(id='conv_s43_2m', seed=43, steps=2_000_000,
             checkpoint_every=250_000, checkpoint_eval_episodes=10,
             eval_episodes=30, tag='conv', priority=1)),
    exp(dict(id='seed44_1m', seed=44, steps=1_000_000,
             eval_episodes=30, tag='seed', priority=1)),

    # ---- Tier 2: hyperparameter ablation (500k steps, seed 42) -----------
    exp(dict(id='net128', net='128,128', seed=42, steps=500_000,
             eval_episodes=10, tag='ab_net128', priority=2)),
    exp(dict(id='net64', net='64,64', seed=42, steps=500_000,
             eval_episodes=10, tag='ab_net64', priority=2)),
    exp(dict(id='lr1e4', lr=1e-4, seed=42, steps=500_000,
             eval_episodes=10, tag='ab_lr1e4', priority=2)),
    exp(dict(id='lr1e3', lr=1e-3, seed=42, steps=500_000,
             eval_episodes=10, tag='ab_lr1e3', priority=2)),
    exp(dict(id='batch128', batch=128, seed=42, steps=500_000,
             eval_episodes=10, tag='ab_batch128', priority=2)),
    exp(dict(id='batch512', batch=512, seed=42, steps=500_000,
             eval_episodes=10, tag='ab_batch512', priority=2)),

    # ---- Tier 3: baselines (if time allows) ------------------------------
    dict(id='ppo_1m', algo='ppo', ppo_n_steps=2048, seed=42, steps=1_000_000,
         eval_episodes=30, tag='baseline_ppo', priority=3),
    exp(dict(id='td3_1m', algo='td3', seed=42, steps=1_000_000,
             eval_episodes=30, tag='baseline_td3', priority=3)),
]

# Pre-existing reference (not re-run). Merged into the report.
REFERENCE = dict(id='ref_sac42_1m', algo='sac', seed=42, steps=1_000_000,
                 net=[256, 256], lr=3e-4, batch=256,
                 final_return_mean=5701.04, final_return_std=157.17,
                 final_x_progress_mean=286.39, final_length_mean=1000.0,
                 final_fall_rate=0.0, eval_episodes=10,
                 run_dir='artifacts/ant_sac_full_20261021T194131399885Z',
                 note='pre-existing full run (this campaign)')
