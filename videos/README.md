# Recorded demos

All recordings are inference only, not new training.

- demo_initial_seed50.mp4: reconstructed seed-50 initial policy, 0 steps.
- demo_500k_seed50.mp4: selected seed-50 configuration at 500k steps.
- presentation_demo.mp4: selected seed-50 final model at 1M steps.
These three use the same reset seed, 100050.
- demo_250k_seed43.mp4: DIFFERENT seed-43/base run at 250k steps, reset 100050.
It is not an intermediate stage of the seed-50 trajectory.
- demo_td3_seed60.mp4 and demo_ppo_seed60.mp4: follow-up models at 1M
steps, evaluated on reset seed 100060. The hero SAC video is not a matched
seed-60 comparison.
