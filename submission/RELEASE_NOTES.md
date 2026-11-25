# SAC × Ant-v5 — project materials

Download `sac_ant_v5_submission_english.zip` for the compact project bundle:
notebook, source code, evaluation data, selected SAC model, final demo and a
presentation content draft. This is not a finished slide deck. The ZIP is an
independently verified snapshot; the repository README additionally provides
GitHub navigation.

## Videos

- `demo_initial_seed50.mp4`: reconstructed initial seed-50 policy, 0 training steps.
  Its evaluation matches the originally recorded initial-policy evaluation.
- `demo_500k_seed50.mp4`: seed-50 checkpoint at 500k steps.
- `presentation_demo.mp4`: final selected SAC seed-50 model, 1M steps.
  These three videos use reset seed 100050.
- `demo_250k_seed43.mp4`: separate seed-43/base-configuration run at 250k steps;
  **not** an intermediate checkpoint of the seed-50 learning trajectory.
- `demo_td3_seed60.mp4` and `demo_ppo_seed60.mp4`: follow-up models trained for 1M
  steps, both recorded on reset seed 100060. These are separate example episodes,
  not a population-level ranking. The selected SAC hero video is not a matched
  seed-60 SAC comparator.

The visible floor extent was enlarged for clarity and a progress overlay added.
No training occurs during recording. Models and environment trajectories can be
sensitive to software versions. Zero observed falls does not imply zero fall risk.

## Model and data

`sac_final.zip` is the selected seed-50 SAC policy (learning rate 1e-3, batch 128).
The English submission ZIP includes this same model and final video. Training
replay buffers and local environments are not published. SHA-256 of the compact
ZIP is provided in its `.sha256` asset.

The controlled follow-up used three independent training seeds per algorithm,
not 90 independent trainings. See `EXPERIMENT_PROTOCOL.md` and
`research/followup/analysis.md` in the repository.
