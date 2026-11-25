# SAC × Ant-v5 — final project materials

This archive contains a trained continuous-control agent, a recorded demo, the
experiment notebook and source code, primary results, and a **presentation content
draft**. The team still needs to turn that draft into its final slides and add names
before uploading to Moodle. The announcement did not specify a slide format, a
presentation duration, a time zone or an upload-size limit; confirm those separately.

## Start here

- **Presentation draft and speaker notes:** `PRESENTATION_DRAFT.md` — slide-ready
  English text and suggested speech, with detailed English explanations for the team.
  This is not a finished PPTX/PDF deck.
- **Video:** `artifacts/ant_sac_full_20261021T194131399885Z/best_config/seed_50/presentation_demo.mp4`.
  This is a 50-second recording of the selected trained policy on reset seed 100050,
  not a live training session. The visible floor extent was enlarged and a progress
  overlay added; the episode return (5569.4366), x-progress (271.1292 m) and length
  (1000 steps) match the original per-episode evaluation exactly. `rollout.mp4` is
  the earlier recording; `make_presentation_video.py` creates the presentation version.
- **Notebook:** `sac_ant_v5.ipynb`. Near the top, find *Worked example: one SAC critic
  update*. Its single code cell runs in plain Python without any installation, MuJoCo,
  the trained model or training. The saved output shows the target, Q and loss before
  and after one gradient step. The example is illustrative, not a literal trace of
  the SB3 neural-network optimizer. Other notebook cells implement the real experiment.
- **Experiment sources:** `train_sac.py`, `run_experiment.py`, notebook cells, and
  supporting evaluation, monitoring and analysis scripts. See `BUNDLE_README.md`
  for setup, data locations and inference. The extensive `README.md` describes the
  original research workflow in English.
- **Recorded comparisons:** `research/followup/analysis.md`, per-run CSV/JSON under
  `research/followup/runs/`, and the preregistration `EXPERIMENT_PROTOCOL.md`.
- **Plot:** `artifacts/ant_sac_full_20261021T194131399885Z/hardening/followup.png`.

## Important interpretation

The demo is selected SAC seed 50 (learning rate 1e-3, batch 128). It is separate
from the base SAC runs (learning rate 3e-4, batch 256) in the later SAC/TD3/PPO
comparison. Three independent training seeds per algorithm support descriptive
comparisons of these fixed configurations, not a general algorithm ranking. A
policy with zero observed falls in 30 evaluations is not guaranteed never to fall.
The notebook's older saved training outputs are from a smoke run, not the full run.

## Before submission

1. Build the final presentation from the draft, add names and check the format.
2. Play the MP4 on the presentation machine and review the actual footage.
3. Run the standalone example cell; do **not** click Run All unless you intend to
   perform the smoke training run. Full training is optional and takes much longer.
4. Verify the Moodle deadline/time zone, size limit and presentation duration.
5. If the archive was extracted with other programs/users, ensure file permissions
   permit reading. This archive explicitly records standard 0644 permissions.

The archive contains no replay buffer, local virtual environment or credentials.
Some historical scripts for rerunning full experiments expect the original local
run tree and are included as source; the selected model and all data needed for
the reported follow-up analysis are included. `MANIFEST.json` hashes the archive
contents other than the manifest itself.
