# SAC for Ant-v5 — presentation draft and presenter guide

**Status:** a content draft, not a designed final deck. English slide text and suggested speech are ready to adapt; English presenter notes explain the reasoning and evidence for the team. Replace team member names and adjust length to the instructor's allotted time, which has not been specified. The draft is not a script that must be read verbatim.

**Research distinction used throughout:** the *hero* video/model is the selected SAC configuration, seed 50 (`lr=0.001`, batch 128). The later controlled algorithm comparison uses the **base** SAC configuration (`lr=0.0003`, batch 256) on seeds 60–62. Do not present the hero's score as one of the three SAC follow-up scores.

---

## Slide 1 — Problem and objective

### On the slide (English)
**Learning to walk in Ant-v5 with Soft Actor-Critic**

- Goal: train a simulated quadruped to move forward.
- Method: off-policy reinforcement learning with continuous actions.
- Team 01 — [add names]

### Suggested speech (English)
> Our goal was to teach a simulated four-legged agent to move forward. The agent receives observations, chooses continuous motor commands and learns from rewards. We used Soft Actor-Critic, or SAC, and evaluated not only reward but also forward progress and falls.

### Background explanation
**What we explain:** this is a specific control problem, not just maximizing a number. The agent is a learned policy; the environment is the simulator. We do not manually program a sequence of leg movements: we learn a mapping from observations to actions.

**How to illustrate it:** a large frame from `presentation_demo.mp4` and a simple loop: observation → agent → action → environment → reward. Do not claim that one successful video proves the absence of falls.

**Sources:** video `artifacts/ant_sac_full_20261021T194131399885Z/best_config/seed_50/presentation_demo.mp4`; [Gymnasium Ant-v5](https://gymnasium.farama.org/environments/mujoco/ant/).

---

## Slide 2 — Environment and reward

### On the slide (English)
**Ant-v5: observations, actions and rewards**

- Observation: 105 values describing position, velocity and contact forces.
- Action: 8 continuous motor commands, each in [−1, 1].
- Reward = forward reward + healthy reward − control cost − contact cost.
- Episodes stop on unhealthy state or truncate at 1,000 steps.

### Suggested speech (English)
> Ant-v5 is a MuJoCo continuous-control environment. In our configured environment the observation has 105 values, and the action has eight continuous components. The reward encourages forward motion and staying healthy, while penalizing control effort and contact forces. A time limit is a truncation; it is not the same as falling.

### Background explanation
**Purpose:** the audience should understand the agent's inputs, controls and source of learning. The dimension 105 belongs to our configuration with contact forces enabled, not every possible Ant-v5 setup. The healthy reward means a high return alone does not prove effective movement; that is why we also measure x-progress.

**Terms:** `terminated` means an environment-defined ending, used here as observed falling/unhealthy state; `truncated` means an external episode-length limit. The distinction matters for critic bootstrapping. We make no claim about a physical step size or the absolute fidelity of a real robot model.

**How to illustrate it:** an Ant silhouette, a block diagram “105 → policy → 8”, and a short reward formula. Source: `run_experiment.py` → `make_env`, and the environment section of `sac_ant_v5.ipynb`.

---

## Slide 3 — Why SAC?

### On the slide (English)
**Soft Actor-Critic**

- Stochastic actor proposes continuous actions.
- Two critics estimate action values; target networks stabilize learning.
- Entropy encourages exploration; replay buffer reuses past transitions.
- Implemented with Stable-Baselines3, not written from scratch.

### Suggested speech (English)
> SAC is designed for continuous action spaces. Its actor proposes actions, while two critics estimate how useful those actions are. During learning the method also rewards entropy, which encourages exploration. We used Stable-Baselines3 for the implementation and a replay buffer to reuse collected experience.

### Background explanation
**What we explain:** the actor answers “what should I do?”, while the Q-critic answers “how useful is this choice, including its future consequences?”. Taking the minimum of two critic estimates reduces the risk of an overly optimistic target, but does not guarantee error-free estimates. Target networks update gradually; replay makes the method off-policy because collected data can be reused. Entropy is not randomness for its own sake: it is part of the objective. Its influence is controlled by `alpha`, which was adjusted automatically in our setup.

**How to illustrate it:** Actor → action → environment → replay → Critic → Actor; mark the automatic temperature and two Q-functions. Do not say we implemented the library's SAC ourselves.

**Sources:** [SAC paper](https://arxiv.org/abs/1801.01290), [SB3 SAC documentation](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html), `EXPERIMENT_PROTOCOL.md`.

---

## Slide 4 — One worked critic update

### On the slide (English)
**One simplified SAC critic step**

- Soft target: `y = r + γ(min(Q₁′, Q₂′) − α log π(a′|s′))`
- Example: `1 + 0.99 × (3.5 − 0.2 × (−1.2)) = 4.7026`
- Scalar critic `Q: 2.0000 → 2.5405` after one gradient step.
- Toy arithmetic only: the real implementation trains neural networks in batches.

### Suggested speech (English)
> The critic tries to match a target that combines immediate reward with a discounted estimate of the next state, including the entropy term. In this small non-terminal example, the target is 4.7026. Starting from a value estimate of 2, one gradient step moves it to 2.5405 and reduces squared error. The notebook contains the runnable calculation.

### Background explanation
**Step-by-step explanation:** reward = 1; the two target critics output 4 and 3.5, so take the minimum, 3.5. With `alpha = 0.2`, `log π = −1.2`, `gamma = 0.99`, the target is 4.7026. Current Q = 2, squared loss = 7.30405. The derivative is `2(Q−y) = −5.4052`; at learning rate 0.1, the new Q is 2.54052 and loss is 4.67459. The target stays fixed during this step.

**Important caveats:** the slide formula is for a *non-terminal* transition. A true terminal transition removes the bootstrap; an ordinary time-limit truncation should not automatically count as terminal. The actor samples the next action; `log π` is a log density. This is **one scalar value**, not a complete network model. Real SB3 uses batches, neural networks, Adam and two critics. Reduced loss in this toy step does not establish overall policy improvement.

**How to illustrate it:** open and run the `Worked example: one SAC critic update` cell in `sac_ant_v5.ipynb`; it needs only Python, not MuJoCo or training. Show an arrow moving Q toward the target.

---

## Slide 5 — How we trained the agent

### On the slide (English)
**Training pipeline**

- Collect transitions → replay buffer → actor/critic updates.
- Ant-v5; 1,000,000 environment steps per main run.
- SAC base: MLP 256×256, learning rate 3e−4, batch 256.
- 10,000 warm-up steps; automatic entropy coefficient.
- Training, periodic evaluation and final evaluation use distinct seeds.

### Suggested speech (English)
> The agent interacts with Ant and stores transitions in a replay buffer. After an initial warm-up, SAC samples batches and updates the networks. Our base configuration uses two hidden layers of 256 units and one million environment steps per training run. Evaluation is separate from learning and uses deterministic actions.

### Background explanation
**Main point:** this is an actual process, not just theory. Follow-up SAC records 990,000 gradient updates after warm-up; do not promise an equivalent count for PPO, whose counter uses a different unit. These settings describe **base SAC in the follow-up**, not the hero policy in the video.

**Hero:** a separate SAC configuration was selected using exploratory ablations: 256×256, learning rate `1e−3`, batch 128, seed 50, 1M steps. This is a selected demonstration policy, not a random representative of the follow-up. The selection process introduces selection bias.

**Sources:** `EXPERIMENT_PROTOCOL.md`; `research/followup/runs/sac_60/summary.json`; notebook. Experimental Linux environment versions: Gymnasium 1.2.1, MuJoCo 3.3.7, SB3 2.7.0, PyTorch 2.5.1. Matching outputs across versions is not guaranteed.

---

## Slide 6 — How we measured performance

### On the slide (English)
**Evaluation: more than reward**

- Deterministic evaluation; separate reset seeds.
- Metrics: episode return, x-progress and falls.
- Follow-up: 3 independent training seeds × 30 evaluation episodes per algorithm.
- Training seeds—not evaluation episodes—are independent experimental units.

### Suggested speech (English)
> A high reward alone does not prove that the ant walks forward, because the environment also rewards being healthy. We therefore report forward displacement and falls alongside return. In the follow-up we trained each algorithm three times, then evaluated each trained model on 30 reset seeds. Those episodes measure a policy; they are not 30 independent training runs.

### Background explanation
**Why it matters:** this separates training from evaluation. `return` is the sum of episode rewards; `x-progress` is final x minus initial x; a fall in our table is termination due to an unhealthy state. A table's `±` value for an individual model is the sample standard deviation **across episodes**, not a confidence interval across training seeds.

**Follow-up pairing:** for algorithms sharing a training-seed label, the same 30 reset seeds are used. Reset-seed sets partially overlap across different training seeds. Three seeds are a small sample and do not justify strong statistical conclusions. Differences can be described without a pooled-episode p-value.

**Source:** `EXPERIMENT_PROTOCOL.md`, `research/followup/analysis.md` and primary CSVs in `research/followup/runs/`.

---

## Slide 7 — Results: controlled follow-up

### On the slide (English)
**Fixed configurations, 1M environment steps**

| Training seed | SAC | TD3 | PPO |
|---|---:|---:|---:|
| 60 | 3,367 | 3,368 | 1,088 |
| 61 | 4,244 | 3,514 | 948 |
| 62 | 4,427 | 3,044 | 809 |
| Mean of 3 runs | **4,012** | **3,309** | **948** |

TD3 uses Gaussian action noise σ=0.1. Three training seeds per algorithm; results are descriptive.

### Suggested speech (English)
> In this follow-up we compared fixed SAC, TD3 and PPO configurations using the same environment-step budget and three matched training-seed labels. SAC and TD3 were almost equal on seed 60, while SAC had higher mean return on seeds 61 and 62. PPO's returns were lower in these runs. This is a small comparison of particular configurations, not proof of a general algorithm ranking.

### Background explanation
**What is compared:** base SAC, TD3 and PPO, with 30 deterministic resets for every trained seed. The mean row is the arithmetic mean of the three model means. Matching training-seed labels helps organize pairs, but does not make different algorithms identical random processes. TD3 here uses exploration noise 0.1, unlike the older single-seed TD3 baseline without noise.

**What not to claim:** these algorithms were not comprehensively tuned; equal environment interactions do not imply equal runtime, update counts or computational cost. SAC − TD3 differences by seed are −1.0, +729.3, +1382.8. Do not announce a statistically significant win.

**How to illustrate it:** `artifacts/ant_sac_full_20261021T194131399885Z/hardening/followup.png` shows individual training-seed points and their mean as a horizontal bar. Do not use the old `results_slide.png` as the main new comparison: it contains historical single-seed baselines and crowded labels.

**Source:** `research/followup/analysis.md`, `research/followup/runs/`.

---

## Slide 8 — Hero model and demo

### On the slide (English)
**Selected SAC policy: seed 50**

- Selected configuration: `lr=1e−3`, batch 128, 1M steps.
- 30 deterministic evaluation resets: return **5,818 ± 148**, x-progress **≈283 m**.
- Observed falls: **0/30** — not proof of zero fall probability.
- Demo: recorded rollout, no training during playback.

### Suggested speech (English)
> This is the selected policy from a separate exploratory SAC configuration. Across 30 deterministic evaluation resets its mean return was about 5,818 and mean forward progress about 283 metres in the simulator. No falls were observed in those 30 episodes, but this does not guarantee the policy never falls. We will now show a recorded rollout.

### Background explanation
**Why separate it from the comparison:** the hero is demonstration seed 50 after configuration selection, not SAC seed 60 from the preceding slide. Do not insert 5,818 into the follow-up mean table as though it were a comparable overall result. `±148` is sample SD across 30 reset episodes of one policy, not uncertainty across three training runs. A different software stack can change trajectories.

**Video:** `artifacts/ant_sac_full_20261021T194131399885Z/best_config/seed_50/presentation_demo.mp4` (50 seconds). This is one deterministic rollout with reset seed 100050, not an average episode. For clarity only the floor's visible extent was enlarged and an x-progress overlay was added. The episode matches the original CSV evaluation exactly: return 5569.4366, x-progress 271.1292 m, 1000 steps. Before presenting, check playback on your machine; an excerpt is acceptable if identified as an excerpt.

**Sources:** `artifacts/campaign/best/seed_50/summary.json`, `evaluation_all.csv`; `OVERNIGHT_REPORT.md` §5. A separate held-out comparison of the pre-final and final models is in `artifacts/campaign/hero_checkpoint_comparison.csv`.

---

## Slide 9 — Limitations and conclusion

### On the slide (English)
**What we learned — and what we cannot claim**

- SAC learned forward locomotion in our Ant-v5 setup.
- Outcomes varied across training seeds and configurations.
- Follow-up results describe 3 seeds, one environment and fixed settings.
- More seeds, environments and tuning are needed for broader comparisons.

### Suggested speech (English)
> We successfully trained and demonstrated a policy that moves forward in our configured Ant environment. Results vary across seeds, and our algorithm comparison is limited to a few runs and fixed hyperparameters. Therefore we report the observations without claiming general superiority or guaranteed robustness. More independent runs and environments would be the natural next step.

### Background explanation
**Main conclusion:** we can claim a functioning trained policy and describe measured results. We cannot say “never falls”, “proven generally better than PPO and TD3”, “general convergence achieved”, or “the difference is statistically significant”.

**Why:** three training seeds are few; the hero and hyperparameters were selected after exploration; Ant is one task; computational costs differ across algorithms. Older base and selected-configuration runs also differ in seeds and, sometimes, evaluation protocols. Related episodes do not replace independent training runs.

---

## Suggested questions and short answers

**Why SAC instead of DQN?** Ant actions are continuous; SAC directly models a continuous policy, whereas standard DQN assumes discrete actions.

**What is entropy for?** It is part of the SAC objective and promotes a less concentrated policy during learning; the temperature controls the trade-off. Evaluation here uses deterministic actions.

**Why two critics?** SAC uses the minimum of two estimates in the target to mitigate overestimation. This is a design choice, not a guarantee of correctness.

**Why separate fall rate and x-progress from return?** Healthy reward contributes to return, so reward alone can be misleading about locomotion.

**Are 90 episodes the same as 90 independent training runs?** No. In the follow-up each algorithm has three trained models; the 30 resets evaluate each model.

**Why is the demo score higher than the follow-up SAC mean?** The demo is a selected seed/configuration from a separate exploratory experiment; the follow-up is the base configuration on different training seeds. They are not a controlled head-to-head comparison.

**Does 0/30 mean the policy never falls?** No. It means zero observed falls in that set of 30 evaluations, under that software setup.

**Did you implement SAC yourself?** No. We used Stable-Baselines3 and wrote environment setup, experiments, evaluation and analysis. The small example illustrates the update, not the full library implementation.

## Evidence and material locations

- `EXPERIMENT_PROTOCOL.md` — preregistered follow-up configurations.
- `research/followup/analysis.md` and `research/followup/runs/` — means and per-episode evidence.
- `OVERNIGHT_REPORT.md` — historical campaign, hero and limitations.
- `sac_ant_v5.ipynb` — experiment plus the standalone toy update cell near the top.
- `artifacts/ant_sac_full_20261021T194131399885Z/hardening/followup.png` — follow-up graph.
- `artifacts/ant_sac_full_20261021T194131399885Z/best_config/seed_50/presentation_demo.mp4` — recorded demo (one reset; visible-floor adjustment and progress overlay).
- Original method: Haarnoja et al., *Soft Actor-Critic* (2018), https://arxiv.org/abs/1801.01290; environment https://gymnasium.farama.org/environments/mujoco/ant/; implementation https://stable-baselines3.readthedocs.io/en/master/modules/sac.html.
