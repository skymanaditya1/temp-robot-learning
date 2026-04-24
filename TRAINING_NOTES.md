# RBY1 ACT training notes

Rationale for the hyperparameters in `train_rby1_pick_act.sh`. Update this file
when a value changes so the reasoning stays discoverable.

## Action chunking

- `chunk_size=20`
  - ACT paper uses 100 at 50 Hz control (= 2 s lookahead).
  - We run at ~10 Hz, so 20 steps also ≈ 2 s — same effective lookahead.
  - Do not read `20` as "a shorter chunk than the paper"; it matches in
    wall-clock terms, which is what matters for the task.

- `n_action_steps=1`, `temporal_ensemble_coeff=0.01`
  - Matches original ACT (Zhao et al. 2023). Every control step, run a
    forward pass, update a running exponentially-weighted mean over all
    overlapping predictions for each future timestep, and execute the
    popped front of the buffer.
  - Weights: `w_i = exp(-0.01 * age_i)` — older predictions have more
    weight (their chunks spent more steps averaging into the running mean).
  - Tradeoff: 20x more forward passes at rollout than
    `n_action_steps=chunk_size`. On Jetson this blocks 10 Hz control
    unless inference is offloaded to a workstation.
  - Fallback if we cannot afford per-step inference: set
    `temporal_ensemble_coeff=None` and `n_action_steps=chunk_size`.
    Expect jumpier trajectories at chunk boundaries and staler actions
    toward the end of each chunk.

## KL weight

- `kl_weight=10` (default in the script, configurable via 4th positional arg)
  - ACT paper value. Their setup used 50 demos per task — same scale as ours.
  - High KL pulls the VAE posterior toward `N(0, I)`, which makes the
    zero-latent approximation used at inference time a good one.
  - Risk when demos are few or very homogeneous: posterior collapse —
    the encoder matches the prior rather than encoding useful style
    information, latent becomes noise, predictions regress to the
    conditional mean. This manifests as a training-loss plateau.
  - Tuning guide when this happens:
    - Try `kl_weight=5` before going lower.
    - `kl_weight=1` is aggressive; `0.1` is almost certainly too low —
      zero-latent inference becomes a poor approximation and rollout
      quality can degrade even if train loss drops.
  - Diagnose posterior health by logging `mean(μ²)` and `mean(σ²)` from
    the VAE encoder:
    - Healthy: μ spread across roughly [-2, 2], σ noticeably below 1.
    - Collapsed: μ ≈ 0, σ ≈ 1 (matches the prior exactly, carries no info).

## Normalization

- `observation.state` / `action`: `MEAN_STD` with `eps=1e-8`.
  Frozen joints (std ≈ 0) cause normalized values to blow up. **Patch
  `meta/stats.json` to set `std=1.0` for any dim with `std < 1e-4`
  before training.** The unpatched dataset dominates the L1 signal with
  noise from frozen right-arm dims.
- Images: ImageNet stats (`use_imagenet_stats=True` in
  `configs/default.py`).

## Data

- 50 demos per task matches ACT paper scale. Do not assume more demos
  are required unless diagnostics say so.
- No image augmentation. Original ACT paper uses none either — confirmed
  by reading `tonyzhaozh/act`. Add this only if diagnostics point to
  over-reliance on visual features that augmentation would break
  (colour/lighting shortcutting).

## Logging

Logged per step (train), and every `val_freq=2000` steps (val, prefix `eval/`):

- `loss`, `l1_loss`, `kld_loss`
- `l1_loss_right_arm`, `l1_loss_left_arm`,
  `l1_loss_right_gripper`, `l1_loss_left_gripper`

Per-component L1 helps answer "is one joint group dominating the loss?"
— the usual culprit is the dominant-motion arm hiding a frozen other arm.

Val split:
- 10 % of episodes held out (episode-level, not frame-level).
- Seed defaults to `cfg.seed`. Split is dumped to
  `<output_dir>/val_split.json` for reproducibility.

## Changing any of these

Edit `train_rby1_pick_act.sh` and update this file in the same commit.
CLI-overriding at launch time is discouraged — the script is the single
source of truth so W&B notes stay consistent with what actually ran.
