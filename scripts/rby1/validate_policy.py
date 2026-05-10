"""Offline policy validation across multiple episodes (LeRobotDataset).

Runs the trained policy in open-loop replay mode against recorded episodes
and reports per-episode + cross-episode metrics. Use this for batched sanity
checks ("does the policy fit the training set?", "how does ckpt 90K compare
to 280K?", etc.).

This is the LeRobotDataset / non-ROS analog of minitation's val_policy.py.
The shared idea: replay observations, compare predicted vs. recorded actions,
aggregate metrics. Differences from overfit_inference_plot.py:
  - Loops over multiple episodes (not just one).
  - Saves per-episode metrics JSON + an aggregate summary JSON.
  - Per-episode try/except so one bad episode doesn't abort the run.
  - Optional --split-file (e.g. the split.json LeRobotDataset can write at
    training time) so you can validate exactly the held-out val split.
  - Optional --plot-indices to keep plots focused on dims of interest.

Example:
    conda run -n policy_inference python scripts/rby1/validate_policy.py \\
        --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model \\
        --dataset-repo-id local/rby1_pick_v3_20260422_174437_left_only \\
        --dataset-root datasets/local/rby1_pick_v3_20260422_174437_left_only \\
        --episodes 0,1,2 \\
        --output-dir temp_images/validation_<run>_last
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import PreTrainedPolicy, make_pre_post_processors
from lerobot.policies.factory import get_policy_class


# ──────────────────────────────────────────────────────────────────────────────
# Per-episode bookkeeping
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class EpisodeResult:
    """Metrics for a single episode."""

    episode_idx: int
    num_frames: int
    overall_mae: float
    per_joint_mae: dict[str, float]
    per_joint_max_err: dict[str, float]
    per_joint_p99_err: dict[str, float]
    inference_seconds: float
    plot_path: str | None = None
    error: str | None = None  # populated if validation crashed mid-episode

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Argument helpers
# ──────────────────────────────────────────────────────────────────────────────


def _parse_episodes_arg(arg: str | None, all_episodes: list[int]) -> list[int]:
    """Resolve --episodes / --all-episodes into a concrete list of indices."""
    if arg is None:
        return all_episodes
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    bad = [i for i in out if i not in all_episodes]
    if bad:
        raise SystemExit(f"--episodes contains indices not in dataset: {bad}")
    return sorted(set(out))


def _episodes_from_split(split_file: Path, split_type: str) -> list[int]:
    """Read episode indices from a split.json (`{train: [...], val: [...]}`)."""
    data = json.loads(Path(split_file).read_text())
    if split_type not in data:
        raise SystemExit(f"--split-type {split_type!r} not found in {split_file}; "
                         f"available: {list(data.keys())}")
    return sorted(int(i) for i in data[split_type])


def _parse_plot_indices(arg: str | None, n_dof: int) -> list[int]:
    if arg is None:
        return list(range(n_dof))
    out = sorted({int(p) for p in arg.split(",") if p.strip() != ""})
    bad = [i for i in out if not (0 <= i < n_dof)]
    if bad:
        raise SystemExit(f"--plot-indices out of range [0, {n_dof}): {bad}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Per-episode validation loop
# ──────────────────────────────────────────────────────────────────────────────


def _build_batch(sample: dict, device: torch.device, task: str) -> dict:
    batch: dict = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
    batch["task"] = task
    batch["robot_type"] = "rby1"
    return batch


def _validate_episode(
    *,
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    dataset: LeRobotDataset,
    episode_idx: int,
    action_names: list[str],
    plot_indices: list[int],
    device: torch.device,
    fresh_chunk_every_step: bool,
    output_dir: Path,
) -> EpisodeResult:
    """Replay observations from one episode through the policy and score them."""
    ep_meta = dataset.meta.episodes[episode_idx]
    ep_from = int(ep_meta["dataset_from_index"])
    ep_to = int(ep_meta["dataset_to_index"])
    n_frames = ep_to - ep_from
    first_sample = dataset[ep_from]
    task_str = first_sample.get("task", "") or ""
    n_dof = len(action_names)

    print(f"  ep {episode_idx}: frames=[{ep_from}, {ep_to}) ({n_frames}), task={task_str!r}")

    gt = np.zeros((n_frames, n_dof), dtype=np.float32)
    pred = np.zeros((n_frames, n_dof), dtype=np.float32)

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, frame_idx in enumerate(range(ep_from, ep_to)):
            sample = dataset[frame_idx]
            gt[i] = sample["action"].numpy()

            if fresh_chunk_every_step:
                policy.reset()

            batch = _build_batch(sample, device, task_str)
            batch = preprocessor(batch)
            action = policy.select_action(batch)
            action = postprocessor(action)
            pred[i] = action.squeeze(0).float().cpu().numpy()
    elapsed = time.perf_counter() - t0

    err = pred - gt
    abs_err = np.abs(err)
    per_joint_mae = {n: float(abs_err[:, j].mean()) for j, n in enumerate(action_names)}
    per_joint_max = {n: float(abs_err[:, j].max()) for j, n in enumerate(action_names)}
    per_joint_p99 = {n: float(np.percentile(abs_err[:, j], 99)) for j, n in enumerate(action_names)}
    overall_mae = float(abs_err.mean())

    # Plot
    n_plot = len(plot_indices)
    ncols = min(4, n_plot)
    nrows = int(np.ceil(n_plot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), sharex=True, squeeze=False)
    t = np.arange(n_frames)
    flat = axes.flatten()
    for slot, j in enumerate(plot_indices):
        ax = flat[slot]
        ax.plot(t, gt[:, j], label="GT", linewidth=1.4)
        ax.plot(t, pred[:, j], label="Pred", linewidth=1.0, alpha=0.85, linestyle="--")
        ax.set_title(f"{action_names[j]} (MAE={per_joint_mae[action_names[j]]:.4f})", fontsize=9)
        ax.grid(True, alpha=0.3)
        if slot == 0:
            ax.legend(fontsize=8)
    for slot in range(n_plot, len(flat)):
        flat[slot].axis("off")
    fig.suptitle(f"Validation — ep {episode_idx} | overall MAE = {overall_mae:.4f}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path = output_dir / f"episode_{episode_idx:03d}.png"
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)

    return EpisodeResult(
        episode_idx=episode_idx,
        num_frames=n_frames,
        overall_mae=overall_mae,
        per_joint_mae=per_joint_mae,
        per_joint_max_err=per_joint_max,
        per_joint_p99_err=per_joint_p99,
        inference_seconds=elapsed,
        plot_path=str(plot_path),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cross-episode aggregation
# ──────────────────────────────────────────────────────────────────────────────


def _aggregate(results: list[EpisodeResult], action_names: list[str]) -> dict:
    succ = [r for r in results if r.error is None]
    if not succ:
        return {"num_episodes": 0, "successes": 0, "failures": len(results)}

    overall = np.array([r.overall_mae for r in succ])
    per_joint = {n: np.array([r.per_joint_mae[n] for r in succ]) for n in action_names}
    per_joint_max = {n: np.array([r.per_joint_max_err[n] for r in succ]) for n in action_names}

    return {
        "num_episodes": len(results),
        "successes": len(succ),
        "failures": len(results) - len(succ),
        "overall_mae_mean": float(overall.mean()),
        "overall_mae_std": float(overall.std()),
        "overall_mae_min": float(overall.min()),
        "overall_mae_max": float(overall.max()),
        "per_joint_mae_mean": {n: float(v.mean()) for n, v in per_joint.items()},
        "per_joint_mae_std": {n: float(v.std()) for n, v in per_joint.items()},
        "per_joint_max_err_max": {n: float(v.max()) for n, v in per_joint_max.items()},
        "per_episode_overall_mae": {r.episode_idx: r.overall_mae for r in succ},
    }


def _print_summary(agg: dict, action_names: list[str]) -> None:
    print()
    print("=" * 72)
    print(f"AGGREGATE — {agg['successes']}/{agg['num_episodes']} episodes succeeded")
    print("=" * 72)
    if agg["successes"] == 0:
        print("(no successful episodes)")
        return
    print(f"  overall MAE   mean={agg['overall_mae_mean']:.4f}  "
          f"std={agg['overall_mae_std']:.4f}  "
          f"range=[{agg['overall_mae_min']:.4f}, {agg['overall_mae_max']:.4f}]")
    print()
    print(f"  per-joint MAE (mean ± std across episodes):")
    for n in action_names:
        print(f"    {n:24s}  {agg['per_joint_mae_mean'][n]:8.4f}  ± {agg['per_joint_mae_std'][n]:7.4f}  "
              f"(worst-ep max-err {agg['per_joint_max_err_max'][n]:.4f})")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Path to .../checkpoints/<id>/pretrained_model")
    ap.add_argument("--dataset-repo-id", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--episodes", default=None,
                    help="Comma-separated indices, ranges allowed: '0,1,3-5'. Default: all.")
    ap.add_argument("--split-file", default=None,
                    help="Path to a split.json {'train':[...], 'val':[...]}; if set, uses --split-type.")
    ap.add_argument("--split-type", default="val", choices=["train", "val"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write per-ep metrics + plots. Default: temp_images/validation_<stamp>/")
    ap.add_argument("--plot-indices", default=None,
                    help="Comma-separated dim indices to include in plots. Default: all.")
    ap.add_argument("--task", default=None, help="Override task string if not present in dataset.")
    ap.add_argument("--fresh-chunk-every-step", action="store_true",
                    help="Reset policy state before every frame (open-loop per-step prediction).")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Output dir
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path("/data/objsearch/rby1_policy_learning/temp_images") / f"validation_{stamp}"
    else:
        out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Persist run config (small JSON dump of argv) for reproducibility.
    (out_root / "config.json").write_text(json.dumps(vars(args), indent=2))

    # Load dataset
    dataset = LeRobotDataset(repo_id=args.dataset_repo_id, root=args.dataset_root)
    all_episodes = sorted(int(e) for e in range(len(dataset.meta.episodes)))
    action_names: list[str] = dataset.features["action"]["names"]
    n_dof = len(action_names)

    # Resolve episode list
    if args.split_file is not None:
        episodes = _episodes_from_split(Path(args.split_file), args.split_type)
        # intersect with what's in the dataset (split files may reference more)
        episodes = [e for e in episodes if e in all_episodes]
    else:
        episodes = _parse_episodes_arg(args.episodes, all_episodes)

    plot_indices = _parse_plot_indices(args.plot_indices, n_dof)

    print("=" * 72)
    print(f"Validation run | output → {out_root}")
    print(f"  checkpoint          : {args.checkpoint}")
    print(f"  dataset             : {args.dataset_repo_id} ({args.dataset_root})")
    print(f"  episodes ({len(episodes):3d}) : {episodes if len(episodes) <= 20 else f'{episodes[:10]}...{episodes[-5:]}'}")
    print(f"  device              : {args.device}")
    print(f"  plot indices        : {plot_indices}")
    print(f"  fresh chunk per step: {args.fresh_chunk_every_step}")
    print(f"  action dims ({n_dof:2d})    : {action_names}")
    print("=" * 72)

    # Load policy
    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy_cfg.device = str(device)
    policy_cls = get_policy_class(policy_cfg.type)
    policy: PreTrainedPolicy = policy_cls.from_pretrained(args.checkpoint, config=policy_cfg)
    policy.to(device).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.checkpoint,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # Validate each episode
    results: list[EpisodeResult] = []
    for ep in episodes:
        try:
            res = _validate_episode(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                episode_idx=ep,
                action_names=action_names,
                plot_indices=plot_indices,
                device=device,
                fresh_chunk_every_step=args.fresh_chunk_every_step,
                output_dir=out_root,
            )
            (out_root / f"episode_{ep:03d}_metrics.json").write_text(json.dumps(res.to_dict(), indent=2))
            print(f"    overall MAE = {res.overall_mae:.4f}  ({res.inference_seconds:.1f}s)")
            results.append(res)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  ✗ ep {ep} FAILED: {e}\n{tb}", file=sys.stderr)
            results.append(EpisodeResult(
                episode_idx=ep, num_frames=0, overall_mae=float("nan"),
                per_joint_mae={n: float("nan") for n in action_names},
                per_joint_max_err={n: float("nan") for n in action_names},
                per_joint_p99_err={n: float("nan") for n in action_names},
                inference_seconds=0.0, error=str(e),
            ))

    # Aggregate + dump summary
    agg = _aggregate(results, action_names)
    _print_summary(agg, action_names)
    (out_root / "summary.json").write_text(json.dumps(
        {"aggregate": agg, "episodes": [r.to_dict() for r in results]}, indent=2,
    ))
    print(f"\n✓ Validation complete. Output: {out_root}")


if __name__ == "__main__":
    main()
