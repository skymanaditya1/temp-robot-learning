"""Summary statistics + per-joint trajectory plot for a LeRobot RBY1 dataset.

Loads a LeRobotDataset, prints aggregate statistics and per-joint distribution
tables, and saves a 4x4 grid of per-joint trajectories (mean across episodes
with +/-1 std band) with episode phase normalized to [0, 1].

Example:
    python scripts/rby1/analyze_dataset.py \
        --dataset-name rby1_pick_v3_20260422_174437
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


FEATURE_CHOICES = ("observation.state", "action")


def _fmt_joint_row(name: str, vals: np.ndarray) -> str:
    q = np.quantile(vals, [0.1, 0.5, 0.9])
    return (
        f"  {name:22s} "
        f"min={vals.min():+.4f}  q10={q[0]:+.4f}  med={q[1]:+.4f}  "
        f"mean={vals.mean():+.4f}  q90={q[2]:+.4f}  max={vals.max():+.4f}  "
        f"std={vals.std():.4f}"
    )


def analyze(args):
    root = Path(args.datasets_root) / args.dataset_name
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    ds = LeRobotDataset(repo_id=f"local/{args.dataset_name}", root=str(root))

    fps = ds.meta.fps
    total_episodes = ds.num_episodes
    total_frames = len(ds)
    duration_s = total_frames / fps

    ep_meta = ds.meta.episodes
    ep_lengths = np.array(
        [int(ep_meta[i]["dataset_to_index"]) - int(ep_meta[i]["dataset_from_index"])
         for i in range(total_episodes)],
        dtype=np.int64,
    )

    print("=" * 70)
    print(f"Dataset: local/{args.dataset_name}")
    print(f"  fps                 : {fps}")
    print(f"  total episodes      : {total_episodes}")
    print(f"  total frames        : {total_frames}")
    print(f"  total duration      : {duration_s:.1f} s  ({duration_s/60:.2f} min)")
    print()
    print("Episode length (frames):")
    print(f"  min / median / mean / max : "
          f"{ep_lengths.min()} / {int(np.median(ep_lengths))} / "
          f"{ep_lengths.mean():.1f} / {ep_lengths.max()}")
    print(f"  min / median / mean / max : "
          f"{ep_lengths.min()/fps:.1f} s / {int(np.median(ep_lengths))/fps:.1f} s / "
          f"{ep_lengths.mean()/fps:.1f} s / {ep_lengths.max()/fps:.1f} s")
    print()

    # Task distribution
    try:
        tasks = [ep_meta[i]["tasks"] for i in range(total_episodes)]
        flat = [t if isinstance(t, str) else (t[0] if len(t) else "") for t in tasks]
        unique, counts = np.unique(flat, return_counts=True)
        print("Tasks:")
        for u, c in zip(unique, counts):
            print(f"  [{c:3d} ep]  {u}")
        print()
    except Exception as e:
        print(f"(could not read task distribution: {e})")
        print()

    # Load the chosen feature for every frame
    feature = args.feature
    names = ds.features[feature]["names"]
    n_dof = len(names)
    print(f"Per-joint stats ({feature}, {n_dof} DoF, all frames):")

    # Bulk load — dataset parquet read is fast; one row per frame.
    values = np.stack([ds[i][feature].numpy() for i in range(total_frames)])
    for j, name in enumerate(names):
        print(_fmt_joint_row(name, values[:, j]))
    print()

    # Plot: mean trajectory vs. normalized episode phase with +/-1 std band.
    n_phase = 100
    phase = np.linspace(0.0, 1.0, n_phase)
    # Per-episode interpolated trajectories: shape (n_episodes, n_phase, n_dof)
    per_ep = np.zeros((total_episodes, n_phase, n_dof), dtype=np.float32)
    for e in range(total_episodes):
        lo = int(ep_meta[e]["dataset_from_index"])
        hi = int(ep_meta[e]["dataset_to_index"])
        ep_len = hi - lo
        if ep_len < 2:
            per_ep[e] = values[lo]
            continue
        src_phase = np.linspace(0.0, 1.0, ep_len)
        ep_vals = values[lo:hi]
        for j in range(n_dof):
            per_ep[e, :, j] = np.interp(phase, src_phase, ep_vals[:, j])

    mean_traj = per_ep.mean(axis=0)
    std_traj = per_ep.std(axis=0)

    ncols = 4
    nrows = int(np.ceil(n_dof / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), sharex=True)
    axes = axes.flatten()
    for j, name in enumerate(names):
        ax = axes[j]
        # Thin per-episode traces behind the mean for a visual sense of spread.
        for e in range(total_episodes):
            ax.plot(phase, per_ep[e, :, j], color="gray", alpha=0.15, linewidth=0.5)
        ax.plot(phase, mean_traj[:, j], color="tab:blue", linewidth=1.5, label="mean")
        ax.fill_between(
            phase,
            mean_traj[:, j] - std_traj[:, j],
            mean_traj[:, j] + std_traj[:, j],
            color="tab:blue", alpha=0.2, label=r"$\pm 1\sigma$",
        )
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, loc="best")
    for j in range(n_dof, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        f"{args.dataset_name} — {feature}  "
        f"({total_episodes} episodes, {total_frames} frames)",
        fontsize=11,
    )
    fig.supxlabel("episode phase (0 → 1)", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    out = Path(args.out) if args.out else Path(
        f"/data/objsearch/rby1_policy_learning/temp_images/"
        f"dataset_summary_{args.dataset_name}_{feature}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Saved plot -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", required=True,
                   help="Folder name under --datasets-root (e.g. rby1_pick_v3_20260422_174437)")
    p.add_argument("--feature", default="observation.state", choices=FEATURE_CHOICES,
                   help="Which 16-D feature to summarize / plot")
    p.add_argument("--datasets-root",
                   default="/data/objsearch/rby1_policy_learning/datasets/local")
    p.add_argument("--out", default=None,
                   help="Output plot path (default auto-derived under temp_images/)")
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
