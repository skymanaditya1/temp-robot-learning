"""Overfit test: run a trained RBY1 policy on a training episode and plot
predicted vs. ground-truth actions for every joint (7L + 7R arm + 2 grippers).

Example:
    python scripts/rby1/overfit_inference_plot.py \
        --checkpoint /data/objsearch/rby1_policy_learning/outputs/train/act_rby1_pick_v2_20260419_180507/checkpoints/last/pretrained_model \
        --dataset-repo-id local/rby1_pick_v2_20260419_180507 \
        --dataset-root /data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_v2_20260419_180507 \
        --episode 0 \
        --out /tmp/overfit_ep0.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import PreTrainedPolicy, make_pre_post_processors
from lerobot.policies.factory import get_policy_class


def build_batch(sample: dict, device: torch.device, task: str) -> dict:
    """Convert a LeRobotDataset sample into a batched dict for policy inference."""
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(device)
    batch["task"] = task
    batch["robot_type"] = "rby1"
    return batch


def run(args):
    device = torch.device(args.device)

    dataset = LeRobotDataset(repo_id=args.dataset_repo_id, root=args.dataset_root)

    ep_idx = args.episode
    ep_meta = dataset.meta.episodes[ep_idx]
    ep_from = int(ep_meta["dataset_from_index"])
    ep_to = int(ep_meta["dataset_to_index"])
    first_sample = dataset[ep_from]
    task_str = args.task or first_sample.get("task", "") or ""

    print(f"Episode {ep_idx}: frames [{ep_from}, {ep_to}), task={task_str!r}")

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

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    action_names = dataset.features["action"]["names"]
    n_dof = len(action_names)

    gt_actions = np.zeros((ep_to - ep_from, n_dof), dtype=np.float32)
    pred_actions = np.zeros((ep_to - ep_from, n_dof), dtype=np.float32)

    with torch.inference_mode():
        for i, frame_idx in enumerate(range(ep_from, ep_to)):
            sample = dataset[frame_idx]
            gt_actions[i] = sample["action"].numpy()

            if args.fresh_chunk_every_step:
                policy.reset()

            batch = build_batch(sample, device, task_str)
            batch = preprocessor(batch)
            action = policy.select_action(batch)
            action = postprocessor(action)
            pred_actions[i] = action.squeeze(0).float().cpu().numpy()

            if i % 20 == 0:
                print(f"  frame {i}/{ep_to - ep_from}")

    # Per-joint MAE summary
    mae = np.abs(pred_actions - gt_actions).mean(axis=0)
    print("\nPer-joint MAE:")
    for name, e in zip(action_names, mae):
        print(f"  {name:22s} {e:.5f}")
    print(f"Overall MAE: {mae.mean():.5f}")

    # Plot
    ncols = 4
    nrows = int(np.ceil(n_dof / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), sharex=True)
    axes = axes.flatten()
    t = np.arange(ep_to - ep_from)
    for j, name in enumerate(action_names):
        ax = axes[j]
        ax.plot(t, gt_actions[:, j], label="GT", linewidth=1.4)
        ax.plot(t, pred_actions[:, j], label="Pred", linewidth=1.0, alpha=0.85, linestyle="--")
        ax.set_title(f"{name}  (MAE={mae[j]:.4f})", fontsize=9)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    for j in range(n_dof, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        f"Overfit check — episode {ep_idx} | ckpt={Path(args.checkpoint).parent.parent.name}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if args.out is None:
        out = Path(
            f"/data/objsearch/rby1_policy_learning/temp_images/overfit_ep{args.episode}.png"
        )
    else:
        out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nSaved plot -> {out}")
    if args.show:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to .../checkpoints/<id>/pretrained_model")
    p.add_argument("--dataset-repo-id", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--task", default=None, help="Override task string if not in dataset metadata")
    p.add_argument(
        "--fresh-chunk-every-step",
        action="store_true",
        help="Call policy.reset() each frame so every prediction is the first step of a fresh chunk "
             "(true open-loop per-step prediction). Default re-uses ACT's internal action queue.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output image path. If omitted, saves to "
             "/data/objsearch/rby1_policy_learning/temp_images/overfit_ep{episode}.png",
    )
    p.add_argument("--show", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
