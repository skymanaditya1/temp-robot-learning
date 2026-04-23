"""Dump the per-frame camera images for one episode, both pre- and post-normalizer.

For a given episode, iterates every frame (or every --stride-th frame), runs the
policy's preprocessor pipeline, and writes out:

  <out_dir>/pre_norm/<cam>/frame-XXXXXX.png
      The raw image the dataset feeds to the preprocessor (CHW float in [0,1],
      converted to HWC uint8 PNG). This is what a human would see as a "photo".

  <out_dir>/post_norm/<cam>/frame-XXXXXX.png
      The normalized tensor the transformer actually ingests. Values live in
      roughly [-2.5, +2.5] after mean/std normalization; we linearly rescale
      that range to [0, 255] for viewability. Still channels-RGB, still uint8.

Both image sets are saved with PIL (RGB convention). Normalization stats are
loaded from the checkpoint so post_norm images reflect the exact tensor the
policy sees at inference time.

Example:
    python scripts/rby1/dump_episode_images.py \
        --dataset-name rby1_pick_v3_20260422_174437 \
        --episode 10 \
        --checkpoint /data/.../outputs/train/rby1_pick_v3_..._act_vega/checkpoints/last/pretrained_model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors


POST_NORM_RANGE = 3.0  # clip post-norm to [-3, +3] then rescale to [0, 255]


def _to_uint8_pre_norm(t: torch.Tensor) -> np.ndarray:
    """CHW float in [0, 1]  ->  HWC uint8."""
    arr = t.clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
    return arr


def _to_uint8_post_norm(t: torch.Tensor, half_range: float = POST_NORM_RANGE) -> np.ndarray:
    """CHW float in ~[-2.5, +2.5] -> HWC uint8 via [-half_range, +half_range] -> [0, 255]."""
    arr = t.clamp(-half_range, half_range)
    arr = (arr + half_range) / (2.0 * half_range)  # [0, 1]
    arr = arr.mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
    return arr


def _save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def run(args):
    device = torch.device(args.device)

    ds_root = Path(args.datasets_root) / args.dataset_name
    if not ds_root.is_dir():
        raise SystemExit(f"Dataset root not found: {ds_root}")
    ds = LeRobotDataset(repo_id=f"local/{args.dataset_name}", root=str(ds_root))

    ep = ds.meta.episodes[args.episode]
    lo = int(ep["dataset_from_index"])
    hi = int(ep["dataset_to_index"])
    ep_len = hi - lo
    print(f"Episode {args.episode}: frames [{lo}, {hi})  ({ep_len} frames)")

    # Build the preprocessor pipeline with exactly the stats the policy was trained
    # with. We don't need the policy model itself — just the preprocessor's
    # normalizer — but loading the config is the same pattern the overfit script uses.
    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy_cfg.device = str(device)
    pre, _post = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.checkpoint,
        dataset_stats=ds.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    pre.reset()

    # Resolve output dir
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"/data/objsearch/rby1_policy_learning/temp_images/episode_images/"
        f"{args.dataset_name}/ep{args.episode}"
    )
    print(f"Writing to: {out_dir}")

    cam_keys = [k for k in ds.features if k.startswith("observation.images.")]
    print(f"Cameras ({len(cam_keys)}): {[c.split('.')[-1] for c in cam_keys]}")

    saved = 0
    stride = max(1, args.stride)
    with torch.inference_mode():
        for local_i, frame_idx in enumerate(range(lo, hi)):
            if local_i % stride != 0:
                continue
            s = ds[frame_idx]

            # Pre-norm: dataset already gives CHW float in [0, 1].
            for cam_key in cam_keys:
                cam_name = cam_key.split(".")[-1]
                pre_arr = _to_uint8_pre_norm(s[cam_key])
                _save_png(pre_arr, out_dir / "pre_norm" / cam_name / f"frame-{local_i:06d}.png")

            # Post-norm: batch up, run preprocessor, then per-cam rescale.
            batch = {k: v.unsqueeze(0).to(device) for k, v in s.items() if isinstance(v, torch.Tensor)}
            batch["task"] = ""
            batch["robot_type"] = "rby1"
            b2 = pre(batch)
            for cam_key in cam_keys:
                cam_name = cam_key.split(".")[-1]
                post_arr = _to_uint8_post_norm(b2[cam_key][0])
                _save_png(post_arr, out_dir / "post_norm" / cam_name / f"frame-{local_i:06d}.png")

            saved += 1
            if local_i % 20 == 0:
                print(f"  frame {local_i}/{ep_len}")

    print(f"\nDone. Saved {saved} frames × 2 norms × {len(cam_keys)} cams "
          f"= {saved * 2 * len(cam_keys)} PNGs under {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--checkpoint", required=True,
                   help="Path to .../checkpoints/<id>/pretrained_model. Needed for normalizer stats.")
    p.add_argument("--datasets-root",
                   default="/data/objsearch/rby1_policy_learning/datasets/local")
    p.add_argument("--out-dir", default=None,
                   help="Defaults to temp_images/episode_images/<dataset_name>/ep<N>/")
    p.add_argument("--stride", type=int, default=1,
                   help="Save every Nth frame (default 1 — all frames)")
    p.add_argument("--device", default="cpu",
                   help="cpu is fine; this only runs preprocessing, not the policy")
    run(p.parse_args())


if __name__ == "__main__":
    main()
