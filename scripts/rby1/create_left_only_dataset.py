"""Create a derived LeRobotDataset that keeps only left-arm + left-gripper dims.

Drops `right_arm_0..6.pos` (indices 0..6) and `right_gripper.pos` (index 14)
from both `observation.state` and `action`, shrinking their feature dimension
from 16 to 8. Videos, timestamps, episode/frame indices, and tasks are unchanged.

Example:
    python scripts/rby1/create_left_only_dataset.py \
        --src /data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_v3_20260422_174437 \
        --dst /data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_v3_20260422_174437_left_only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DROPPED_JOINT_NAMES = {
    "right_arm_0.pos",
    "right_arm_1.pos",
    "right_arm_2.pos",
    "right_arm_3.pos",
    "right_arm_4.pos",
    "right_arm_5.pos",
    "right_arm_6.pos",
    "right_gripper.pos",
}

VECTOR_FEATURES = ("observation.state", "action")
STAT_KEYS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


def _compute_keep_indices(names: list[str]) -> list[int]:
    """Return indices of `names` that are NOT in DROPPED_JOINT_NAMES."""
    return [i for i, n in enumerate(names) if n not in DROPPED_JOINT_NAMES]


def _slice_array(arr: list | np.ndarray, keep: list[int]) -> list:
    return [arr[i] for i in keep]


def _patch_info_json(path: Path, keep_by_feature: dict[str, list[int]],
                     kept_names_by_feature: dict[str, list[str]]) -> None:
    info = json.loads(path.read_text())
    for feat in VECTOR_FEATURES:
        if feat not in info["features"]:
            continue
        keep = keep_by_feature[feat]
        ft = info["features"][feat]
        ft["shape"] = [len(keep)]
        ft["names"] = kept_names_by_feature[feat]
    path.write_text(json.dumps(info, indent=4))


def _patch_stats_json(path: Path, keep_by_feature: dict[str, list[int]]) -> None:
    stats = json.loads(path.read_text())
    for feat in VECTOR_FEATURES:
        if feat not in stats:
            continue
        keep = keep_by_feature[feat]
        for sk in STAT_KEYS:
            if sk in stats[feat]:
                stats[feat][sk] = _slice_array(stats[feat][sk], keep)
        # count stays unchanged
    path.write_text(json.dumps(stats, indent=4))


def _patch_data_parquet(path: Path, keep_by_feature: dict[str, list[int]]) -> None:
    table = pq.read_table(path)
    new_cols: dict[str, pa.Array] = {}
    for col_name in table.column_names:
        if col_name in VECTOR_FEATURES:
            keep = keep_by_feature[col_name]
            # Each row is a list-of-float; slice to the kept indices.
            original = table[col_name].to_pylist()
            sliced = [[row[i] for i in keep] for row in original]
            new_cols[col_name] = pa.array(sliced, type=pa.list_(pa.float32()))
        else:
            new_cols[col_name] = table[col_name]
    new_table = pa.table(new_cols)
    pq.write_table(new_table, path)


def _patch_episodes_parquet(path: Path, keep_by_feature: dict[str, list[int]]) -> None:
    table = pq.read_table(path)
    new_cols: dict[str, pa.Array] = {}
    for col_name in table.column_names:
        # Inline per-episode stats are named like 'stats/<feat>/<stat>' (e.g. 'stats/action/mean').
        # Each cell is a 16-length list we need to slice to 8 for the two vector features.
        sliced = False
        for feat in VECTOR_FEATURES:
            prefix = f"stats/{feat}/"
            if col_name.startswith(prefix):
                stat_name = col_name[len(prefix):]
                if stat_name == "count":
                    break  # count is scalar, don't slice
                keep = keep_by_feature[feat]
                original = table[col_name].to_pylist()
                # Some rows may already be None; guard.
                new_list = []
                for row in original:
                    if row is None:
                        new_list.append(None)
                    else:
                        new_list.append([row[i] for i in keep])
                new_cols[col_name] = pa.array(new_list, type=pa.list_(pa.float32()))
                sliced = True
                break
        if not sliced:
            new_cols[col_name] = table[col_name]
    new_table = pa.table(new_cols)
    pq.write_table(new_table, path)


def _sanity_check(src_root: Path, dst_root: Path,
                  keep_by_feature: dict[str, list[int]]) -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as e:  # noqa: BLE001
        print(f"[warn] cannot import LeRobotDataset ({e}); skipping structural sanity check")
        return

    src_ds = LeRobotDataset(repo_id=f"local/{src_root.name}", root=str(src_root))
    dst_ds = LeRobotDataset(repo_id=f"local/{dst_root.name}", root=str(dst_root))

    assert len(src_ds) == len(dst_ds), f"length mismatch: {len(src_ds)} vs {len(dst_ds)}"
    assert src_ds.num_episodes == dst_ds.num_episodes

    # Feature schema
    for feat in VECTOR_FEATURES:
        dst_shape = tuple(dst_ds.features[feat]["shape"])
        dst_names = dst_ds.features[feat]["names"]
        keep = keep_by_feature[feat]
        assert dst_shape == (len(keep),), f"{feat} dst shape {dst_shape} != ({len(keep)},)"
        src_names = src_ds.features[feat]["names"]
        expected_names = [src_names[i] for i in keep]
        assert dst_names == expected_names, f"{feat} names mismatch:\n  dst={dst_names}\n  exp={expected_names}"

    # Bitwise equivalence of a sample (first and last frame)
    import torch
    for idx in (0, len(src_ds) - 1):
        for feat in VECTOR_FEATURES:
            src_val = src_ds[idx][feat]
            dst_val = dst_ds[idx][feat]
            keep = keep_by_feature[feat]
            expected = src_val[keep] if isinstance(src_val, torch.Tensor) else np.asarray(src_val)[keep]
            if not torch.equal(dst_val, expected if isinstance(expected, torch.Tensor) else torch.tensor(expected)):
                raise SystemExit(f"{feat}[{idx}] mismatch after slicing")

    print(f"[sanity] len={len(dst_ds)}, episodes={dst_ds.num_episodes}, "
          f"observation.state shape={tuple(dst_ds.features['observation.state']['shape'])}, "
          f"action shape={tuple(dst_ds.features['action']['shape'])}")
    print(f"[sanity] kept action names: {dst_ds.features['action']['names']}")


def convert(src_root: Path, dst_root: Path, overwrite: bool) -> None:
    if not src_root.is_dir():
        sys.exit(f"src not found: {src_root}")
    if dst_root.exists():
        if not overwrite:
            sys.exit(f"dst already exists (use --overwrite): {dst_root}")
        print(f"[1/6] Removing existing destination {dst_root}")
        shutil.rmtree(dst_root)

    print(f"[1/6] Copying {src_root} -> {dst_root} (full tree including videos)")
    shutil.copytree(src_root, dst_root)

    # Read source info to get the feature-name order → derive keep indices per feature.
    src_info = json.loads((src_root / "meta" / "info.json").read_text())
    keep_by_feature: dict[str, list[int]] = {}
    kept_names_by_feature: dict[str, list[str]] = {}
    for feat in VECTOR_FEATURES:
        names = src_info["features"][feat]["names"]
        keep = _compute_keep_indices(names)
        keep_by_feature[feat] = keep
        kept_names_by_feature[feat] = [names[i] for i in keep]
    print(f"[2/6] Kept indices for both {VECTOR_FEATURES}: {keep_by_feature['observation.state']}")
    print(f"       (len={len(keep_by_feature['observation.state'])} dims)")

    print("[3/6] Patching meta/info.json (shape + names)")
    _patch_info_json(dst_root / "meta" / "info.json", keep_by_feature, kept_names_by_feature)

    print("[4/6] Patching meta/stats.json (global stat arrays)")
    _patch_stats_json(dst_root / "meta" / "stats.json", keep_by_feature)

    print("[5/6] Patching data/chunk-000/file-000.parquet (per-frame rows)")
    for pq_file in (dst_root / "data").rglob("*.parquet"):
        print(f"       - {pq_file.relative_to(dst_root)}")
        _patch_data_parquet(pq_file, keep_by_feature)

    print("[5/6] Patching meta/episodes/chunk-000/file-000.parquet (per-episode inline stats)")
    for pq_file in (dst_root / "meta" / "episodes").rglob("*.parquet"):
        print(f"       - {pq_file.relative_to(dst_root)}")
        _patch_episodes_parquet(pq_file, keep_by_feature)

    print("[6/6] Sanity checks")
    _sanity_check(src_root, dst_root, keep_by_feature)
    print(f"\nDone. New dataset at: {dst_root}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    convert(args.src.resolve(), args.dst.resolve(), args.overwrite)


if __name__ == "__main__":
    main()
