"""Create a copy of a LeRobotDataset with BGR-as-RGB videos converted to true RGB.

Datasets recorded via the ZMQ camera consumer before the BGR->RGB fix in
`src/lerobot/cameras/zmq/camera_zmq.py` stored channel-swapped pixels in the
mp4 files. This script produces a new dataset directory with the channels
swapped so that videos are correctly ordered RGB, while leaving all joint
state, actions, and metadata untouched.

Example:
    python scripts/rby1/convert_dataset_bgr_to_rgb.py \
        --src /data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_v2_20260419_180507 \
        --dst /data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_v2_20260419_180507_rgb
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import av
import numpy as np


def _should_ignore_videos(_src, names):
    return [n for n in names if n == "videos"]


def _count_frames(mp4_path: Path) -> int:
    with av.open(str(mp4_path)) as c:
        s = c.streams.video[0]
        # `frames` is often 0 for streams without a count; fall back to manual count.
        if s.frames:
            return int(s.frames)
        n = 0
        for _ in c.decode(video=0):
            n += 1
        return n


def _reencode_swap_channels(src_path: Path, dst_path: Path) -> tuple[int, int]:
    """Decode `src_path`, swap R<->B per frame, encode to `dst_path`.

    Returns (src_frame_count, dst_frame_count).
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    src = av.open(str(src_path))
    src_stream = src.streams.video[0]
    decoder_name = src_stream.codec_context.name  # decoder name (e.g. 'libdav1d')
    codec_tag = getattr(src_stream.codec_context, "codec_tag", "") or ""
    pix_fmt = src_stream.codec_context.pix_fmt  # e.g. 'yuv420p'
    width = src_stream.codec_context.width
    height = src_stream.codec_context.height
    rate = src_stream.average_rate  # Fraction (e.g. 10/1)

    # Detect AV1 via the stored codec_tag ('av01') since the decoder name can
    # be 'libdav1d' rather than 'av1'. Prefer libsvtav1 (what lerobot records
    # with), then libaom-av1, then the generic name.
    is_av1 = "av01" in str(codec_tag).lower() or "av1" in decoder_name.lower() or "dav1d" in decoder_name.lower()
    if is_av1:
        encoder_candidates = ["libsvtav1", "libaom-av1", "av1"]
    else:
        encoder_candidates = [decoder_name]

    last_err: Exception | None = None
    for encoder_name in encoder_candidates:
        try:
            dst = av.open(str(dst_path), mode="w")
            out_stream = dst.add_stream(encoder_name, rate=rate)
            out_stream.width = width
            out_stream.height = height
            out_stream.pix_fmt = pix_fmt
            # Match the CRF lerobot uses by default (30) for libsvtav1.
            if encoder_name == "libsvtav1":
                out_stream.options = {"crf": "30", "g": "2"}
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            try:
                dst.close()
            except Exception:
                pass
            if (Path(str(dst_path))).exists():
                Path(str(dst_path)).unlink()
            continue
    else:
        raise RuntimeError(
            f"Could not open any encoder from {encoder_candidates} for {dst_path}: {last_err}"
        )

    src_count = 0
    dst_count = 0
    for frame in src.decode(video=0):
        src_count += 1
        arr = frame.to_ndarray(format="rgb24")  # (H,W,3) in RGB byte order
        # Swap R<->B. The array PyAV gave us is "correct RGB" in memory, but
        # the pixels recorded into the source mp4 were actually BGR-interpreted-
        # as-RGB, so swapping here produces true RGB.
        arr = np.ascontiguousarray(arr[:, :, ::-1])
        new_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in out_stream.encode(new_frame):
            dst.mux(packet)
            dst_count += 1 if packet.size else 0  # packets aren't frames; recount below

    # Flush encoder
    for packet in out_stream.encode():
        dst.mux(packet)

    dst.close()
    src.close()

    dst_count = _count_frames(dst_path)
    return src_count, dst_count


def convert(src_root: Path, dst_root: Path, overwrite: bool) -> None:
    if not src_root.is_dir():
        sys.exit(f"src does not exist: {src_root}")
    if dst_root.exists():
        if not overwrite:
            sys.exit(f"dst already exists (use --overwrite to replace): {dst_root}")
        shutil.rmtree(dst_root)

    print(f"[1/5] Copying non-video contents {src_root} -> {dst_root}")
    shutil.copytree(src_root, dst_root, ignore=_should_ignore_videos)

    print("[2/5] Re-encoding videos with R<->B swap")
    src_videos = src_root / "videos"
    dst_videos = dst_root / "videos"
    dst_videos.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(src_videos.rglob("*.mp4"))
    if not mp4s:
        sys.exit(f"No .mp4 files found under {src_videos}")

    total_src = 0
    total_dst = 0
    for src_mp4 in mp4s:
        rel = src_mp4.relative_to(src_videos)
        dst_mp4 = dst_videos / rel
        print(f"    {rel}")
        s, d = _reencode_swap_channels(src_mp4, dst_mp4)
        total_src += s
        total_dst += d
        if s != d:
            sys.exit(f"Frame count mismatch for {rel}: src={s}, dst={d}")
        print(f"      frames: {s} ok")

    print(f"  total frames encoded: {total_dst} (src total: {total_src})")

    print("[3/5] Updating meta/info.json")
    info_path = dst_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    old_repo_id = info.get("repo_id")
    new_repo_id = f"local/{dst_root.name}"
    info["repo_id"] = new_repo_id
    info_path.write_text(json.dumps(info, indent=4))
    print(f"    repo_id: {old_repo_id} -> {new_repo_id}")

    print("[4/5] Swapping channel order in meta/stats.json image stats")
    stats_path = dst_root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stat_keys_to_swap = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
    for feat_name, feat_stats in stats.items():
        if not feat_name.startswith("observation.images."):
            continue
        for sk in stat_keys_to_swap:
            if sk in feat_stats and len(feat_stats[sk]) == 3:
                feat_stats[sk] = list(reversed(feat_stats[sk]))
        print(f"    {feat_name}: swapped {stat_keys_to_swap}")
    stats_path.write_text(json.dumps(stats, indent=4))

    print("[5/5] Sanity checks")
    _sanity_check(src_root, dst_root)
    print(f"Done. New dataset at: {dst_root}")


def _sanity_check(src_root: Path, dst_root: Path) -> None:
    # Frame-count parity per mp4
    for src_mp4 in sorted((src_root / "videos").rglob("*.mp4")):
        rel = src_mp4.relative_to(src_root / "videos")
        dst_mp4 = dst_root / "videos" / rel
        s = _count_frames(src_mp4)
        d = _count_frames(dst_mp4)
        if s != d:
            sys.exit(f"[sanity] frame count mismatch {rel}: src={s}, dst={d}")

    # Per-channel mean of first frame should flip R<->B
    any_cam = next((src_root / "videos").rglob("*.mp4"))
    rel = any_cam.relative_to(src_root / "videos")
    dst_cam = dst_root / "videos" / rel
    with av.open(str(any_cam)) as c:
        s_frame = next(c.decode(video=0)).to_ndarray(format="rgb24")
    with av.open(str(dst_cam)) as c:
        d_frame = next(c.decode(video=0)).to_ndarray(format="rgb24")
    src_means = s_frame.reshape(-1, 3).mean(axis=0)
    dst_means = d_frame.reshape(-1, 3).mean(axis=0)
    print(f"    src RGB means: {src_means.round(2).tolist()}")
    print(f"    dst RGB means: {dst_means.round(2).tolist()}   (should be src reversed)")
    # Allow small codec-roundtrip drift
    drift = np.abs(dst_means - src_means[::-1]).max()
    if drift > 2.0:
        sys.exit(f"[sanity] channel-swap check failed: max drift {drift:.2f} > 2.0")

    # Load via LeRobotDataset to confirm the dataset is still valid.
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] could not import LeRobotDataset for final check: {e}")
        return
    ds_src = LeRobotDataset(repo_id=f"local/{src_root.name}", root=str(src_root))
    ds_dst = LeRobotDataset(repo_id=f"local/{dst_root.name}", root=str(dst_root))
    assert len(ds_src) == len(ds_dst), f"len mismatch {len(ds_src)} vs {len(ds_dst)}"
    assert ds_src.num_episodes == ds_dst.num_episodes
    # Actions and state should be bit-identical.
    s0 = ds_src[0]
    d0 = ds_dst[0]
    import torch

    assert torch.equal(s0["action"], d0["action"]), "action tensors differ"
    assert torch.equal(s0["observation.state"], d0["observation.state"]), "state tensors differ"
    print(f"    LeRobotDataset ok: len={len(ds_dst)}, episodes={ds_dst.num_episodes}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    convert(args.src.resolve(), args.dst.resolve(), args.overwrite)


if __name__ == "__main__":
    main()
