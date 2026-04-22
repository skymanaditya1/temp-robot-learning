"""Grab a single frame from a ZED ZMQ publisher and verify channel order.

Writes two PNGs to compare:
  - <out_dir>/zmq_raw_bgr.png     — `cv2.imdecode` output, no conversion (BGR order)
  - <out_dir>/zmq_fixed_rgb.png   — output after `cv2.cvtColor(BGR2RGB)` (what the
                                    patched ZmqCamera consumer now emits)

If the publisher + consumer fix are working, the "fixed" PNG should display
with correct colors and the "raw" PNG should look channel-swapped.

Example:
    python scripts/rby1/verify_zmq_rgb.py                 # head_cam @ 5555
    python scripts/rby1/verify_zmq_rgb.py --port 5556     # right_wrist_cam
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import zmq
from PIL import Image


def grab_frame(address: str, port: int, camera_name: str, timeout_ms: int = 5000) -> bytes:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVHWM, 1)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.connect(f"tcp://{address}:{port}")

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    events = dict(poller.poll(timeout_ms))
    if sock not in events:
        raise RuntimeError(f"No ZMQ message on tcp://{address}:{port} within {timeout_ms}ms")

    msg = sock.recv_string()
    data = json.loads(msg)
    images = data.get("images", {})
    if camera_name in images:
        img_b64 = images[camera_name]
    elif images:
        img_b64 = next(iter(images.values()))
    else:
        raise RuntimeError("No 'images' key in received message")

    sock.close()
    ctx.term()
    return base64.b64decode(img_b64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--address", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument(
        "--camera-name",
        default="head_cam",
        help="Camera name key expected in the ZMQ payload (e.g. head_cam, left_wrist_cam, right_wrist_cam)",
    )
    p.add_argument("--out-dir", default="/data/objsearch/rby1_policy_learning/temp_images")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to tcp://{args.address}:{args.port} (camera={args.camera_name})")
    jpeg_bytes = grab_frame(args.address, args.port, args.camera_name)
    print(f"Got JPEG of size {len(jpeg_bytes)} bytes")

    # Raw decode — returns BGR (OpenCV convention).
    raw_bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if raw_bgr is None:
        raise RuntimeError("cv2.imdecode failed")

    # Emulate the patched consumer.
    fixed_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    # Per-channel means. Remember: in the raw BGR array, channel 0 == B, 2 == R.
    raw_means = raw_bgr.reshape(-1, 3).mean(axis=0)
    fixed_means = fixed_rgb.reshape(-1, 3).mean(axis=0)
    print()
    print(f"  raw (BGR order)    [B, G, R] means: {raw_means.round(2).tolist()}")
    print(f"  fixed (RGB order)  [R, G, B] means: {fixed_means.round(2).tolist()}")
    print()

    # Save via PIL, which interprets the input array as RGB. This makes the
    # visual comparison unambiguous:
    #   - Feeding the raw (BGR-ordered) array to PIL mislabels channels -> the
    #     saved PNG shows reds<->blues swapped.
    #   - Feeding the fixed (RGB-ordered) array to PIL matches PIL's convention
    #     -> the saved PNG shows correct colors.
    raw_path = out_dir / "zmq_raw_bgr.png"
    fixed_path = out_dir / "zmq_fixed_rgb.png"
    Image.fromarray(raw_bgr).save(raw_path)
    Image.fromarray(fixed_rgb).save(fixed_path)
    print(f"Saved (via PIL, which treats the input array as RGB):")
    print(f"  {raw_path}     (raw BGR array fed to PIL -> should look channel-swapped)")
    print(f"  {fixed_path}   (fixed RGB array fed to PIL -> should look correct)")
    print()
    print("Tip: point the camera at something clearly red or blue for an obvious")
    print("visual difference between the two PNGs.")


if __name__ == "__main__":
    main()
