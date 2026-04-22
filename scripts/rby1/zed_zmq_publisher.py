#!/usr/bin/env python3
"""
ZED ZMQ Publisher — streams RGB frames from ZED cameras over ZMQ
in the JSON format expected by LeRobot's ZMQCamera.

Supports both:
  - ZED stereo cameras (ZED 2, ZED 2i, ZED Mini) via USB — uses sl.Camera
  - ZED One cameras (ZED X One UHD/GS/HDR) via GMSL — uses sl.CameraOne

Cameras are identified by serial number and type (stereo or mono).

Protocol (JSON string over ZMQ PUB):
    {
        "timestamps": {"<camera_name>": <float>},
        "images": {"<camera_name>": "<base64-encoded-jpeg>"}
    }

Usage:
    # Single stereo camera (USB ZED 2i):
    python zed_zmq_publisher.py --cameras chest_cam:stereo:36747794:5555

    # Single mono camera (GMSL ZED One):
    python zed_zmq_publisher.py --cameras right_wrist:mono:314996496:5556

    # All 3 cameras:
    python zed_zmq_publisher.py \\
        --cameras \\
            chest_cam:stereo:36747794:5555 \\
            right_wrist_cam:mono:314996496:5556 \\
            left_wrist_cam:mono:311689153:5557

    # With custom resolution and FPS:
    python zed_zmq_publisher.py \\
        --cameras chest_cam:stereo:36747794:5555 \\
        --resolution hd1080 \\
        --fps 30
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass
from threading import Event, Thread

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

shutdown_event = Event()


def signal_handler(sig, frame):
    logger.info("Shutdown requested...")
    shutdown_event.set()


@dataclass
class CameraSpec:
    name: str
    cam_type: str  # "stereo" or "mono"
    serial: int
    port: int
    resolution: str | None = None  # optional per-camera override (e.g. "hd720"); None => use global --resolution


def parse_camera_spec(spec: str) -> CameraSpec:
    """Parse 'name:type:serial:port[:resolution]' string into CameraSpec.

    The optional 5th field lets a single camera override the global
    --resolution (e.g. the head cam capturing hd720 while ZED One wrist cams
    stay at hd1080, which is the only resolution they support).
    """
    parts = spec.split(":")
    if len(parts) not in (4, 5):
        raise ValueError(
            f"Camera spec must be 'name:type:serial:port[:resolution]' "
            f"(e.g., 'chest_cam:stereo:36747794:5555' or "
            f"'head_cam:stereo:32938613:5555:hd720'), got '{spec}'"
        )
    cam_type = parts[1].lower()
    if cam_type not in ("stereo", "mono"):
        raise ValueError(f"Camera type must be 'stereo' or 'mono', got '{cam_type}'")
    resolution = parts[4] if len(parts) == 5 else None
    return CameraSpec(
        name=parts[0],
        cam_type=cam_type,
        serial=int(parts[2]),
        port=int(parts[3]),
        resolution=resolution,
    )


def run_camera_publisher(
    camera_spec: CameraSpec,
    resolution: str,
    fps: int,
    jpeg_quality: int,
    depth_mode: str,
    resize_wh: tuple[int, int] | None = None,
):
    """Run a ZMQ publisher for a single ZED camera in its own thread."""
    import pyzed.sl as sl
    import zmq

    res_map = {
        "hd2k": sl.RESOLUTION.HD2K,
        "hd1200": sl.RESOLUTION.HD1200,
        "hd1080": sl.RESOLUTION.HD1080,
        "hd720": sl.RESOLUTION.HD720,
        "svga": sl.RESOLUTION.SVGA,
        "vga": sl.RESOLUTION.VGA,
        "auto": sl.RESOLUTION.AUTO,
    }

    if camera_spec.cam_type == "mono":
        # ZED One (GMSL) — use CameraOne + InitParametersOne
        zed = sl.CameraOne()
        init_params = sl.InitParametersOne()
        init_params.camera_resolution = res_map.get(resolution, sl.RESOLUTION.HD1080)
        init_params.camera_fps = fps
        init_params.set_from_serial_number(camera_spec.serial)
    else:
        # ZED stereo (USB) — use Camera + InitParameters
        depth_map = {
            "none": sl.DEPTH_MODE.NONE,
            "performance": sl.DEPTH_MODE.PERFORMANCE,
            "ultra": sl.DEPTH_MODE.ULTRA,
            "neural": sl.DEPTH_MODE.NEURAL,
        }
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = res_map.get(resolution, sl.RESOLUTION.HD1080)
        init_params.camera_fps = fps
        init_params.depth_mode = depth_map.get(depth_mode, sl.DEPTH_MODE.NONE)
        init_params.coordinate_units = sl.UNIT.METER
        init_params.set_from_serial_number(camera_spec.serial)

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        logger.error(
            f"[{camera_spec.name}] Failed to open ZED camera "
            f"(type={camera_spec.cam_type}, serial={camera_spec.serial}): {err}"
        )
        return

    cam_info = zed.get_camera_information()
    actual_res = cam_info.camera_configuration.resolution
    logger.info(
        f"[{camera_spec.name}] Opened {cam_info.camera_model} "
        f"(S/N {cam_info.serial_number}, {actual_res.width}x{actual_res.height} @ {fps}fps)"
    )

    # Pre-allocate ZED Mat
    image_mat = sl.Mat()
    runtime_params = sl.RuntimeParameters() if camera_spec.cam_type == "stereo" else None

    # Setup ZMQ publisher
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://0.0.0.0:{camera_spec.port}")
    logger.info(f"[{camera_spec.name}] Publishing on port {camera_spec.port}")

    # Give subscribers time to connect
    time.sleep(0.5)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    frame_count = 0
    last_log_time = time.time()

    # Mono cameras use VIEW.LEFT for the single image
    view = sl.VIEW.LEFT

    try:
        while not shutdown_event.is_set():
            err = zed.grab(runtime_params) if runtime_params is not None else zed.grab()
            if err != sl.ERROR_CODE.SUCCESS:
                if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                logger.warning(f"[{camera_spec.name}] Grab failed: {err}")
                time.sleep(0.01)
                continue

            # Retrieve RGB image
            zed.retrieve_image(image_mat, view)
            frame = image_mat.get_data()  # BGRA numpy array

            # Convert BGRA to BGR for JPEG encoding
            frame_bgr = frame[:, :, :3]

            # Center-crop to match the output aspect ratio, then resize. This avoids
            # the horizontal squash you'd get from forcing 16:9 source into a 4:3 target.
            if resize_wh is not None:
                h, w = frame_bgr.shape[:2]
                target_w = int(round(h * resize_wh[0] / resize_wh[1]))
                if target_w < w:
                    x_start = (w - target_w) // 2
                    frame_bgr = frame_bgr[:, x_start:x_start + target_w]
                elif target_w > w:
                    # Source narrower than target -> crop top/bottom instead.
                    target_h = int(round(w * resize_wh[1] / resize_wh[0]))
                    y_start = (h - target_h) // 2
                    frame_bgr = frame_bgr[y_start:y_start + target_h, :]
                frame_bgr = cv2.resize(frame_bgr, resize_wh, interpolation=cv2.INTER_AREA)

            # Encode as JPEG
            success, jpeg_buf = cv2.imencode(".jpg", frame_bgr, encode_params)
            if not success:
                logger.warning(f"[{camera_spec.name}] JPEG encode failed")
                continue

            # Build JSON message
            img_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")
            timestamp = time.time()

            message = json.dumps({
                "timestamps": {camera_spec.name: timestamp},
                "images": {camera_spec.name: img_b64},
            })

            socket.send_string(message)
            frame_count += 1

            # Log FPS periodically
            now = time.time()
            if now - last_log_time >= 10.0:
                elapsed = now - last_log_time
                fps_actual = frame_count / elapsed
                logger.info(f"[{camera_spec.name}] Publishing at {fps_actual:.1f} FPS")
                frame_count = 0
                last_log_time = now

    except Exception:
        logger.exception(f"[{camera_spec.name}] Error in publisher loop")
    finally:
        logger.info(f"[{camera_spec.name}] Shutting down...")
        zed.close()
        socket.close()
        context.term()


def main():
    parser = argparse.ArgumentParser(
        description="Publish ZED camera frames over ZMQ for LeRobot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        required=True,
        help=(
            "Camera specs as 'name:type:serial:port'. "
            "Type is 'stereo' (USB ZED 2/2i/Mini) or 'mono' (GMSL ZED One). "
            "Example: 'chest_cam:stereo:36747794:5555'"
        ),
    )
    parser.add_argument(
        "--resolution",
        default="hd1080",
        choices=["hd2k", "hd1200", "hd1080", "hd720", "svga", "vga", "auto"],
        help="Camera resolution (default: hd1080). Note: ZED One only supports hd1080/hd1200/auto.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Camera FPS (default: 30)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG encoding quality 0-100 (default: 90)",
    )
    parser.add_argument(
        "--depth-mode",
        default="none",
        choices=["none", "performance", "ultra", "neural"],
        help="ZED depth mode for stereo cameras (default: none). Ignored for mono cameras.",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Resize frames to WIDTH HEIGHT before publishing (e.g. --resize 640 480). Default: no resizing.",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    resize_wh = tuple(args.resize) if args.resize is not None else None
    if resize_wh is not None:
        logger.info(f"Resizing frames to {resize_wh[0]}x{resize_wh[1]} before publishing")

    camera_specs = [parse_camera_spec(s) for s in args.cameras]

    # Validate no duplicate ports
    ports = [cs.port for cs in camera_specs]
    if len(ports) != len(set(ports)):
        logger.error("Duplicate ports detected in camera specs")
        sys.exit(1)

    logger.info(f"Starting {len(camera_specs)} camera publisher(s)...")

    threads: list[Thread] = []
    for spec in camera_specs:
        cam_resolution = spec.resolution or args.resolution
        if spec.resolution is not None:
            logger.info(f"[{spec.name}] per-camera resolution override: {cam_resolution}")
        t = Thread(
            target=run_camera_publisher,
            args=(spec, cam_resolution, args.fps, args.jpeg_quality, args.depth_mode, resize_wh),
            daemon=True,
            name=f"zed_pub_{spec.name}",
        )
        t.start()
        threads.append(t)

    # Wait for shutdown
    try:
        while not shutdown_event.is_set():
            shutdown_event.wait(timeout=1.0)
    except KeyboardInterrupt:
        shutdown_event.set()

    logger.info("Waiting for publisher threads to finish...")
    for t in threads:
        t.join(timeout=5.0)

    logger.info("Done.")


if __name__ == "__main__":
    main()
