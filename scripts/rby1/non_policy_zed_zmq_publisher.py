#!/usr/bin/env python3
"""
Non-policy ZED ZMQ Publisher — streams RGB + right RGB + depth + intrinsics
from a stereo ZED camera over ZMQ, at the camera's native resolution.

Derived from zed_zmq_publisher.py but scoped to the non-policy pipeline (e.g.
rby1_standalone pick/place + drawer skills), where downstream code needs
per-frame depth and intrinsics to do pixel → 3D unprojection, plane fitting,
and point-cloud fusion.

Differences vs zed_zmq_publisher.py:
  - Stereo-only. Mono (ZED One) cameras are rejected.
  - Always runs a real depth mode (default: neural) and ships depth + right RGB
    + intrinsics on every frame.
  - Coordinate units are millimetres internally so depth goes on the wire
    without a per-frame metre→mm multiply.
  - No --resize flag — frames and intrinsics are always at native resolution so
    the published K matches the images exactly.

Protocol (JSON string over ZMQ PUB):
    {
        "timestamps":   {"<camera_name>": <float seconds>},
        "images":       {"<camera_name>": "<base64 JPEG, left RGB>"},
        "right_images": {"<camera_name>": "<base64 JPEG, right RGB>"},
        "depths":       {"<camera_name>": "<base64 PNG, uint16 millimetres>"},
        "intrinsics": {
            "<camera_name>": {
                "K_left":           [[fx,0,cx],[0,fy,cy],[0,0,1]],
                "K_right":          [[fx,0,cx],[0,fy,cy],[0,0,1]],
                "distortion_left":  [... 12 floats ...],
                "distortion_right": [... 12 floats ...],
                "baseline":         <float metres>
            }
        }
    }

Depth is float32 millimetres from the ZED SDK, NaN/inf → 0, clipped to
[0, 65535] and PNG-encoded so invalid pixels survive as zeros.

Usage:
    python non_policy_zed_zmq_publisher.py \\
        --cameras head_camera:stereo:36747794:5558
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
    cam_type: str  # must be "stereo" for this publisher
    serial: int
    port: int


def parse_camera_spec(spec: str) -> CameraSpec:
    """Parse 'name:type:serial:port' string into CameraSpec. Stereo only."""
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Camera spec must be 'name:type:serial:port' "
            f"(e.g., 'head_camera:stereo:36747794:5558'), got '{spec}'"
        )
    cam_type = parts[1].lower()
    if cam_type != "stereo":
        raise ValueError(
            f"non_policy_zed_zmq_publisher is stereo-only; got cam_type='{cam_type}'"
        )
    return CameraSpec(name=parts[0], cam_type=cam_type, serial=int(parts[2]), port=int(parts[3]))


# Publishes one ZMQ message per second regardless of the camera's internal
# grab rate. 1 Hz is downstream pipeline's sampling rate; the ZED's own fps
# only controls grab/depth cadence inside the SDK.
PUBLISH_HZ = 1.0

# ZED SDK rejects fps=1 at HD1080 (supported: 15/30/60). 15 is the lowest
# accepted value, which keeps grab latency low without burning extra GPU.
CAMERA_FPS = 15


def run_camera_publisher(
    camera_spec: CameraSpec,
    resolution: str,
    jpeg_quality: int,
    depth_mode: str,
):
    """Run a ZMQ publisher for a single stereo ZED camera in its own thread."""
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
    depth_map = {
        "performance": sl.DEPTH_MODE.PERFORMANCE,
        "ultra": sl.DEPTH_MODE.ULTRA,
        "neural": sl.DEPTH_MODE.NEURAL,
    }

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = res_map.get(resolution, sl.RESOLUTION.HD1080)
    init_params.camera_fps = CAMERA_FPS
    init_params.depth_mode = depth_map.get(depth_mode, sl.DEPTH_MODE.NEURAL)
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.depth_minimum_distance = 300.0  # mm
    init_params.set_from_serial_number(camera_spec.serial)

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        logger.error(
            f"[{camera_spec.name}] Failed to open ZED camera "
            f"(serial={camera_spec.serial}): {err}"
        )
        return

    cam_info = zed.get_camera_information()
    actual_res = cam_info.camera_configuration.resolution
    logger.info(
        f"[{camera_spec.name}] Opened {cam_info.camera_model} "
        f"(S/N {cam_info.serial_number}, {actual_res.width}x{actual_res.height}, "
        f"camera_fps={CAMERA_FPS}, publish_hz={PUBLISH_HZ}, depth={depth_mode})"
    )

    # Intrinsics at native resolution — published unchanged every frame.
    calib = cam_info.camera_configuration.calibration_parameters
    left_cam, right_cam = calib.left_cam, calib.right_cam
    intrinsics_block = {
        "K_left": [
            [float(left_cam.fx), 0.0, float(left_cam.cx)],
            [0.0, float(left_cam.fy), float(left_cam.cy)],
            [0.0, 0.0, 1.0],
        ],
        "K_right": [
            [float(right_cam.fx), 0.0, float(right_cam.cx)],
            [0.0, float(right_cam.fy), float(right_cam.cy)],
            [0.0, 0.0, 1.0],
        ],
        "distortion_left":  [float(v) for v in left_cam.disto],
        "distortion_right": [float(v) for v in right_cam.disto],
        "baseline":         float(calib.get_camera_baseline() / 1000.0),  # mm → m
    }
    logger.info(
        f"[{camera_spec.name}] Intrinsics: "
        f"fx={left_cam.fx:.1f} fy={left_cam.fy:.1f} "
        f"cx={left_cam.cx:.1f} cy={left_cam.cy:.1f} "
        f"baseline={intrinsics_block['baseline']:.4f} m"
    )

    image_mat = sl.Mat()
    right_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://0.0.0.0:{camera_spec.port}")
    logger.info(f"[{camera_spec.name}] Publishing on port {camera_spec.port}")

    time.sleep(0.5)

    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    frame_count = 0
    last_log_time = time.time()

    publish_period = 1.0 / PUBLISH_HZ
    next_deadline = time.time() + publish_period

    try:
        while not shutdown_event.is_set():
            err = zed.grab(runtime_params)
            if err != sl.ERROR_CODE.SUCCESS:
                if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                logger.warning(f"[{camera_spec.name}] Grab failed: {err}")
                time.sleep(0.01)
                continue

            # -- Left RGB --
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            left_bgr = image_mat.get_data()[:, :, :3]
            ok, left_buf = cv2.imencode(".jpg", left_bgr, jpeg_params)
            if not ok:
                logger.warning(f"[{camera_spec.name}] JPEG encode failed (left)")
                continue
            left_b64 = base64.b64encode(left_buf.tobytes()).decode("ascii")

            # -- Right RGB --
            zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
            right_bgr = right_mat.get_data()[:, :, :3]
            ok, right_buf = cv2.imencode(".jpg", right_bgr, jpeg_params)
            if not ok:
                logger.warning(f"[{camera_spec.name}] JPEG encode failed (right)")
                continue
            right_b64 = base64.b64encode(right_buf.tobytes()).decode("ascii")

            # -- Depth (uint16 mm, PNG) --
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth_mm_f = depth_mat.get_data()
            depth_mm_f = np.nan_to_num(depth_mm_f, nan=0.0, posinf=0.0, neginf=0.0)
            depth_u16 = np.clip(depth_mm_f, 0.0, 65535.0).astype(np.uint16)
            ok, depth_buf = cv2.imencode(".png", depth_u16, png_params)
            if not ok:
                logger.warning(f"[{camera_spec.name}] PNG encode failed (depth)")
                continue
            depth_b64 = base64.b64encode(depth_buf.tobytes()).decode("ascii")

            timestamp = time.time()
            message = {
                "timestamps":   {camera_spec.name: timestamp},
                "images":       {camera_spec.name: left_b64},
                "right_images": {camera_spec.name: right_b64},
                "depths":       {camera_spec.name: depth_b64},
                "intrinsics":   {camera_spec.name: intrinsics_block},
            }
            socket.send_string(json.dumps(message))
            frame_count += 1

            now = time.time()
            if now - last_log_time >= 10.0:
                elapsed = now - last_log_time
                fps_actual = frame_count / elapsed
                logger.info(f"[{camera_spec.name}] Publishing at {fps_actual:.1f} FPS")
                frame_count = 0
                last_log_time = now

            sleep_for = next_deadline - time.time()
            if sleep_for > 0:
                if shutdown_event.wait(timeout=sleep_for):
                    break
                next_deadline += publish_period
            else:
                next_deadline = time.time() + publish_period

    except Exception:
        logger.exception(f"[{camera_spec.name}] Error in publisher loop")
    finally:
        logger.info(f"[{camera_spec.name}] Shutting down...")
        zed.close()
        socket.close()
        context.term()


def main():
    parser = argparse.ArgumentParser(
        description="Publish ZED stereo frames + depth + intrinsics over ZMQ (non-policy pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        required=True,
        help=(
            "Stereo camera specs as 'name:stereo:serial:port'. "
            "Example: 'head_camera:stereo:36747794:5558'"
        ),
    )
    parser.add_argument(
        "--resolution",
        default="hd1080",
        choices=["hd2k", "hd1200", "hd1080", "hd720", "svga", "vga", "auto"],
        help="Camera resolution (default: hd1080).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG encoding quality 0-100 (default: 90).",
    )
    parser.add_argument(
        "--depth-mode",
        default="neural",
        choices=["performance", "ultra", "neural"],
        help="ZED depth mode (default: neural).",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    camera_specs = [parse_camera_spec(s) for s in args.cameras]

    ports = [cs.port for cs in camera_specs]
    if len(ports) != len(set(ports)):
        logger.error("Duplicate ports detected in camera specs")
        sys.exit(1)

    logger.info(f"Starting {len(camera_specs)} stereo camera publisher(s)...")

    threads: list[Thread] = []
    for spec in camera_specs:
        t = Thread(
            target=run_camera_publisher,
            args=(spec, args.resolution, args.jpeg_quality, args.depth_mode),
            daemon=True,
            name=f"zed_pub_{spec.name}",
        )
        t.start()
        threads.append(t)

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
