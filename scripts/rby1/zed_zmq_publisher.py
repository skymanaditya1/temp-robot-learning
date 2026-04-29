#!/usr/bin/env python3
"""
ZED ZMQ Publisher — streams ZED camera frames over ZMQ in JSON.

Supports two per-camera profiles:

  - profile=rgb   (default): publishes left RGB JPEG only. Works for both ZED
                  stereo (USB) and ZED One mono (GMSL). Optional resize.
                  Message: {timestamps, images}.

  - profile=rgbd:           publishes left RGB JPEG + right RGB JPEG +
                  depth (uint16 mm, PNG) + intrinsics, at native resolution
                  (no resize). Stereo-only. Used by the non-policy pipeline
                  for pixel -> 3D unprojection.
                  Message: {timestamps, images, right_images, depths, intrinsics}.

Cameras are identified by serial number and type (stereo or mono).

Camera spec format:
    name:type:serial:port[:resolution[:profile]]

  - resolution (optional): per-camera override of --resolution (e.g. hd720).
  - profile    (optional): "rgb" (default) or "rgbd".

The --publish-hz flag throttles publishing to a fixed rate (e.g. 1 Hz for
the non-policy pipeline). If unset, the publisher emits one message per
successful grab.

Usage:
    # Policy preset: 3 cams, all rgb, publish every grab, resized to 640x480.
    python zed_zmq_publisher.py \\
        --cameras \\
            head_cam:stereo:32938613:5555:hd720 \\
            right_wrist_cam:mono:306470987:5556 \\
            left_wrist_cam:mono:301119863:5557 \\
        --resolution hd1080 --fps 10 --resize 640 480

    # Non-policy preset: 1 stereo cam, rgbd, publish 1 Hz, native res.
    python zed_zmq_publisher.py \\
        --cameras head_camera:stereo:32938613:5558:hd1080:rgbd \\
        --depth-mode neural --publish-hz 1
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


VALID_PROFILES = ("rgb", "rgbd")


@dataclass
class CameraSpec:
    name: str
    cam_type: str  # "stereo" or "mono"
    serial: int
    port: int
    resolution: str | None = None  # optional per-camera override of global --resolution
    profile: str = "rgb"            # "rgb" or "rgbd"


def parse_camera_spec(spec: str) -> CameraSpec:
    """Parse 'name:type:serial:port[:resolution[:profile]]' into CameraSpec."""
    parts = spec.split(":")
    if len(parts) < 4 or len(parts) > 6:
        raise ValueError(
            f"Camera spec must be 'name:type:serial:port[:resolution[:profile]]' "
            f"(e.g., 'head_cam:stereo:32938613:5555' or "
            f"'head_camera:stereo:32938613:5558:hd1080:rgbd'), got '{spec}'"
        )
    cam_type = parts[1].lower()
    if cam_type not in ("stereo", "mono"):
        raise ValueError(f"Camera type must be 'stereo' or 'mono', got '{cam_type}'")

    resolution = parts[4] if len(parts) >= 5 and parts[4] else None
    profile = (parts[5].lower() if len(parts) == 6 else "rgb")
    if profile not in VALID_PROFILES:
        raise ValueError(f"Profile must be one of {VALID_PROFILES}, got '{profile}'")
    if profile == "rgbd" and cam_type != "stereo":
        raise ValueError(
            f"profile='rgbd' requires cam_type='stereo' (got '{cam_type}' for '{parts[0]}')"
        )

    return CameraSpec(
        name=parts[0],
        cam_type=cam_type,
        serial=int(parts[2]),
        port=int(parts[3]),
        resolution=resolution,
        profile=profile,
    )


def run_camera_publisher(
    camera_spec: CameraSpec,
    resolution: str,
    fps: int,
    jpeg_quality: int,
    depth_mode: str,
    depth_min_distance_mm: float,
    publish_hz: float | None,
    resize_wh: tuple[int, int] | None = None,
):
    """Run a ZMQ publisher for a single ZED camera in its own thread.

    Behavior depends on camera_spec.profile:
      - rgb : retrieve left RGB only, optional resize, publish {images}.
      - rgbd: retrieve left+right RGB + depth, native res, publish
              {images, right_images, depths, intrinsics}.
    """
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
        "none": sl.DEPTH_MODE.NONE,
        "performance": sl.DEPTH_MODE.PERFORMANCE,
        "ultra": sl.DEPTH_MODE.ULTRA,
        "neural": sl.DEPTH_MODE.NEURAL,
    }

    profile = camera_spec.profile
    is_rgbd = profile == "rgbd"

    if camera_spec.cam_type == "mono":
        # ZED One (GMSL) — use CameraOne + InitParametersOne. rgb-only.
        zed = sl.CameraOne()
        init_params = sl.InitParametersOne()
        init_params.camera_resolution = res_map.get(resolution, sl.RESOLUTION.HD1080)
        init_params.camera_fps = fps
        init_params.set_from_serial_number(camera_spec.serial)
    else:
        # ZED stereo (USB) — use Camera + InitParameters.
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = res_map.get(resolution, sl.RESOLUTION.HD1080)
        init_params.camera_fps = fps
        if is_rgbd:
            init_params.depth_mode = depth_map.get(depth_mode, sl.DEPTH_MODE.NEURAL)
            init_params.coordinate_units = sl.UNIT.MILLIMETER
            init_params.depth_minimum_distance = depth_min_distance_mm
        else:
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
        f"(S/N {cam_info.serial_number}, {actual_res.width}x{actual_res.height} @ {fps}fps, "
        f"profile={profile})"
    )

    # rgbd: publish intrinsics on every message (cheap; same matrices each frame).
    intrinsics_block = None
    if is_rgbd:
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
            "baseline":         float(calib.get_camera_baseline() / 1000.0),  # mm -> m
        }
        logger.info(
            f"[{camera_spec.name}] Intrinsics: "
            f"fx={left_cam.fx:.1f} fy={left_cam.fy:.1f} "
            f"cx={left_cam.cx:.1f} cy={left_cam.cy:.1f} "
            f"baseline={intrinsics_block['baseline']:.4f} m"
        )

    image_mat = sl.Mat()
    right_mat = sl.Mat() if is_rgbd else None
    depth_mat = sl.Mat() if is_rgbd else None
    runtime_params = sl.RuntimeParameters() if camera_spec.cam_type == "stereo" else None

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://0.0.0.0:{camera_spec.port}")
    logger.info(f"[{camera_spec.name}] Publishing on port {camera_spec.port}")

    time.sleep(0.5)

    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    frame_count = 0
    last_log_time = time.time()

    publish_period = (1.0 / publish_hz) if publish_hz and publish_hz > 0 else None
    next_deadline = (time.time() + publish_period) if publish_period else None

    # Mono cameras use VIEW.LEFT for the single image
    view_left = sl.VIEW.LEFT
    view_right = sl.VIEW.RIGHT if is_rgbd else None

    try:
        while not shutdown_event.is_set():
            err = zed.grab(runtime_params) if runtime_params is not None else zed.grab()
            if err != sl.ERROR_CODE.SUCCESS:
                if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                logger.warning(f"[{camera_spec.name}] Grab failed: {err}")
                time.sleep(0.01)
                continue

            # -- Left RGB --
            zed.retrieve_image(image_mat, view_left)
            left_bgr = image_mat.get_data()[:, :, :3]  # BGRA -> BGR

            # rgb-only: optional center-crop + resize to keep aspect.
            if not is_rgbd and resize_wh is not None:
                h, w = left_bgr.shape[:2]
                target_w = int(round(h * resize_wh[0] / resize_wh[1]))
                if target_w < w:
                    x_start = (w - target_w) // 2
                    left_bgr = left_bgr[:, x_start:x_start + target_w]
                elif target_w > w:
                    target_h = int(round(w * resize_wh[1] / resize_wh[0]))
                    y_start = (h - target_h) // 2
                    left_bgr = left_bgr[y_start:y_start + target_h, :]
                left_bgr = cv2.resize(left_bgr, resize_wh, interpolation=cv2.INTER_AREA)

            ok, left_buf = cv2.imencode(".jpg", left_bgr, jpeg_params)
            if not ok:
                logger.warning(f"[{camera_spec.name}] JPEG encode failed (left)")
                continue
            left_b64 = base64.b64encode(left_buf.tobytes()).decode("ascii")

            timestamp = time.time()
            message: dict = {
                "timestamps": {camera_spec.name: timestamp},
                "images":     {camera_spec.name: left_b64},
            }

            if is_rgbd:
                # -- Right RGB --
                zed.retrieve_image(right_mat, view_right)
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

                message["right_images"] = {camera_spec.name: right_b64}
                message["depths"]       = {camera_spec.name: depth_b64}
                message["intrinsics"]   = {camera_spec.name: intrinsics_block}

            socket.send_string(json.dumps(message))
            frame_count += 1

            now = time.time()
            if now - last_log_time >= 10.0:
                elapsed = now - last_log_time
                fps_actual = frame_count / elapsed
                logger.info(f"[{camera_spec.name}] Publishing at {fps_actual:.1f} FPS")
                frame_count = 0
                last_log_time = now

            # Throttle publish rate if --publish-hz was set.
            if publish_period is not None:
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
        description="Publish ZED camera frames over ZMQ for LeRobot (rgb and rgbd profiles)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        required=True,
        help=(
            "Camera specs as 'name:type:serial:port[:resolution[:profile]]'. "
            "Type is 'stereo' (USB ZED 2/2i/Mini) or 'mono' (GMSL ZED One). "
            "Profile is 'rgb' (default) or 'rgbd' (stereo-only)."
        ),
    )
    parser.add_argument(
        "--resolution",
        default="hd1080",
        choices=["hd2k", "hd1200", "hd1080", "hd720", "svga", "vga", "auto"],
        help="Default camera resolution (default: hd1080). Per-camera spec can override.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Camera grab fps passed to ZED SDK (default: 30). May be coerced by SDK.",
    )
    parser.add_argument(
        "--publish-hz",
        type=float,
        default=None,
        help="Optional throttle: publish at this Hz (decoupled from grab rate). "
             "Default: publish on every successful grab.",
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
        help="ZED depth mode (rgbd profile only; default: none -> neural for rgbd cams). "
             "Ignored for mono and rgb-profile cameras.",
    )
    parser.add_argument(
        "--depth-min-distance-mm",
        type=float,
        default=300.0,
        help="Minimum depth distance in mm for rgbd cameras (default: 300).",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Resize rgb-profile frames to WIDTH HEIGHT before publishing "
             "(e.g. --resize 640 480). Ignored for rgbd profile (which always "
             "publishes native res so intrinsics K matches images).",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    resize_wh = tuple(args.resize) if args.resize is not None else None
    if resize_wh is not None:
        logger.info(f"Resizing rgb-profile frames to {resize_wh[0]}x{resize_wh[1]}")
    if args.publish_hz is not None:
        logger.info(f"Throttling publish rate to {args.publish_hz} Hz")

    camera_specs = [parse_camera_spec(s) for s in args.cameras]

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
        # rgbd defaults depth_mode to neural if user left it as 'none'.
        depth_mode_for_cam = args.depth_mode
        if spec.profile == "rgbd" and depth_mode_for_cam == "none":
            depth_mode_for_cam = "neural"
        t = Thread(
            target=run_camera_publisher,
            args=(
                spec,
                cam_resolution,
                args.fps,
                args.jpeg_quality,
                depth_mode_for_cam,
                args.depth_min_distance_mm,
                args.publish_hz,
                resize_wh,
            ),
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
