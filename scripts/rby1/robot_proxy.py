"""Robot proxy daemon (runs on the Jetson).

Owns the only connection to the RBY1 robot:
  - gRPC arm control via rby1_sdk (SDK commanding mode).
  - Dynamixel gripper bus via /dev/rby1_gripper.

Exposes the robot over ZMQ so a remote client (a workstation) can drive rollout
without needing direct access to either cable.

Protocol
--------
STATE PUB  — tcp://0.0.0.0:5560
  Publishes a JSON string at ~100 Hz:
    {"ts": <float>,
     "joints":   {"<joint_name>": <float>, ...},
     "grippers": {"<gripper_name>": <meters>, ...}}

ACTION PULL — tcp://0.0.0.0:5561
  Accepts a JSON string per message:
    {"ts": <float optional>,
     "joints":   {"<joint_name>": <float>, ...},
     "grippers": {"<gripper_name>": <meters>, ...}}

Action messages are applied via RBY1.send_action(), which handles:
  - arm joint commands via gRPC
  - gripper continuous-position commands via the existing 20 Hz DXL writer

Usage
-----
  python scripts/rby1/robot_proxy.py
  # or via the bash wrapper:
  scripts/rby1/start_robot_proxy.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from threading import Event, Thread

import zmq

from lerobot.robots.rby1.config_rby1 import RBY1Config
from lerobot.robots.rby1.robot_rby1 import RBY1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("robot_proxy")

shutdown = Event()


def _install_signal_handlers():
    def _handle(sig, _frame):
        logger.info(f"received signal {sig}, shutting down...")
        shutdown.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def state_publisher_loop(robot: RBY1, pub_socket: zmq.Socket, rate_hz: float) -> None:
    """Publish robot state at `rate_hz` on `pub_socket`."""
    period = 1.0 / rate_hz
    next_tick = time.perf_counter()
    joint_keys = list(robot._joints_dict.keys())
    gripper_keys = list(robot._gripper_keys.keys())

    logger.info(
        f"state publisher: {len(joint_keys)} joint dims + {len(gripper_keys)} grippers @ {rate_hz:.0f} Hz"
    )

    while not shutdown.is_set():
        try:
            # Pull current state (no cameras — proxy is configured with cameras={})
            with robot._state_lock:
                state = robot._latest_state
            robot._read_gripper_positions()

            joints_out = {}
            if state is not None:
                for key in joint_keys:
                    joints_out[key] = float(state[robot._joints_dict[key]])
            else:
                for key in joint_keys:
                    joints_out[key] = 0.0

            grippers_out = {}
            for key in gripper_keys:
                dxl_id = robot._gripper_keys[key]
                raw_enc = float(robot._gripper_positions[dxl_id])
                grippers_out[key] = robot._encoder_to_meters(raw_enc, dxl_id)

            msg = {
                "ts": time.perf_counter(),
                "joints": joints_out,
                "grippers": grippers_out,
            }
            pub_socket.send_string(json.dumps(msg), flags=zmq.NOBLOCK)
        except zmq.Again:
            # buffer full; skip this tick
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning(f"state publisher tick error: {e}")

        next_tick += period
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            # fell behind; skip ahead
            next_tick = time.perf_counter()


def _snapshot_right_side(robot: RBY1, timeout_s: float = 2.0) -> dict:
    """Capture the current right-arm joint positions + right-gripper meters.

    These values are merged into every incoming action message so the right
    side of the robot is actively held at the pose it was in at proxy
    startup, while the workstation only commands left-arm + left-gripper.
    """
    # Wait until the SDK state callback has populated _latest_state
    t0 = time.perf_counter()
    while robot._latest_state is None:
        if time.perf_counter() - t0 > timeout_s:
            raise TimeoutError(
                f"Robot state did not arrive within {timeout_s:.1f}s — cannot snapshot right side"
            )
        time.sleep(0.02)

    with robot._state_lock:
        state = list(robot._latest_state)

    # RBY1 state vector: indices 8..14 are right-arm joints 0..6
    right_arm = {
        f"right_arm_{i}.pos": float(state[8 + i])
        for i in range(7)
    }

    # Right gripper: read DXL encoder, convert to meters
    robot._read_gripper_positions()
    raw_enc = float(robot._gripper_positions[0])  # DXL id 0 = right gripper
    right_gripper_m = float(robot._encoder_to_meters(raw_enc, 0))

    return {"arm": right_arm, "gripper": right_gripper_m}


def action_consumer_loop(robot: RBY1, pull_socket: zmq.Socket,
                         frozen_right: dict | None = None) -> None:
    """Receive action messages and dispatch them to the robot.

    If `frozen_right` is provided, every incoming action is augmented (via
    setdefault, so the workstation can still override) with right-arm joint
    targets and a right-gripper meters target so the right side is held.
    """
    valid_keys = set(robot._joints_dict.keys()) | set(robot._gripper_keys.keys())
    logger.info(
        f"action consumer: accepting joint/gripper targets (keys: {sorted(valid_keys)[:4]}...)"
    )
    if frozen_right is not None:
        logger.info(
            f"action consumer: will hold right side at "
            f"arm={list(frozen_right['arm'].values())} gripper={frozen_right['gripper']:.4f}m"
        )
    tick = 0
    while not shutdown.is_set():
        try:
            raw = pull_socket.recv_string(flags=zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.002)
            continue
        except Exception as e:  # noqa: BLE001
            logger.warning(f"action consumer recv error: {e}")
            continue

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"action consumer: bad JSON ({e}); dropping")
            continue

        # Merge joints + grippers into the flat dict RBY1.send_action expects
        action: dict[str, float] = {}
        for k, v in msg.get("joints", {}).items():
            if k in valid_keys:
                action[k] = float(v)
        for k, v in msg.get("grippers", {}).items():
            if k in valid_keys:
                action[k] = float(v)

        # Hold the right side at its frozen pose unless the workstation
        # explicitly commanded those dims (16-D policies still work).
        if frozen_right is not None:
            for k, v in frozen_right["arm"].items():
                if k in valid_keys:
                    action.setdefault(k, v)
            if "right_gripper.pos" in valid_keys:
                action.setdefault("right_gripper.pos", frozen_right["gripper"])

        if not action:
            logger.warning("action consumer: empty action; dropping")
            continue

        try:
            robot.send_action(action)
            tick += 1
            if tick % 100 == 1:
                logger.info(f"action consumer tick {tick}: keys={len(action)} sample={next(iter(action.items()))}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"action consumer send error: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-address", default="192.168.30.1:50051")
    parser.add_argument("--state-port", type=int, default=5560)
    parser.add_argument("--action-port", type=int, default=5561)
    parser.add_argument("--state-rate-hz", type=float, default=100.0)
    parser.add_argument("--with-torso", action="store_true", default=False)
    parser.add_argument("--with-head", action="store_true", default=False)
    parser.add_argument("--gripper-current-cap", type=float, default=5.0)
    args = parser.parse_args()

    _install_signal_handlers()

    # Build RBY1 with NO cameras (cameras are published by start_zed_publisher.sh),
    # use_external_commands=False so the DXL bus + SDK commanding + gripper homing
    # all come up like a normal rollout would.
    config = RBY1Config(
        robot_address=args.robot_address,
        use_external_commands=False,
        with_torso=args.with_torso,
        with_head=args.with_head,
        with_right_arm=True,
        with_left_arm=True,
        with_right_gripper=True,
        with_left_gripper=True,
        gripper_current_cap=args.gripper_current_cap,
        cameras={},
    )

    logger.info("instantiating RBY1 (SDK commanding mode)...")
    robot = RBY1(config)
    logger.info("connecting to hardware (this includes gripper homing ~5–10 s)...")
    robot.connect()

    logger.info("snapshotting right-arm + right-gripper rest pose...")
    frozen_right = _snapshot_right_side(robot, timeout_s=2.0)
    logger.info(
        f"right arm frozen @ {[round(v, 3) for v in frozen_right['arm'].values()]} rad, "
        f"right gripper frozen @ {frozen_right['gripper']:.4f} m"
    )

    logger.info("robot ready; opening ZMQ sockets")

    ctx = zmq.Context.instance()

    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 10)  # drop stale messages if subscriber is slow
    pub.bind(f"tcp://0.0.0.0:{args.state_port}")
    logger.info(f"state PUB bound on tcp://0.0.0.0:{args.state_port}")

    pull = ctx.socket(zmq.PULL)
    pull.setsockopt(zmq.RCVHWM, 10)
    pull.bind(f"tcp://0.0.0.0:{args.action_port}")
    logger.info(f"action PULL bound on tcp://0.0.0.0:{args.action_port}")

    # Small delay so subscribers can connect before we start spamming
    time.sleep(0.5)

    state_thread = Thread(
        target=state_publisher_loop,
        args=(robot, pub, args.state_rate_hz),
        name="state_pub",
        daemon=True,
    )
    state_thread.start()

    try:
        action_consumer_loop(robot, pull, frozen_right=frozen_right)
    finally:
        logger.info("shutting down: draining + disconnecting...")
        shutdown.set()
        state_thread.join(timeout=2.0)
        try:
            pub.close(linger=0)
            pull.close(linger=0)
            ctx.term()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"robot disconnect error: {e}")
        logger.info("bye")


if __name__ == "__main__":
    main()
