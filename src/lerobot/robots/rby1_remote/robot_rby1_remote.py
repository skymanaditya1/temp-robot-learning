"""Remote RBY1 adapter.

Runs on a workstation. Presents a standard `Robot` interface (so
`lerobot-record` / `lerobot-eval` can use it unmodified), but all hardware
I/O is forwarded to a `robot_proxy.py` daemon running on the Jetson via
ZMQ. Cameras are read directly from the Jetson's ZED ZMQ publisher via
`ZMQCamera`.

Feature schema intentionally mirrors the real `RBY1` robot so datasets and
policies trained on `rby1` work with `rby1_remote` interchangeably.
"""

from __future__ import annotations

import json
import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import zmq

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation

from ..robot import Robot
from ..rby1.robot_rby1 import (
    RBY1_HEAD_JOINTS,
    RBY1_LEFT_ARM_JOINTS,
    RBY1_RIGHT_ARM_JOINTS,
    RBY1_TORSO_JOINTS,
)
from .config_rby1_remote import RBY1RemoteConfig

logger = logging.getLogger(__name__)


class RBY1Remote(Robot):
    """RBY1 seen through a ZMQ bridge."""

    config_class = RBY1RemoteConfig
    name = "rby1_remote"

    def __init__(self, config: RBY1RemoteConfig):
        super().__init__(config)
        self.config = config

        self._ctx: zmq.Context | None = None
        self._state_sub: zmq.Socket | None = None
        self._action_push: zmq.Socket | None = None
        self._state_thread: Thread | None = None
        self._state_stop = Event()

        self._latest_state_lock = Lock()
        self._latest_state: dict[str, Any] | None = None  # last received state dict
        self._latest_state_ts_local: float | None = None  # perf_counter on this host when received

        self._is_connected = False

        self.cameras = make_cameras_from_configs(config.cameras)

        self._joints_dict: dict[str, int] = self._build_joints_dict()

    def _build_joints_dict(self) -> dict[str, int]:
        joints: dict[str, int] = {}
        if self.config.with_torso:
            joints.update(RBY1_TORSO_JOINTS)
        if self.config.with_right_arm:
            joints.update(RBY1_RIGHT_ARM_JOINTS)
        if self.config.with_left_arm:
            joints.update(RBY1_LEFT_ARM_JOINTS)
        if self.config.with_head:
            joints.update(RBY1_HEAD_JOINTS)
        return joints

    @property
    def _gripper_keys(self) -> dict[str, int]:
        keys: dict[str, int] = {}
        if self.config.with_right_gripper:
            keys["right_gripper.pos"] = 0
        if self.config.with_left_gripper:
            keys["left_gripper.pos"] = 1
        return keys

    # ------------------------------------------------------------------
    # Robot interface: features
    # ------------------------------------------------------------------

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self.motors_features, **self.camera_features}

    @property
    def action_features(self) -> dict[str, type]:
        return self.motors_features

    @property
    def camera_features(self) -> dict[str, tuple[int | None, int | None, int]]:
        return {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras}

    @property
    def motors_features(self) -> dict[str, type]:
        features: dict[str, type] = dict.fromkeys(self._joints_dict.keys(), float)
        features.update(dict.fromkeys(self._gripper_keys.keys(), float))
        return features

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self, calibrate: bool = False) -> None:  # noqa: ARG002
        logger.info(
            f"Connecting to RBY1 proxy at {self.config.jetson_host} "
            f"(state={self.config.state_port}, action={self.config.action_port})"
        )
        self._ctx = zmq.Context.instance()

        self._state_sub = self._ctx.socket(zmq.SUB)
        self._state_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self._state_sub.setsockopt(zmq.RCVHWM, 10)
        self._state_sub.setsockopt(zmq.CONFLATE, 1)  # keep only latest
        self._state_sub.connect(f"tcp://{self.config.jetson_host}:{self.config.state_port}")

        self._action_push = self._ctx.socket(zmq.PUSH)
        self._action_push.setsockopt(zmq.SNDHWM, 10)
        self._action_push.setsockopt(zmq.LINGER, 0)
        self._action_push.connect(f"tcp://{self.config.jetson_host}:{self.config.action_port}")

        # Start the state-subscriber thread
        self._state_stop.clear()
        self._state_thread = Thread(
            target=self._state_sub_loop, name="rby1_remote_state_sub", daemon=True
        )
        self._state_thread.start()

        # Wait up to 3 s for the first state message to arrive
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 3.0:
            with self._latest_state_lock:
                if self._latest_state is not None:
                    break
            time.sleep(0.02)
        with self._latest_state_lock:
            if self._latest_state is None:
                raise ConnectionError(
                    f"No state received from proxy at {self.config.jetson_host}:{self.config.state_port} "
                    f"within 3s — is robot_proxy.py running?"
                )
        logger.info("State stream live")

        # Connect cameras (already-running ZED ZMQ publishers)
        for cam_key, cam in self.cameras.items():
            logger.info(f"Connecting {cam_key}...")
            cam.connect()
        logger.info("RBY1Remote connected")
        self._is_connected = True

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        logger.info("Disconnecting RBY1Remote...")

        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"camera disconnect error: {e}")

        self._state_stop.set()
        if self._state_thread is not None:
            self._state_thread.join(timeout=2.0)
            self._state_thread = None

        try:
            if self._state_sub is not None:
                self._state_sub.close(linger=0)
            if self._action_push is not None:
                self._action_push.close(linger=0)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"socket close error: {e}")

        self._state_sub = None
        self._action_push = None
        self._is_connected = False
        logger.info("RBY1Remote disconnected")

    # ------------------------------------------------------------------
    # State subscription thread
    # ------------------------------------------------------------------

    def _state_sub_loop(self) -> None:
        assert self._state_sub is not None
        poller = zmq.Poller()
        poller.register(self._state_sub, zmq.POLLIN)
        while not self._state_stop.is_set():
            events = dict(poller.poll(200))
            if self._state_sub not in events:
                continue
            try:
                raw = self._state_sub.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            except Exception as e:  # noqa: BLE001
                logger.warning(f"state sub recv error: {e}")
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning(f"state sub: bad JSON ({e}); dropping")
                continue
            with self._latest_state_lock:
                self._latest_state = msg
                self._latest_state_ts_local = time.perf_counter()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self) -> RobotObservation:
        if not self._is_connected:
            raise ConnectionError("RBY1Remote is not connected")

        with self._latest_state_lock:
            state = self._latest_state
            state_local_ts = self._latest_state_ts_local
        if state is None or state_local_ts is None:
            raise RuntimeError("RBY1Remote: no state received yet")

        age_ms = (time.perf_counter() - state_local_ts) * 1000.0
        if age_ms > self.config.state_max_age_ms:
            raise TimeoutError(
                f"RBY1Remote: last robot state is {age_ms:.1f} ms old "
                f"(max allowed: {self.config.state_max_age_ms} ms). Proxy stalled?"
            )

        obs: dict[str, Any] = {}

        joints = state.get("joints", {}) or {}
        grippers = state.get("grippers", {}) or {}
        for key in self._joints_dict.keys():
            obs[key] = float(joints.get(key, 0.0))
        for key in self._gripper_keys.keys():
            obs[key] = float(grippers.get(key, 0.0))

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()

        return obs

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._is_connected:
            raise ConnectionError("RBY1Remote is not connected")

        joints_out: dict[str, float] = {}
        grippers_out: dict[str, float] = {}
        for key, val in action.items():
            if key in self._joints_dict:
                joints_out[key] = float(val)
            elif key in self._gripper_keys:
                grippers_out[key] = float(val)

        if not joints_out and not grippers_out:
            # Nothing applicable to send; return as-is
            return action

        msg = {
            "ts": time.perf_counter(),
            "joints": joints_out,
            "grippers": grippers_out,
        }
        try:
            assert self._action_push is not None
            self._action_push.send_string(json.dumps(msg), flags=zmq.NOBLOCK)
        except zmq.Again:
            logger.warning("action push: buffer full (proxy slow?); dropping this command")

        return action
