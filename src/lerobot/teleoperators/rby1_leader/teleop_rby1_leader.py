"""
RBY1 master arm teleoperator for LeRobot.

Reads joint positions from an RBY1 master arm device and returns them
as target actions for the follower robot.
"""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any

import numpy as np

from lerobot.robots.rby1.robot_rby1 import (
    RBY1_HEAD_JOINTS,
    RBY1_LEFT_ARM_JOINTS,
    RBY1_RIGHT_ARM_JOINTS,
    RBY1_TORSO_JOINTS,
)

from ..teleoperator import Teleoperator
from .config_rby1_leader import RBY1LeaderConfig

logger = logging.getLogger(__name__)


class RBY1Leader(Teleoperator):
    """RBY1 master arm teleoperator."""

    config_class = RBY1LeaderConfig
    name = "rby1_leader"

    def __init__(self, config: RBY1LeaderConfig):
        super().__init__(config)
        self.config = config

        self._robot = None
        self._is_connected = False

        # Thread-safe state from SDK callback
        self._state_lock = Lock()
        self._latest_state: np.ndarray | None = None

        # Gripper state
        self._gripper_bus = None
        self._gripper_positions = [0.0, 0.0]  # [right, left]

        # Build joint mapping
        self._joints_dict = self._build_joints_dict()

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

    @property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = dict.fromkeys(self._joints_dict.keys(), float)
        features.update(dict.fromkeys(self._gripper_keys.keys(), float))
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

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

    def connect(self, calibrate: bool = True) -> None:
        import rby1_sdk

        logger.info(f"Connecting to RBY1 leader at {self.config.robot_address}...")

        self._robot = rby1_sdk.create_robot_a(self.config.robot_address)
        if not self._robot.connect():
            raise ConnectionError(
                f"Failed to connect to RBY1 leader at {self.config.robot_address}"
            )

        # Power on
        if not self._robot.is_power_on(".*"):
            logger.info("Powering on leader...")
            if not self._robot.power_on(".*"):
                raise RuntimeError("Failed to power on leader")

        # Start state update callback
        self._robot.start_state_update(self._state_callback, self.config.state_update_hz)

        # Initialize gripper bus (read-only)
        if self.config.with_right_gripper or self.config.with_left_gripper:
            try:
                self._gripper_bus = rby1_sdk.DynamixelBus(rby1_sdk.upc.GripperDeviceName)
                self._gripper_bus.open_port()
                self._gripper_bus.set_baud_rate(self.config.gripper_baudrate)
                logger.info("Leader gripper bus opened (read-only)")
            except Exception as e:
                logger.warning(f"Failed to open leader gripper bus: {e}")
                self._gripper_bus = None

        self._is_connected = True

        # Wait for first state update
        timeout = 5.0
        start = time.time()
        while self._latest_state is None and time.time() - start < timeout:
            time.sleep(0.01)
        if self._latest_state is None:
            logger.warning("No state update received from leader within timeout")

        logger.info("RBY1 leader connected")

    def _state_callback(self, state) -> None:
        with self._state_lock:
            self._latest_state = state.position.copy()

    def _encoder_to_meters(self, raw_enc: float) -> float:
        """Convert raw Dynamixel encoder value to gripper width in meters (0.0–0.1m)."""
        enc_open = self.config.gripper_enc_open
        enc_closed = self.config.gripper_enc_closed
        # Linear interpolation: enc_open -> 0.1m (fully open), enc_closed -> 0.0m (fully closed)
        width_m = 0.1 * (raw_enc - enc_closed) / (enc_open - enc_closed)
        return float(np.clip(width_m, 0.0, 0.1))

    def _read_gripper_positions(self) -> None:
        if self._gripper_bus is None:
            return
        try:
            rv = self._gripper_bus.group_fast_sync_read_encoder([0, 1])
            if rv is not None:
                for dev_id, enc in rv:
                    if dev_id < 2:
                        self._gripper_positions[dev_id] = enc
        except Exception:
            pass

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()

        action: dict[str, float] = {}

        # Read joint positions from master arm
        with self._state_lock:
            state = self._latest_state

        if state is not None:
            for key, idx in self._joints_dict.items():
                action[key] = float(state[idx])
        else:
            for key in self._joints_dict:
                action[key] = 0.0

        # Read gripper positions and convert from raw encoder to meters (0.0–0.1m)
        self._read_gripper_positions()
        for key, dxl_id in self._gripper_keys.items():
            raw_enc = float(self._gripper_positions[dxl_id])
            action[key] = self._encoder_to_meters(raw_enc)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"RBY1 leader read action: {dt_ms:.1f}ms")

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Force feedback not supported on RBY1 master arm
        pass

    def disconnect(self) -> None:
        if not self._is_connected:
            return

        if self._gripper_bus is not None:
            try:
                self._gripper_bus.close_port()
            except Exception:
                pass
            self._gripper_bus = None

        if self._robot is not None:
            try:
                self._robot.disconnect()
            except Exception:
                pass
            self._robot = None

        self._is_connected = False
        logger.info("RBY1 leader disconnected")
