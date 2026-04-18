"""
Rainbow Robotics RBY1 robot driver for LeRobot.

Uses rby1_sdk for gRPC communication and Dynamixel SDK for gripper control.
State is read via a background callback from the SDK; commands are sent as
joint position commands via the ComponentBasedCommand builder.

Reference implementations:
    - /data/objsearch/rby1_standalone/record.py (state callback)
    - /data/objsearch/rby1_standalone/rby1_server_simple.py (command building)
    - /data/objsearch/rby1_standalone/gripper_controller.py (Dynamixel grippers)
"""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs

try:
    from lerobot.types import RobotAction, RobotObservation
except ImportError:
    from lerobot.processor import RobotAction, RobotObservation

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_rby1 import RBY1Config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Joint name mappings: {lerobot_key: state_vector_index}
# State vector from rby1_sdk: indices [0:2] base, [2:8] torso, [8:15] right_arm,
# [15:22] left_arm, [22:24] head.
# --------------------------------------------------------------------------- #

RBY1_TORSO_JOINTS = {
    "torso_0.pos": 2,
    "torso_1.pos": 3,
    "torso_2.pos": 4,
    "torso_3.pos": 5,
    "torso_4.pos": 6,
    "torso_5.pos": 7,
}

RBY1_RIGHT_ARM_JOINTS = {
    "right_arm_0.pos": 8,
    "right_arm_1.pos": 9,
    "right_arm_2.pos": 10,
    "right_arm_3.pos": 11,
    "right_arm_4.pos": 12,
    "right_arm_5.pos": 13,
    "right_arm_6.pos": 14,
}

RBY1_LEFT_ARM_JOINTS = {
    "left_arm_0.pos": 15,
    "left_arm_1.pos": 16,
    "left_arm_2.pos": 17,
    "left_arm_3.pos": 18,
    "left_arm_4.pos": 19,
    "left_arm_5.pos": 20,
    "left_arm_6.pos": 21,
}

RBY1_HEAD_JOINTS = {
    "head_0.pos": 22,
    "head_1.pos": 23,
}

RBY1_GRIPPER_KEYS = {
    "right_gripper.pos": 0,  # Dynamixel ID 0
    "left_gripper.pos": 1,   # Dynamixel ID 1
}


class RBY1(Robot):
    """Rainbow Robotics RBY1 dual-arm mobile manipulator."""

    config_class = RBY1Config
    name = "rby1"

    def __init__(self, config: RBY1Config):
        super().__init__(config)
        self.config = config

        self._robot = None  # rby1_sdk robot handle
        self._is_connected = False

        # Thread-safe state from SDK callback
        self._state_lock = Lock()
        self._latest_state: np.ndarray | None = None

        # Gripper state (read via Dynamixel bus)
        self._gripper_bus = None
        self._gripper_positions = [0.0, 0.0]  # [right, left]

        # Cameras
        self.cameras = make_cameras_from_configs(config.cameras)

        # Build joint mapping based on config
        self._joints_dict: dict[str, int] = self._build_joints_dict()

        self.logs: dict[str, float] = {}

    def _build_joints_dict(self) -> dict[str, int]:
        """Build the active joint name -> state vector index mapping."""
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
        """Active gripper keys."""
        keys: dict[str, int] = {}
        if self.config.with_right_gripper:
            keys["right_gripper.pos"] = 0
        if self.config.with_left_gripper:
            keys["left_gripper.pos"] = 1
        return keys

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

    def connect(self, calibrate: bool = False) -> None:
        import rby1_sdk

        logger.info(f"Connecting to RBY1 at {self.config.robot_address}...")

        self._robot = rby1_sdk.create_robot_a(self.config.robot_address)
        if not self._robot.connect():
            raise ConnectionError(
                f"Failed to connect to RBY1 at {self.config.robot_address}"
            )

        # Power on
        if not self._robot.is_power_on(".*"):
            logger.info("Powering on robot...")
            if not self._robot.power_on(".*"):
                raise RuntimeError("Failed to power on robot")

        # Start state update callback
        self._robot.start_state_update(self._state_callback, self.config.state_update_hz)
        logger.info(f"State updates started at {self.config.state_update_hz} Hz")

        mode = "external (send_action is a no-op)" if self.config.use_external_commands else "SDK commanding"
        logger.info(f"Command mode: {mode}")

        # Initialize gripper bus (read-only)
        if self.config.with_right_gripper or self.config.with_left_gripper:
            try:
                self._gripper_bus = rby1_sdk.DynamixelBus(rby1_sdk.upc.GripperDeviceName)
                self._gripper_bus.open_port()
                self._gripper_bus.set_baud_rate(self.config.gripper_baudrate)
                logger.info("Gripper bus opened (read-only)")
            except Exception as e:
                logger.warning(f"Failed to open gripper bus: {e}. Gripper readings will be zero.")
                self._gripper_bus = None

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True
        logger.info("RBY1 connected successfully")

        # Wait for first state update
        timeout = 5.0
        start = time.time()
        while self._latest_state is None and time.time() - start < timeout:
            time.sleep(0.01)
        if self._latest_state is None:
            logger.warning("No state update received within timeout")

    def _state_callback(self, state) -> None:
        """Called by rby1_sdk at state_update_hz. Thread-safe."""
        with self._state_lock:
            self._latest_state = state.position.copy()

    def _encoder_to_meters(self, raw_enc: float) -> float:
        """Convert raw Dynamixel encoder value to gripper width in meters (0.0–0.1m)."""
        enc_open = self.config.gripper_enc_open
        enc_closed = self.config.gripper_enc_closed
        width_m = 0.1 * (raw_enc - enc_closed) / (enc_open - enc_closed)
        return float(np.clip(width_m, 0.0, 0.1))

    def _read_gripper_positions(self) -> None:
        """Read gripper encoder positions from Dynamixel bus."""
        if self._gripper_bus is None:
            return
        try:
            rv = self._gripper_bus.group_fast_sync_read_encoder([0, 1])
            if rv is not None:
                for dev_id, enc in rv:
                    if dev_id < 2:
                        self._gripper_positions[dev_id] = enc
        except Exception:
            pass  # Gripper read failures are non-critical

    def get_observation(self) -> RobotObservation:
        obs_dict: RobotObservation = {}

        before_read_t = time.perf_counter()

        # Read joint state
        with self._state_lock:
            state = self._latest_state

        if state is not None:
            for key, idx in self._joints_dict.items():
                obs_dict[key] = float(state[idx])
        else:
            for key in self._joints_dict:
                obs_dict[key] = 0.0

        # Read gripper positions and convert from raw encoder to meters (0.0–0.1m)
        self._read_gripper_positions()
        for key, dxl_id in self._gripper_keys.items():
            raw_enc = float(self._gripper_positions[dxl_id])
            obs_dict[key] = self._encoder_to_meters(raw_enc)

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.read_latest()

        return obs_dict

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._is_connected or self._robot is None:
            raise ConnectionError("RBY1 is not connected")

        before_write_t = time.perf_counter()

        # Separate joint actions from gripper actions
        joint_goals: dict[str, float] = {}
        gripper_goals: dict[str, float] = {}

        for key, val in action.items():
            if key in self._gripper_keys:
                gripper_goals[key] = float(val)
            elif key in self._joints_dict:
                joint_goals[key] = float(val)

        # When an external controller (e.g. the Rainbow master arm teleop) is already
        # driving the robot, skip sending commands via the SDK. Otherwise the robot's
        # ControlManager rejects us with "new control's priority is not higher".
        if self.config.use_external_commands:
            self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t
            return action

        import rby1_sdk

        duration = self.config.command_duration

        # Apply safety limits if configured
        if self.config.max_relative_target is not None and joint_goals:
            with self._state_lock:
                state = self._latest_state

            if state is not None:
                goal_present_pos = {}
                for key, goal in joint_goals.items():
                    idx = self._joints_dict[key]
                    goal_present_pos[key] = (goal, float(state[idx]))
                joint_goals = ensure_safe_goal_position(
                    goal_present_pos, self.config.max_relative_target
                )

        # Build SDK command
        comp_command = rby1_sdk.ComponentBasedCommandBuilder()
        body_cmd = rby1_sdk.BodyComponentBasedCommandBuilder()
        has_body_cmd = False

        # Torso command
        if self.config.with_torso:
            torso_pos = [joint_goals.get(k, 0.0) for k in RBY1_TORSO_JOINTS]
            torso_cmd = (
                rby1_sdk.JointPositionCommandBuilder()
                .set_minimum_time(duration)
                .set_position(torso_pos)
                .set_command_header(
                    rby1_sdk.CommandHeaderBuilder().set_control_hold_time(2 * duration)
                )
            )
            body_cmd.set_torso_command(torso_cmd)
            has_body_cmd = True

        # Right arm command
        if self.config.with_right_arm:
            right_pos = [joint_goals.get(k, 0.0) for k in RBY1_RIGHT_ARM_JOINTS]
            right_cmd = (
                rby1_sdk.JointPositionCommandBuilder()
                .set_minimum_time(duration)
                .set_position(right_pos)
                .set_command_header(
                    rby1_sdk.CommandHeaderBuilder().set_control_hold_time(2 * duration)
                )
            )
            body_cmd.set_right_arm_command(right_cmd)
            has_body_cmd = True

        # Left arm command
        if self.config.with_left_arm:
            left_pos = [joint_goals.get(k, 0.0) for k in RBY1_LEFT_ARM_JOINTS]
            left_cmd = (
                rby1_sdk.JointPositionCommandBuilder()
                .set_minimum_time(duration)
                .set_position(left_pos)
                .set_command_header(
                    rby1_sdk.CommandHeaderBuilder().set_control_hold_time(2 * duration)
                )
            )
            body_cmd.set_left_arm_command(left_cmd)
            has_body_cmd = True

        if has_body_cmd:
            comp_command.set_body_command(body_cmd)

        # Head command
        if self.config.with_head:
            head_pos = [joint_goals.get(k, 0.0) for k in RBY1_HEAD_JOINTS]
            head_cmd = (
                rby1_sdk.JointPositionCommandBuilder()
                .set_minimum_time(duration)
                .set_position(head_pos)
                .set_command_header(
                    rby1_sdk.CommandHeaderBuilder().set_control_hold_time(2 * duration)
                )
            )
            head_builder = rby1_sdk.HeadCommandBuilder()
            head_builder.set_command(head_cmd)
            comp_command.set_head_command(head_builder)

        # Send robot command
        robot_command = rby1_sdk.RobotCommandBuilder()
        robot_command.set_command(comp_command)
        self._robot.send_command(robot_command)

        # Send gripper commands (via Dynamixel)
        # Note: gripper control requires a separate RainbowGripperController
        # For now we log gripper goals; full gripper commanding requires
        # the controller to be initialized with torque enabled.
        if gripper_goals:
            logger.debug(f"Gripper goals: {gripper_goals}")

        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t
        return action

    def disconnect(self) -> None:
        if not self._is_connected:
            return

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        # Close gripper bus
        if self._gripper_bus is not None:
            try:
                self._gripper_bus.close_port()
            except Exception:
                pass
            self._gripper_bus = None

        # Disconnect robot
        if self._robot is not None:
            try:
                self._robot.disconnect()
            except Exception:
                pass
            self._robot = None

        self._is_connected = False
        logger.info("RBY1 disconnected")
