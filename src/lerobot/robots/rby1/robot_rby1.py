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
from threading import Lock, Thread
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
        self._command_stream = None  # rby1_sdk command stream (SDK commanding mode only)
        self._is_connected = False

        # Thread-safe state from SDK callback
        self._state_lock = Lock()
        self._latest_state: np.ndarray | None = None

        # Gripper state (read via Dynamixel bus)
        self._gripper_bus = None
        self._gripper_positions = [0.0, 0.0]  # [right, left] — latest encoder reads

        # Gripper command state (SDK commanding mode only)
        self._gripper_write_enabled = False
        self._gripper_target_enc: list[float] = [0.0, 0.0]  # [right, left] — write thread target
        self._gripper_target_lock = Lock()
        self._gripper_writer_running = False
        self._gripper_writer_thread: Thread | None = None

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

        if not self.config.use_external_commands:
            logger.info("Enabling servo + control manager for SDK commanding...")
            if not self._robot.is_servo_on(".*"):
                if not self._robot.servo_on(".*"):
                    raise RuntimeError("Failed to servo on")
            self._robot.reset_fault_control_manager()
            if not self._robot.enable_control_manager():
                raise RuntimeError("Failed to enable control manager")
            # Open a priority command stream for high-rate joint position commanding
            # (matches rby1_standalone/replay.py pattern).
            self._command_stream = self._robot.create_command_stream(
                self.config.command_stream_priority
            )
            logger.info(
                f"Command stream opened (priority={self.config.command_stream_priority})"
            )

        # Initialize gripper bus (read-only by default; upgraded to read+write below
        # when SDK commanding + gripper commanding are both enabled).
        if self.config.with_right_gripper or self.config.with_left_gripper:
            try:
                self._gripper_bus = rby1_sdk.DynamixelBus(rby1_sdk.upc.GripperDeviceName)
                self._gripper_bus.open_port()
                self._gripper_bus.set_baud_rate(self.config.gripper_baudrate)
                logger.info("Gripper bus opened (read-only)")
            except Exception as e:
                logger.warning(f"Failed to open gripper bus: {e}. Gripper readings will be zero.")
                self._gripper_bus = None

        # Gripper write setup (only when SDK commanding + gripper commanding enabled).
        if (
            not self.config.use_external_commands
            and self.config.enable_gripper_commanding
            and self._gripper_bus is not None
        ):
            try:
                logger.info("Configuring gripper bus for writing...")
                for arm in ("right", "left"):
                    self._robot.set_tool_flange_output_voltage(arm, 12)
                self._gripper_bus.set_torque_constant([1, 1])
                self._gripper_bus.group_sync_write_torque_enable([(0, 1), (1, 1)])
                self._gripper_bus.group_sync_write_operating_mode(
                    [
                        (0, rby1_sdk.DynamixelBus.CurrentBasedPositionControlMode),
                        (1, rby1_sdk.DynamixelBus.CurrentBasedPositionControlMode),
                    ]
                )
                self._gripper_bus.group_sync_write_torque_enable([(0, 1), (1, 1)])
                cur = int(self.config.gripper_current_a)
                self._gripper_bus.group_sync_write_send_torque([(0, cur), (1, cur)])

                # Seed the write target with the current reading so grippers hold still
                # until send_action() supplies a real target.
                self._read_gripper_positions()
                with self._gripper_target_lock:
                    self._gripper_target_enc = list(self._gripper_positions)

                self._gripper_writer_running = True
                self._gripper_writer_thread = Thread(
                    target=self._gripper_writer_loop, daemon=True
                )
                self._gripper_writer_thread.start()
                self._gripper_write_enabled = True
                logger.info(
                    f"Gripper writer started (current cap {cur} A, 20 Hz)"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to enable gripper commanding: {e}. Grippers will be read-only."
                )
                self._gripper_write_enabled = False

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

    def _gripper_calibration(self, dxl_id: int) -> tuple[float, float]:
        """Return (enc_open, enc_closed) for the given Dynamixel id (0=right, 1=left)."""
        if dxl_id == 0:
            return self.config.right_gripper_enc_open, self.config.right_gripper_enc_closed
        elif dxl_id == 1:
            return self.config.left_gripper_enc_open, self.config.left_gripper_enc_closed
        raise ValueError(f"unknown gripper dxl_id: {dxl_id}")

    def _encoder_to_meters(self, raw_enc: float, dxl_id: int) -> float:
        """Convert raw Dynamixel encoder value to gripper width in meters (0.0–0.1m)."""
        enc_open, enc_closed = self._gripper_calibration(dxl_id)
        width_m = 0.1 * (raw_enc - enc_closed) / (enc_open - enc_closed)
        return float(np.clip(width_m, 0.0, 0.1))

    def _meters_to_encoder(self, width_m: float, dxl_id: int) -> float:
        """Inverse of _encoder_to_meters. Clamps input to the 0.0-0.1m range first."""
        width_m = float(np.clip(width_m, 0.0, 0.1))
        enc_open, enc_closed = self._gripper_calibration(dxl_id)
        return enc_closed + (width_m / 0.1) * (enc_open - enc_closed)

    def _gripper_writer_loop(self) -> None:
        """Background thread: at 20 Hz, write the latest gripper target positions."""
        period = 0.05  # 20 Hz
        while self._gripper_writer_running:
            try:
                with self._gripper_target_lock:
                    targets = list(self._gripper_target_enc)
                if self._gripper_bus is not None:
                    self._gripper_bus.group_sync_write_send_position(
                        [(dev_id, q) for dev_id, q in enumerate(targets)]
                    )
            except Exception as e:
                # Don't kill the thread on transient bus errors.
                logger.debug(f"gripper writer tick error: {e}")
            time.sleep(period)

    def _read_gripper_positions(self) -> None:
        """Read gripper encoder positions from Dynamixel bus.

        With the master-arm teleop process contending on the same RS-485 bus,
        the second id in a `group_fast_sync_read_encoder([0, 1])` request often
        gets its response packet stomped. We retry any missing id individually.
        """
        if self._gripper_bus is None:
            return
        try:
            rv = self._gripper_bus.group_fast_sync_read_encoder([0, 1])
        except Exception:
            rv = None
        got: set[int] = set()
        if rv is not None:
            for dev_id, enc in rv:
                if dev_id < 2:
                    self._gripper_positions[dev_id] = enc
                    got.add(int(dev_id))
        for missing in {0, 1} - got:
            try:
                rv2 = self._gripper_bus.group_fast_sync_read_encoder([missing])
                if rv2 is not None:
                    for dev_id, enc in rv2:
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
            obs_dict[key] = self._encoder_to_meters(raw_enc, dxl_id)

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

        if self._command_stream is None:
            raise RuntimeError(
                "Command stream not initialized; connect() must run with "
                "use_external_commands=False"
            )

        import rby1_sdk

        duration = self.config.command_duration

        # Snapshot latest robot state so joints we aren't commanding can hold still
        with self._state_lock:
            state = self._latest_state

        # Apply safety limits if configured
        if self.config.max_relative_target is not None and joint_goals and state is not None:
            goal_present_pos = {
                key: (goal, float(state[self._joints_dict[key]]))
                for key, goal in joint_goals.items()
            }
            joint_goals = ensure_safe_goal_position(
                goal_present_pos, self.config.max_relative_target
            )

        # Build the full 20-joint body vector: torso (6) + right_arm (7) + left_arm (7).
        # For joints the action dict doesn't include (e.g. torso when with_torso=False),
        # hold the current position so the robot doesn't move those joints.
        def _joint_target(name_to_idx: dict[str, int]) -> list[float]:
            out: list[float] = []
            for name, idx in name_to_idx.items():
                if name in joint_goals:
                    out.append(joint_goals[name])
                elif state is not None:
                    out.append(float(state[idx]))
                else:
                    out.append(0.0)
            return out

        body_pos = (
            _joint_target(RBY1_TORSO_JOINTS)
            + _joint_target(RBY1_RIGHT_ARM_JOINTS)
            + _joint_target(RBY1_LEFT_ARM_JOINTS)
        )

        # Single body JointPositionCommand — matches rby1_standalone/replay.py.
        body_cmd = (
            rby1_sdk.JointPositionCommandBuilder()
            .set_command_header(
                rby1_sdk.CommandHeaderBuilder().set_control_hold_time(1)
            )
            .set_minimum_time(duration)
            .set_position(body_pos)
        )
        comp_command = rby1_sdk.ComponentBasedCommandBuilder().set_body_command(body_cmd)

        # Head command (optional; kept separate from body).
        if self.config.with_head:
            head_pos = [joint_goals.get(k, 0.0) for k in RBY1_HEAD_JOINTS]
            head_cmd = (
                rby1_sdk.JointPositionCommandBuilder()
                .set_command_header(
                    rby1_sdk.CommandHeaderBuilder().set_control_hold_time(1)
                )
                .set_minimum_time(duration)
                .set_position(head_pos)
            )
            head_builder = rby1_sdk.HeadCommandBuilder()
            head_builder.set_command(head_cmd)
            comp_command.set_head_command(head_builder)

        robot_command = rby1_sdk.RobotCommandBuilder().set_command(comp_command)
        self._command_stream.send_command(robot_command)

        # Gripper commanding: push the latest target encoder values to the writer thread.
        if gripper_goals and self._gripper_write_enabled:
            with self._gripper_target_lock:
                new_target = list(self._gripper_target_enc)
                for key, dxl_id in self._gripper_keys.items():
                    if key in gripper_goals:
                        new_target[dxl_id] = self._meters_to_encoder(gripper_goals[key], dxl_id)
                self._gripper_target_enc = new_target

        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t
        return action

    def disconnect(self) -> None:
        if not self._is_connected:
            return

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        # Stop gripper writer thread (SDK commanding mode) and disable torque so the
        # grippers don't continue holding force after we disconnect.
        if self._gripper_writer_running:
            self._gripper_writer_running = False
            if self._gripper_writer_thread is not None:
                self._gripper_writer_thread.join(timeout=1.0)
                self._gripper_writer_thread = None
        if self._gripper_write_enabled and self._gripper_bus is not None:
            try:
                self._gripper_bus.group_sync_write_torque_enable([(0, 0), (1, 0)])
            except Exception:
                pass
            self._gripper_write_enabled = False

        # Close command stream (SDK commanding mode only)
        if self._command_stream is not None:
            try:
                # Not all SDK versions expose a stream close; guard defensively.
                if hasattr(self._command_stream, "close"):
                    self._command_stream.close()
            except Exception:
                pass
            self._command_stream = None

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
