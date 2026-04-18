from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("rby1")
@dataclass
class RBY1Config(RobotConfig):
    """Configuration for Rainbow Robotics RBY1 robot.

    The RBY1 is a dual-arm mobile manipulator with:
    - 6-DOF torso
    - Two 7-DOF arms (left and right)
    - 2-DOF head
    - Dynamixel-based grippers (left and right)

    State vector layout (24 joints total):
        [0:2]   base (omitted for safety)
        [2:8]   torso (6 joints)
        [8:15]  right arm (7 joints)
        [15:22] left arm (7 joints)
        [22:24] head (2 joints)
        + 2 gripper encoders (read separately via Dynamixel bus)
    """

    # Robot connection
    robot_address: str = "192.168.30.1:50051"

    # Safety
    max_relative_target: float | None = None

    # SDK parameters
    state_update_hz: int = 100
    # minimum_time for each SDK joint-position command. 0.1 matches the known-working
    # rby1_standalone replay.py, paired with a ~10 Hz command rate.
    command_duration: float = 0.1
    # Stream priority for the command stream (SDK commanding mode). Higher = preempts
    # lower-priority streams. Matches rby1_standalone/replay.py's `create_command_stream(10)`.
    command_stream_priority: int = 10

    # When True, send_action() will NOT send joint commands to the robot via the SDK.
    # Use this when an external controller (e.g. the Rainbow master arm teleop) is
    # already driving the robot. lerobot will still record the action values that
    # the teleoperator reports, but it won't try to preempt the existing control,
    # which avoids "new control's priority is not higher" conflicts.
    use_external_commands: bool = True

    # Joint groups to include in observations/actions
    with_torso: bool = True
    with_right_arm: bool = True
    with_left_arm: bool = True
    with_head: bool = True
    with_right_gripper: bool = True
    with_left_gripper: bool = True

    # Gripper configuration
    gripper_device: str = "/dev/rby1_gripper"
    gripper_baudrate: int = 2_000_000

    # Gripper encoder-to-meters calibration.
    # Raw Dynamixel encoder values at fully open and fully closed positions.
    # Used to convert raw encoder readings to gripper width in meters (0.0–0.1m).
    gripper_enc_open: float = 1.3   # encoder radians when gripper is fully open
    gripper_enc_closed: float = 5.9  # encoder radians when gripper is fully closed

    # Cameras (typically ZMQ cameras for ZED)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
