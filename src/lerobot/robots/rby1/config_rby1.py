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

    # When True and use_external_commands=False, RBY1 will actively drive the
    # Dynamixel grippers via a background thread so recorded gripper values are
    # replayed on the real hardware. Automatically skipped in master-arm mode.
    enable_gripper_commanding: bool = True
    # Per-gripper current cap (amps) in CurrentBasedPositionControlMode. 5 A matches
    # rby1_standalone/replay.py. Lower = softer grasp, higher = firmer but risks damage.
    gripper_current_a: float = 5.0

    # Per-gripper encoder-to-meters calibration. Raw Dynamixel encoder values at
    # fully open and fully closed positions. Defaults measured empirically — the
    # multi-turn encoders span ~9.5 rad of motion with different offsets per side.
    # Recalibrate if dataset values drift outside [0.0, 0.1] m.
    right_gripper_enc_open: float = 4.83
    right_gripper_enc_closed: float = -4.74
    left_gripper_enc_open: float = 1.26
    left_gripper_enc_closed: float = -8.24

    # Cameras (typically ZMQ cameras for ZED)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
