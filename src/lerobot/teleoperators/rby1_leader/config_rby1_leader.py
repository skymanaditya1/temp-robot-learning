from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("rby1_leader")
@dataclass
class RBY1LeaderConfig(TeleoperatorConfig):
    """Configuration for RBY1 master arm teleoperator.

    The master arm is a separate RBY1 (or compatible) device whose joint
    positions are read as target actions for the follower robot.
    """

    # gRPC address of the master arm device
    robot_address: str = "192.168.30.1:50051"

    # SDK callback frequency for reading master arm state
    state_update_hz: int = 100

    # Which joint groups to read from the master arm
    with_torso: bool = False
    with_right_arm: bool = True
    with_left_arm: bool = True
    with_head: bool = False
    with_right_gripper: bool = True
    with_left_gripper: bool = True

    # Gripper configuration (for reading master arm gripper encoders)
    gripper_device: str = "/dev/rby1_gripper"
    gripper_baudrate: int = 2_000_000

    # Gripper encoder-to-meters calibration.
    # Raw Dynamixel encoder values at fully open and fully closed positions.
    # Used to convert raw encoder readings to gripper width in meters (0.0–0.1m).
    # Measure these after each gripper homing if they shift between sessions.
    gripper_enc_open: float = 1.3   # encoder radians when gripper is fully open
    gripper_enc_closed: float = 5.9  # encoder radians when gripper is fully closed

    def __post_init__(self):
        if not (
            self.with_torso
            or self.with_right_arm
            or self.with_left_arm
            or self.with_head
            or self.with_right_gripper
            or self.with_left_gripper
        ):
            raise ValueError(
                "At least one joint group must be enabled on the RBY1 leader "
                "(with_torso, with_right_arm, with_left_arm, with_head, "
                "with_right_gripper, with_left_gripper)"
            )
