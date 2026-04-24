from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("rby1_remote")
@dataclass
class RBY1RemoteConfig(RobotConfig):
    """Configuration for the *remote* RBY1 robot adapter.

    Runs on a workstation. Talks over ZMQ to a `robot_proxy.py` instance on
    the Jetson (which owns the actual hardware — gRPC arm + Dynamixel gripper
    bus). Cameras are read directly from ZED ZMQ publishers on the Jetson via
    the standard LeRobot `ZMQCamera` type.

    Feature schema (joint names, action dims) is intentionally identical to
    RBY1Config so datasets and policies built for `rby1` interchange cleanly
    with `rby1_remote`.
    """

    # Hostname or IP of the Jetson running robot_proxy.py + start_zed_publisher.sh
    jetson_host: str = "10.31.132.177"

    # ZMQ ports on the Jetson
    state_port: int = 5560          # robot_proxy.py state PUB
    action_port: int = 5561         # robot_proxy.py action PULL

    # Freshness threshold for the state stream (ms). If the last state message
    # is older than this, get_observation() raises TimeoutError — same policy
    # as ZMQCamera.read_latest.
    state_max_age_ms: int = 1000

    # Safety (pass-through; not enforced here because commanding happens on the Jetson)
    max_relative_target: float | None = None

    # Joint groups to include in observations/actions. Must match what the
    # Jetson-side proxy was started with, otherwise action keys won't reach
    # the robot.
    with_torso: bool = False
    with_right_arm: bool = True
    with_left_arm: bool = True
    with_head: bool = False
    with_right_gripper: bool = True
    with_left_gripper: bool = True

    # Cameras (CameraConfig objects — typically ZMQCameraConfig whose
    # `server_address` points at `jetson_host`).
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
