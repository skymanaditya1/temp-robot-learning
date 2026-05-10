"""Lerobot replay variant for use when an external owner of /dev/rby1_gripper
(e.g. RainbowGripperController on the UPC) is already running and has homed
the gripper. Releases the bus to lerobot via release_gripper_bus / reclaim_gripper_bus
on the server side; this script just sets --robot.skip_gripper_homing=true so
lerobot doesn\'t re-home and doesn\'t mutate bus state.

Run via:
    python -m lerobot.scripts.lerobot_replay_no_homing --robot.type=rby1 ...
The wrapper auto-injects --robot.skip_gripper_homing=true if you don\'t pass it.
"""

import sys

from lerobot.scripts.lerobot_replay import main as _main


def main():
    if not any(a.startswith("--robot.skip_gripper_homing") for a in sys.argv):
        sys.argv.append("--robot.skip_gripper_homing=true")
    _main()


if __name__ == "__main__":
    main()
