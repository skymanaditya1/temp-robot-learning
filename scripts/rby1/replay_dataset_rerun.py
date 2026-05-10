"""Replay a recorded RBY1 episode into Rerun for timeline-synchronized visualization,
with optional physical robot replay.

Default (no flag): viz-only. Loads images + joint state + action from the dataset and
streams them to Rerun on a shared timeline. The robot is not touched.

With --replay-on-robot: additionally connects to the RBY1 and sends the recorded
actions at dataset fps (mirrors `lerobot-replay`). Make sure the teleop server is
stopped and the E-stop is within reach before running.

Examples:
    # viz-only
    python scripts/rby1/replay_dataset_rerun.py \
        --dataset-repo-id local/rby1_pick_v3_20260422_154546 \
        --dataset-root /data/objsearch/rby1_policy_learning/datasets/local/rby1_pick_v3_20260422_154546 \
        --episode 0

    # save .rrd instead of spawning GUI (useful over SSH)
    python scripts/rby1/replay_dataset_rerun.py ... --save-rrd /tmp/ep0.rrd

    # also replay on the physical robot
    python scripts/rby1/replay_dataset_rerun.py ... --replay-on-robot
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rerun as rr

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _to_hwc_uint8(img) -> np.ndarray:
    """Dataset images come as CHW float in [0, 1]. Rerun wants HWC uint8."""
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def _joint_group(name: str) -> tuple[str, str]:
    """Map a joint feature name to (group_path, leaf_name) for Rerun entity layout."""
    if name.startswith("right_arm"):
        return "right_arm", name
    if name.startswith("left_arm"):
        return "left_arm", name
    if name.startswith("right_gripper"):
        return "right_gripper", name
    if name.startswith("left_gripper"):
        return "left_gripper", name
    return "other", name


def run(args):
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        episodes=[args.episode],
    )
    fps = args.fps or int(dataset.meta.fps)
    action_names: list[str] = dataset.features["action"]["names"]
    state_names: list[str] = dataset.features["observation.state"]["names"]
    image_keys = [k for k in dataset.features if k.startswith("observation.images.")]
    print(f"Episode {args.episode}: {dataset.num_frames} frames, fps={fps}")
    print(f"Image keys: {image_keys}")

    rr.init("rby1_replay", spawn=not args.save_rrd)
    if args.save_rrd:
        rr.save(args.save_rrd)
        print(f"Saving Rerun recording to {args.save_rrd}")

    # Robot bring-up (optional)
    robot = None
    robot_action_processor = None
    if args.replay_on_robot:
        # Lazy imports so viz-only mode doesn't pay the robot-stack import cost.
        from lerobot.processor import make_default_robot_action_processor
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.rby1 import RBY1Config

        print("\n" + "=" * 60)
        print("PHYSICAL REPLAY MODE — robot will move.")
        print("  - Ensure the arms_teleop master server is STOPPED.")
        print("  - Ensure the E-stop is within reach.")
        print("  - Workspace clear.")
        print("=" * 60)
        input("Press Enter to connect and start, or Ctrl-C to abort... ")

        robot_cfg = RBY1Config(
            robot_address=args.robot_address,
            with_torso=False,
            with_head=False,
            use_external_commands=False,
            gripper_current_cap=args.gripper_current_a,
            command_duration=args.command_duration,
        )
        robot = make_robot_from_config(robot_cfg)
        robot_action_processor = make_default_robot_action_processor()
        robot.connect()
        print("Robot connected.")

    try:
        dt = 1.0 / fps
        for idx in range(dataset.num_frames):
            loop_start = time.perf_counter()
            sample = dataset[idx]

            t_sec = float(sample["timestamp"].item()) if "timestamp" in sample else idx * dt
            rr.set_time("time", duration=t_sec)

            # Images
            for k in image_keys:
                short = k.removeprefix("observation.images.")
                rr.log(f"cameras/{short}", rr.Image(_to_hwc_uint8(sample[k])))

            # State + action scalars (overlay on same plot per group)
            state = sample["observation.state"].numpy()
            action = sample["action"].numpy()
            for i, n in enumerate(state_names):
                grp, leaf = _joint_group(n)
                rr.log(f"joints/{grp}/{leaf}/state", rr.Scalars(float(state[i])))
            for i, n in enumerate(action_names):
                grp, leaf = _joint_group(n)
                rr.log(f"joints/{grp}/{leaf}/action", rr.Scalars(float(action[i])))

            # Physical replay
            if robot is not None:
                action_dict = {n: float(action[i]) for i, n in enumerate(action_names)}
                robot_obs = robot.get_observation()
                processed = robot_action_processor((action_dict, robot_obs))
                robot.send_action(processed)

                # Log actual robot-measured joint state as a third overlay
                for n in action_names:
                    if n in robot_obs:
                        grp, leaf = _joint_group(n)
                        rr.log(f"joints/{grp}/{leaf}/robot", rr.Scalars(float(robot_obs[n])))

            # Pace the loop
            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

            if idx % 20 == 0:
                print(f"  frame {idx}/{dataset.num_frames}")
    finally:
        if robot is not None:
            robot.disconnect()
            print("Robot disconnected.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-repo-id", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--fps", type=int, default=None, help="Override playback fps (default: dataset fps)")
    p.add_argument("--save-rrd", type=Path, default=None,
                   help="If set, write an .rrd recording instead of spawning the Rerun viewer.")
    p.add_argument("--replay-on-robot", action="store_true",
                   help="Also send recorded actions to the physical RBY1 at dataset fps.")
    p.add_argument("--robot-address", default="192.168.30.1:50051")
    p.add_argument("--gripper-current-a", type=float, default=5.0,
                   help="Gripper current cap (A). Only used with --replay-on-robot.")
    p.add_argument("--command-duration", type=float, default=0.2,
                   help=("set_minimum_time(τ) for each JointPositionCommand. "
                         "Should be roughly 2× the playback tick interval "
                         "(=2×1/fps) so successive 100ms-spaced waypoints blend "
                         "instead of decelerating to zero at every sample. "
                         "Default 0.2s for 10 fps datasets."))
    run(p.parse_args())


if __name__ == "__main__":
    main()
