#!/bin/bash
# Roll out a trained ACT policy on the physical RBY1 for a single episode.
#
# Prerequisites (not started by this script):
#   - ZED publisher running (scripts/rby1/start_zed_publisher.sh)
#   - RBY1 reachable at 192.168.30.1:50051
#   - No master-arm teleop server running (the policy drives the robot)
#
# Usage:
#   ./rollout_policy.sh <checkpoint_dir> <checkpoint_number>
#
# Args:
#   checkpoint_dir     Path to the training run, e.g.
#                      /data/objsearch/rby1_policy_learning/outputs/train/rby1_pick_v2_act
#   checkpoint_number  Step number or "last", e.g. 180000 or last
#
# Examples:
#   ./rollout_policy.sh /data/objsearch/rby1_policy_learning/outputs/train/rby1_pick_v2_act last
#   ./rollout_policy.sh /data/objsearch/rby1_policy_learning/outputs/train/act_rby1_pick_v2_te0.01 300000

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <checkpoint_dir> <checkpoint_number>"
    exit 1
fi

CKPT_DIR="$1"
CKPT_NUM="$2"
POLICY_PATH="${CKPT_DIR}/checkpoints/${CKPT_NUM}/pretrained_model"

if [ ! -d "${POLICY_PATH}" ]; then
    echo "Error: policy path not found: ${POLICY_PATH}"
    exit 1
fi

# Allow overrides via env vars; fall back to defaults.
TASK="${TASK:-pick block place in bowl}"
EPISODE_TIME_S="${EPISODE_TIME_S:-150}"
FPS="${FPS:-10}"
CONDA_ENV="${CONDA_ENV:-policy_inference}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/data/objsearch/rby1_policy_learning/rollouts}"

# Build a human-readable tag from the checkpoint dir + number for the scratch path.
RUN_TAG="$(basename "${CKPT_DIR}")_ckpt${CKPT_NUM}"
STAMP="$(date +%Y%m%d_%H%M%S)"
DATASET_REPO_ID="local/eval_${RUN_TAG}_${STAMP}"
DATASET_ROOT="${SCRATCH_ROOT}/eval_${RUN_TAG}_${STAMP}"

export CAMS='{"head_cam":{"type":"zmq","server_address":"127.0.0.1","port":5555,"camera_name":"head_cam","width":640,"height":480,"fps":10},"right_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5556,"camera_name":"right_wrist_cam","width":640,"height":480,"fps":10},"left_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5557,"camera_name":"left_wrist_cam","width":640,"height":480,"fps":10}}'

echo "Rolling out policy:"
echo "  checkpoint     : ${POLICY_PATH}"
echo "  task           : ${TASK}"
echo "  episode_time_s : ${EPISODE_TIME_S}"
echo "  fps            : ${FPS}"
echo "  conda env      : ${CONDA_ENV}"
echo "  scratch dir    : ${DATASET_ROOT}"
echo ""

exec conda run -n "${CONDA_ENV}" --no-capture-output lerobot-record \
    --robot.type=rby1 --robot.robot_address=192.168.30.1:50051 \
    --robot.with_torso=false --robot.with_head=false \
    --robot.use_external_commands=false --robot.cameras="$CAMS" \
    --policy.path="${POLICY_PATH}" --policy.device=cuda \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.single_task="${TASK}" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s="${EPISODE_TIME_S}" \
    --dataset.fps="${FPS}" \
    --dataset.push_to_hub=false \
    --dataset.prompt_before_episode=false
