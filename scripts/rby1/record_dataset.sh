#!/bin/bash
# Record a teleop dataset on the RBY1 with the three ZED ZMQ cameras.
#
# Prerequisites (not started by this script):
#   - ZED publisher running on ports 5555/5556/5557
#     (scripts/rby1/start_zed_publisher.sh)
#   - Master-arm teleop server running:
#     sudo .venv/bin/python -m rby1_standalone.arms_teleop --address 192.168.30.1:50051
#
# Usage:
#   ./record_dataset.sh <task_name> <num_episodes> <episode_time_s>
#
# Args:
#   task_name        Task tag used for BOTH --dataset.single_task and the
#                    dataset folder name. Spaces are allowed and preserved in
#                    single_task; for the folder name, spaces are replaced by
#                    underscores. Example: "rby1_pick_v2" or "pick block in bowl".
#   num_episodes     Number of episodes to record (e.g. 50).
#   episode_time_s   Max duration per episode in seconds (e.g. 30).
#
# Env-var overrides (defaults shown):
#   FPS=10
#   CONDA_ENV=policy_new
#   ROBOT_ADDRESS=192.168.30.1:50051
#   DATASETS_ROOT=/data/objsearch/rby1_policy_learning/datasets/local
#   PROMPT_BEFORE_EPISODE=true   (false = no pause between episodes)
#
# Examples:
#   ./record_dataset.sh rby1_pick_v2 50 30
#   ./record_dataset.sh "pick block place in bowl" 50 30
#   PROMPT_BEFORE_EPISODE=false ./record_dataset.sh rby1_quick_test 5 15

set -e

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <task_name> <num_episodes> <episode_time_s>"
    exit 1
fi

TASK_NAME="$1"
NUM_EPISODES="$2"
EPISODE_TIME_S="$3"

FPS="${FPS:-10}"
CONDA_ENV="${CONDA_ENV:-policy_new}"
ROBOT_ADDRESS="${ROBOT_ADDRESS:-192.168.30.1:50051}"
DATASETS_ROOT="${DATASETS_ROOT:-/data/objsearch/rby1_policy_learning/datasets/local}"
PROMPT_BEFORE_EPISODE="${PROMPT_BEFORE_EPISODE:-true}"
HOMED_RANGE_FILE="${HOMED_RANGE_FILE:-/data/objsearch/rby1_policy_learning/session_encoder_values/gripper_homed_range.json}"

# Read this session's live gripper homed range (written by arms_teleop after
# its homing sweep). Recording must use these — not lerobot's static config —
# so encoder->width covers the current multi-turn frame without clipping.
if [ ! -f "$HOMED_RANGE_FILE" ]; then
    echo "ERROR: gripper homed range file not found:" >&2
    echo "  $HOMED_RANGE_FILE" >&2
    echo "Start arms_teleop first; it writes this file after its homing sweep." >&2
    exit 1
fi
read RG_MIN RG_MAX LG_MIN LG_MAX < <(python3 -c "
import json
with open('$HOMED_RANGE_FILE') as f:
    d = json.load(f)
print(d['right'][0], d['right'][1], d['left'][0], d['left'][1])
") || true
if [ -z "$RG_MIN" ] || [ -z "$LG_MAX" ]; then
    echo "ERROR: failed to parse $HOMED_RANGE_FILE" >&2
    exit 1
fi

# Sanitize task name for the dataset folder (spaces -> underscores).
TASK_TAG="${TASK_NAME// /_}"

STAMP="$(date +%Y%m%d_%H%M%S)"
DATASET_NAME="${TASK_TAG}_${STAMP}"
DATASET_ROOT="${DATASETS_ROOT}/${DATASET_NAME}"
DATASET_REPO_ID="local/${DATASET_NAME}"

export CAMS='{"head_cam":{"type":"zmq","server_address":"127.0.0.1","port":5555,"camera_name":"head_cam","width":640,"height":480,"fps":10},"right_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5556,"camera_name":"right_wrist_cam","width":640,"height":480,"fps":10},"left_wrist_cam":{"type":"zmq","server_address":"127.0.0.1","port":5557,"camera_name":"left_wrist_cam","width":640,"height":480,"fps":10}}'

echo "Recording dataset:"
echo "  task (single_task): ${TASK_NAME}"
echo "  num episodes      : ${NUM_EPISODES}"
echo "  episode time (s)  : ${EPISODE_TIME_S}"
echo "  fps               : ${FPS}"
echo "  robot address     : ${ROBOT_ADDRESS}"
echo "  conda env         : ${CONDA_ENV}"
echo "  dataset repo id   : ${DATASET_REPO_ID}"
echo "  dataset root      : ${DATASET_ROOT}"
echo "  prompt before ep  : ${PROMPT_BEFORE_EPISODE}"
echo "  gripper enc range : right=[${RG_MIN}, ${RG_MAX}]  left=[${LG_MIN}, ${LG_MAX}]"
echo ""

exec conda run -n "${CONDA_ENV}" --no-capture-output lerobot-record \
    --robot.type=rby1 --robot.robot_address="${ROBOT_ADDRESS}" \
    --robot.with_torso=false --robot.with_head=false --robot.cameras="$CAMS" \
    --robot.right_gripper_enc_closed="${RG_MIN}" \
    --robot.right_gripper_enc_open="${RG_MAX}" \
    --robot.left_gripper_enc_closed="${LG_MIN}" \
    --robot.left_gripper_enc_open="${LG_MAX}" \
    --teleop.type=rby1_leader --teleop.robot_address="${ROBOT_ADDRESS}" \
    --teleop.with_torso=false --teleop.with_head=false \
    --teleop.right_gripper_enc_closed="${RG_MIN}" \
    --teleop.right_gripper_enc_open="${RG_MAX}" \
    --teleop.left_gripper_enc_closed="${LG_MIN}" \
    --teleop.left_gripper_enc_open="${LG_MAX}" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.single_task="${TASK_NAME}" \
    --dataset.num_episodes="${NUM_EPISODES}" \
    --dataset.episode_time_s="${EPISODE_TIME_S}" \
    --dataset.fps="${FPS}" \
    --dataset.push_to_hub=false \
    --dataset.prompt_before_episode="${PROMPT_BEFORE_EPISODE}"
