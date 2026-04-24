#!/bin/bash
# Roll out a trained ACT policy on a remote RBY1, driving it via a
# `robot_proxy.py` daemon running on the Jetson.
#
# Run this script on the *workstation* (zima). All hardware I/O stays on the
# Jetson. The workstation:
#   - Subscribes to the Jetson's ZED camera ZMQ streams.
#   - Subscribes to the Jetson's robot_proxy state PUB.
#   - Runs policy inference on its own GPU.
#   - Sends action commands back to the Jetson via ZMQ.
#
# Prerequisites (on the Jetson, each in its own tmux/terminal):
#   scripts/rby1/start_zed_publisher.sh
#   scripts/rby1/start_robot_proxy.sh
#
# Usage:
#   ./rollout_policy_remote.sh <checkpoint_dir> <checkpoint_number>
#
# Args:
#   checkpoint_dir     Path to the training run, e.g.
#                      /data/objsearch/rby1_policy_learning/outputs/train/rby1_pick_v3_20260422_174437_act_vega
#   checkpoint_number  Step number or "last".
#
# Env-var overrides (defaults shown):
#   JETSON_HOST=10.31.132.177
#   STATE_PORT=5560
#   ACTION_PORT=5561
#   CAM_HEAD_PORT=5555
#   CAM_RIGHT_WRIST_PORT=5556
#   CAM_LEFT_WRIST_PORT=5557
#   TASK="pick block place in bowl"
#   EPISODE_TIME_S=150
#   FPS=10
#   CONDA_ENV=policy_inference
#   SCRATCH_ROOT=/data/objsearch/rby1_policy_learning/rollouts

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

JETSON_HOST="${JETSON_HOST:-10.31.132.177}"
STATE_PORT="${STATE_PORT:-5560}"
ACTION_PORT="${ACTION_PORT:-5561}"
CAM_HEAD_PORT="${CAM_HEAD_PORT:-5555}"
CAM_RIGHT_WRIST_PORT="${CAM_RIGHT_WRIST_PORT:-5556}"
CAM_LEFT_WRIST_PORT="${CAM_LEFT_WRIST_PORT:-5557}"

TASK="${TASK:-pick block place in bowl}"
EPISODE_TIME_S="${EPISODE_TIME_S:-150}"
FPS="${FPS:-10}"
CONDA_ENV="${CONDA_ENV:-policy_inference}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/data/objsearch/rby1_policy_learning/rollouts}"

RUN_TAG="$(basename "${CKPT_DIR}")_ckpt${CKPT_NUM}_remote"
STAMP="$(date +%Y%m%d_%H%M%S)"
DATASET_REPO_ID="local/eval_${RUN_TAG}_${STAMP}"
DATASET_ROOT="${SCRATCH_ROOT}/eval_${RUN_TAG}_${STAMP}"

# Camera config JSON — all three cameras point at the Jetson
export CAMS="{\
\"head_cam\":{\"type\":\"zmq\",\"server_address\":\"${JETSON_HOST}\",\"port\":${CAM_HEAD_PORT},\"camera_name\":\"head_cam\",\"width\":640,\"height\":480,\"fps\":10},\
\"right_wrist_cam\":{\"type\":\"zmq\",\"server_address\":\"${JETSON_HOST}\",\"port\":${CAM_RIGHT_WRIST_PORT},\"camera_name\":\"right_wrist_cam\",\"width\":640,\"height\":480,\"fps\":10},\
\"left_wrist_cam\":{\"type\":\"zmq\",\"server_address\":\"${JETSON_HOST}\",\"port\":${CAM_LEFT_WRIST_PORT},\"camera_name\":\"left_wrist_cam\",\"width\":640,\"height\":480,\"fps\":10}\
}"

echo "Rolling out policy (REMOTE mode):"
echo "  checkpoint        : ${POLICY_PATH}"
echo "  jetson host       : ${JETSON_HOST}"
echo "  state / action ports: ${STATE_PORT} / ${ACTION_PORT}"
echo "  cameras (on jetson): ${CAM_HEAD_PORT} / ${CAM_RIGHT_WRIST_PORT} / ${CAM_LEFT_WRIST_PORT}"
echo "  task              : ${TASK}"
echo "  episode_time_s    : ${EPISODE_TIME_S}"
echo "  fps               : ${FPS}"
echo "  conda env         : ${CONDA_ENV}"
echo "  scratch dir       : ${DATASET_ROOT}"
echo ""

exec conda run -n "${CONDA_ENV}" --no-capture-output lerobot-record \
    --robot.type=rby1_remote \
    --robot.jetson_host="${JETSON_HOST}" \
    --robot.state_port="${STATE_PORT}" \
    --robot.action_port="${ACTION_PORT}" \
    --robot.with_torso=false --robot.with_head=false \
    --robot.cameras="$CAMS" \
    --policy.path="${POLICY_PATH}" --policy.device=cuda \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.single_task="${TASK}" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s="${EPISODE_TIME_S}" \
    --dataset.fps="${FPS}" \
    --dataset.push_to_hub=false \
    --dataset.prompt_before_episode=false
