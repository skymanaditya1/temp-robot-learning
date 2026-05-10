#!/bin/bash
# Replay an episode from a both-arms dataset on the RBY1 via lerobot-replay.
# Matches the robot config used by record_dataset.sh (both arms + both
# grippers, no torso, no head, no cameras).
#
# Prerequisites:
#   - arms_teleop must NOT be running (lerobot needs to own the gripper bus
#     and home it).
#
# Usage:
#   ./replay_dataset.sh <dataset_name> [episode]
#
# Arguments:
#   dataset_name   Folder name under ${DATASETS_ROOT}
#                  (e.g. quick_test_20260427_141804)
#   episode        Episode index to replay. Default: 0
#
# Env-var overrides (defaults shown):
#   CONDA_ENV=policy_new
#   ROBOT_ADDRESS=192.168.30.1:50051
#   DATASETS_ROOT=/data/objsearch/rby1_policy_learning/datasets/local

set -e

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <dataset_name> [episode]" >&2
    exit 1
fi

DATASET_NAME="$1"
EPISODE="${2:-0}"

CONDA_ENV="${CONDA_ENV:-policy_new}"
ROBOT_ADDRESS="${ROBOT_ADDRESS:-192.168.30.1:50051}"
DATASETS_ROOT="${DATASETS_ROOT:-/data/objsearch/rby1_policy_learning/datasets/local}"

DATASET_ROOT="${DATASETS_ROOT}/${DATASET_NAME}"
DATASET_REPO_ID="local/${DATASET_NAME}"

if [ ! -d "$DATASET_ROOT" ]; then
    echo "Dataset not found: $DATASET_ROOT" >&2
    exit 1
fi

echo "Replaying dataset:"
echo "  dataset repo id : ${DATASET_REPO_ID}"
echo "  dataset root    : ${DATASET_ROOT}"
echo "  episode         : ${EPISODE}"
echo "  robot address   : ${ROBOT_ADDRESS}"
echo ""

exec conda run -n "${CONDA_ENV}" --no-capture-output lerobot-replay \
    --robot.type=rby1 --robot.robot_address="${ROBOT_ADDRESS}" \
    --robot.with_torso=false --robot.with_head=false \
    --robot.use_external_commands=false \
    --robot.cameras='{}' \
    --robot.command_duration=0.3 \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.episode="${EPISODE}"
