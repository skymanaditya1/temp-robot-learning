#!/bin/bash
# Thin wrapper around replay_dataset_rerun.py.
#
# Usage:
#   ./replay_dataset_rerun.sh <dataset_name> [episode] [replay_on_robot]
#
# Arguments:
#   dataset_name     Folder name under /data/objsearch/rby1_policy_learning/datasets/local/
#                    (e.g. rby1_pick_v3_20260422_154546)
#   episode          Episode index to replay. Default: 0
#   replay_on_robot  "true"/"1"/"yes" to send actions to the physical RBY1.
#                    Default: false (viz-only).
#
# Examples:
#   ./replay_dataset_rerun.sh rby1_pick_v3_20260422_154546
#   ./replay_dataset_rerun.sh rby1_pick_v3_20260422_154546 2
#   ./replay_dataset_rerun.sh rby1_pick_v3_20260422_154546 0 true

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_name> [episode] [replay_on_robot]" >&2
    exit 1
fi

DATASET_NAME="$1"
EPISODE="${2:-0}"
REPLAY_ON_ROBOT="${3:-false}"

REPO_ROOT="/data/objsearch/rby1_policy_learning"
DATASET_ROOT="${REPO_ROOT}/datasets/local/${DATASET_NAME}"
REPO_ID="local/${DATASET_NAME}"

if [ ! -d "$DATASET_ROOT" ]; then
    echo "Dataset not found: $DATASET_ROOT" >&2
    exit 1
fi

EXTRA_ARGS=()
case "${REPLAY_ON_ROBOT,,}" in
    true|1|yes|y)
        EXTRA_ARGS+=(--replay-on-robot)
        echo "Physical robot replay: ENABLED"
        ;;
    *)
        echo "Physical robot replay: disabled (viz-only)"
        ;;
esac

echo "Dataset: ${DATASET_NAME}"
echo "Episode: ${EPISODE}"
echo ""

exec conda run -n policy_new --no-capture-output python \
    "${REPO_ROOT}/scripts/rby1/replay_dataset_rerun.py" \
    --dataset-repo-id "${REPO_ID}" \
    --dataset-root "${DATASET_ROOT}" \
    --episode "${EPISODE}" \
    "${EXTRA_ARGS[@]}"
