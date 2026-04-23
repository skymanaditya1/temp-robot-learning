#!/bin/bash
# Summarize + plot per-joint trajectories for a recorded RBY1 dataset.
#
# Usage:
#   ./analyze_dataset.sh <dataset_name>
#
# Args:
#   dataset_name   Folder name under /data/objsearch/rby1_policy_learning/datasets/local/
#                  e.g. rby1_pick_v3_20260422_174437
#
# Env-var overrides (defaults shown):
#   FEATURE=observation.state   (or "action")
#   CONDA_ENV=policy_new
#   DATASETS_ROOT=/data/objsearch/rby1_policy_learning/datasets/local
#   OUT=<auto>                  (default: temp_images/dataset_summary_<name>_<feature>.png)
#
# Examples:
#   ./analyze_dataset.sh rby1_pick_v3_20260422_174437
#   FEATURE=action ./analyze_dataset.sh rby1_pick_v3_20260422_174437

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

DATASET_NAME="$1"

FEATURE="${FEATURE:-observation.state}"
CONDA_ENV="${CONDA_ENV:-policy_new}"
DATASETS_ROOT="${DATASETS_ROOT:-/data/objsearch/rby1_policy_learning/datasets/local}"

OUT_ARGS=()
if [ -n "${OUT:-}" ]; then
    OUT_ARGS=(--out "${OUT}")
fi

exec conda run -n "${CONDA_ENV}" --no-capture-output python \
    /data/objsearch/rby1_policy_learning/scripts/rby1/analyze_dataset.py \
    --dataset-name "${DATASET_NAME}" \
    --feature "${FEATURE}" \
    --datasets-root "${DATASETS_ROOT}" \
    "${OUT_ARGS[@]}"
