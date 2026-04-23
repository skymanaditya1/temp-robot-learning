#!/bin/bash
# Dump per-frame camera images (pre- and post-normalizer) for one episode.
#
# Usage:
#   ./dump_episode_images.sh <episode> <dataset_name> <checkpoint_dir> <checkpoint_number>
#
# Args:
#   episode             Episode index in the dataset (e.g. 0, 10).
#   dataset_name        Folder under /data/objsearch/rby1_policy_learning/datasets/local/
#   checkpoint_dir      Training run directory (contains checkpoints/)
#   checkpoint_number   Checkpoint step or "last"
#
# Env-var overrides (defaults shown):
#   STRIDE=1             Save every Nth frame
#   OUT_DIR=<auto>       Default: temp_images/episode_images/<dataset_name>/ep<N>/
#   DEVICE=cpu           Preprocessing is cheap; cpu is fine
#   CONDA_ENV=policy_inference
#   DATASETS_ROOT=/data/objsearch/rby1_policy_learning/datasets/local
#
# Example:
#   ./dump_episode_images.sh 10 rby1_pick_v3_20260422_174437 \
#       /data/objsearch/rby1_policy_learning/outputs/train/rby1_pick_v3_20260422_174437_act_vega last
#   STRIDE=10 ./dump_episode_images.sh 0 rby1_pick_v3_20260422_174437 \
#       /data/objsearch/rby1_policy_learning/outputs/train/rby1_pick_v3_20260422_174437_act_vega last

set -e

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <episode> <dataset_name> <checkpoint_dir> <checkpoint_number>"
    exit 1
fi

EPISODE="$1"
DATASET_NAME="$2"
CKPT_DIR="$3"
CKPT_NUM="$4"

STRIDE="${STRIDE:-1}"
DEVICE="${DEVICE:-cpu}"
CONDA_ENV="${CONDA_ENV:-policy_inference}"
DATASETS_ROOT="${DATASETS_ROOT:-/data/objsearch/rby1_policy_learning/datasets/local}"

POLICY_PATH="${CKPT_DIR}/checkpoints/${CKPT_NUM}/pretrained_model"
if [ ! -d "${POLICY_PATH}" ]; then
    echo "Error: policy path not found: ${POLICY_PATH}"
    exit 1
fi

OUT_DIR_ARGS=()
if [ -n "${OUT_DIR:-}" ]; then
    OUT_DIR_ARGS=(--out-dir "${OUT_DIR}")
fi

exec conda run -n "${CONDA_ENV}" --no-capture-output python \
    /data/objsearch/rby1_policy_learning/scripts/rby1/dump_episode_images.py \
    --dataset-name "${DATASET_NAME}" \
    --episode "${EPISODE}" \
    --checkpoint "${POLICY_PATH}" \
    --datasets-root "${DATASETS_ROOT}" \
    --stride "${STRIDE}" \
    --device "${DEVICE}" \
    "${OUT_DIR_ARGS[@]}"
