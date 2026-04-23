#!/bin/bash
# Run the overfit inference test: predict actions on frames from the training
# dataset, compare to ground-truth actions, and plot per-joint curves.
#
# Usage:
#   ./overfit_inference_plot.sh <episode> <dataset_folder_name> <checkpoint_dir> <checkpoint_number>
#
# Args:
#   episode               Episode index in the dataset (e.g. 0, 25).
#   dataset_folder_name   Last folder name under /data/objsearch/rby1_policy_learning/datasets/local/
#                         (e.g. rby1_pick_v2_20260419_180507_rgb)
#   checkpoint_dir        Training run directory (the one containing checkpoints/)
#                         e.g. /data/objsearch/rby1_policy_learning/outputs/train/rby1_pick_v2_act
#   checkpoint_number     Checkpoint step under checkpoints/ (e.g. 150000, or "last").
#
# Env-var overrides (defaults shown):
#   DEVICE=cuda         Torch device (policy_inference has CUDA; use DEVICE=cpu otherwise)
#   CONDA_ENV=policy_inference
#   DATASETS_ROOT=/data/objsearch/rby1_policy_learning/datasets/local
#   OUT=<auto>          Auto-derived unique filename under temp_images/ if unset
#
# Examples:
#   ./overfit_inference_plot.sh 0 rby1_pick_v2_20260419_180507_rgb \
#       /data/objsearch/rby1_policy_learning/outputs/train/rby1_pick_v2_act last
#
#   ./overfit_inference_plot.sh 25 rby1_pick_v2_20260419_180507_rgb \
#       /data/objsearch/rby1_policy_learning/outputs/train/act_rby1_pick_v2_te0.01 150000

set -e

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <episode> <dataset_folder_name> <checkpoint_dir> <checkpoint_number>"
    exit 1
fi

EPISODE="$1"
DATASET_NAME="$2"
CKPT_DIR="$3"
CKPT_NUM="$4"

DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-policy_inference}"
DATASETS_ROOT="${DATASETS_ROOT:-/data/objsearch/rby1_policy_learning/datasets/local}"

DATASET_ROOT="${DATASETS_ROOT}/${DATASET_NAME}"
DATASET_REPO_ID="local/${DATASET_NAME}"
POLICY_PATH="${CKPT_DIR}/checkpoints/${CKPT_NUM}/pretrained_model"

if [ ! -d "${DATASET_ROOT}" ]; then
    echo "Error: dataset root not found: ${DATASET_ROOT}"
    exit 1
fi
if [ ! -d "${POLICY_PATH}" ]; then
    echo "Error: policy path not found: ${POLICY_PATH}"
    exit 1
fi

CKPT_TAG="$(basename "${CKPT_DIR}")_ckpt${CKPT_NUM}"
DEFAULT_OUT="/data/objsearch/rby1_policy_learning/temp_images/overfit_${CKPT_TAG}_ep${EPISODE}.png"
OUT="${OUT:-${DEFAULT_OUT}}"

echo "Overfit inference plot:"
echo "  episode      : ${EPISODE}"
echo "  dataset      : ${DATASET_REPO_ID} @ ${DATASET_ROOT}"
echo "  checkpoint   : ${POLICY_PATH}"
echo "  device       : ${DEVICE}"
echo "  conda env    : ${CONDA_ENV}"
echo "  output plot  : ${OUT}"
echo ""

exec conda run -n "${CONDA_ENV}" --no-capture-output python \
    /data/objsearch/rby1_policy_learning/scripts/rby1/overfit_inference_plot.py \
    --checkpoint "${POLICY_PATH}" \
    --dataset-repo-id "${DATASET_REPO_ID}" \
    --dataset-root "${DATASET_ROOT}" \
    --episode "${EPISODE}" \
    --device "${DEVICE}" \
    --fresh-chunk-every-step \
    --out "${OUT}"
