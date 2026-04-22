#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <dataset_name> [steps=300000] [save_freq=10000]" >&2
    echo "example: $0 rby1_pick_v3_20260422_174437 300000 10000" >&2
    exit 1
fi

DATASET_NAME="$1"
STEPS="${2:-300000}"
SAVE_FREQ="${3:-10000}"

REPO_ROOT="/home/adityaag/Research/objsearch/rby1_policy_learning"
DATASET_ROOT="${REPO_ROOT}/datasets/local/${DATASET_NAME}"
REPO_ID="local/${DATASET_NAME}"
JOB_NAME="${DATASET_NAME}_act"
OUTPUT_DIR="outputs/train/${JOB_NAME}"

if [ ! -d "${DATASET_ROOT}" ]; then
    echo "error: dataset not found at ${DATASET_ROOT}" >&2
    exit 1
fi

lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --batch_size=8 \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq=200 \
  --num_workers=4 \
  --seed=1000 \
  --wandb.enable=true \
  --wandb.project=rby1_pick \
  --wandb.notes="ACT, ${DATASET_NAME}, bs=8, ${STEPS} steps"
