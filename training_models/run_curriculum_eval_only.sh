#!/bin/bash
# Evaluate an already-trained curriculum final adapter.
# Usage: bash run_curriculum_eval_only.sh [adapter_path] [batch_size] [limit]

set -euo pipefail

PROJECT_DIR="/root/autodl-tmp/DGM4_Project"
LOG_DIR="${PROJECT_DIR}/training_models/logs"
mkdir -p "${LOG_DIR}"
if [ -z "${DGM4_LOG_REDIRECTED:-}" ]; then
    export DGM4_LOG_REDIRECTED=1
    LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_run_curriculum_eval_only.log"
    echo "${LOG_FILE}" > "${LOG_DIR}/run_curriculum_eval_only.latest.logpath"
    echo "$$" > "${LOG_DIR}/run_curriculum_eval_only.pid"
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi
ADAPTER="${1:-${PROJECT_DIR}/training_models/outputs/qwen3-vl-8b/dgm4-curriculum/stage3-final}"
BATCH_SIZE="${2:-24}"
LIMIT="${3:-0}"
OUTPUT="${ADAPTER}/eval_dgm4_12metrics.json"

cd "${PROJECT_DIR}/training_models"

HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/root/miniconda3/bin/python3 eval_dgm4_qwen3vl.py \
    --adapter "${ADAPTER}" \
    --test-data "datasets/dgm4_stage3_final/test.json" \
    --media-dir "/root/autodl-tmp/datasets" \
    --output "${OUTPUT}" \
    --batch-size "${BATCH_SIZE}" \
    --limit "${LIMIT}"
