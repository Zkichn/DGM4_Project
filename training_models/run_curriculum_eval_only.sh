#!/bin/bash
# Evaluate an already-trained curriculum final adapter.
# Usage: bash run_curriculum_eval_only.sh [adapter_path] [batch_size] [limit]

set -euo pipefail

PROJECT_DIR="/root/autodl-tmp/DGM4_Project"
ADAPTER="${1:-${PROJECT_DIR}/training_models/outputs/qwen3-vl-8b/dgm4-curriculum/stage3-final}"
BATCH_SIZE="${2:-8}"
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
