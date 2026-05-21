#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/DGM4_Project/training_models
STAMP=$(date +"%Y%m%d_%H%M%S")
LOG="rerank_grounding/rerank_stage2_v2_${STAMP}.log"
mkdir -p rerank_grounding/results

BASE_MODEL="/root/autodl-tmp/DGM4_Project/training_models/base_models/Qwen3-VL-8B-Instruct"
ADAPTER="/root/autodl-tmp/DGM4_Project/training_models/outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding-v2"
TEST_DATA="/root/autodl-tmp/DGM4_Project/training_models/datasets/dgm4_stage2_grounding_v2/test.json"
MEDIA_DIR="/root/autodl-tmp/datasets"
INPUT_EVAL="/root/autodl-tmp/DGM4_Project/training_models/outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding-v2/eval_stage2_v2_binary_multilabel_grounding.json"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

{
  echo "================================================"
  echo "Stage2-v2 grounding rerank eval"
  echo "Started: $(date)"
  echo "Mode: ${1:-both}"
  echo "================================================"
  /root/miniconda3/bin/python rerank_grounding/rerank_stage2_v2.py \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --test-data "$TEST_DATA" \
    --media-dir "$MEDIA_DIR" \
    --input-eval "$INPUT_EVAL" \
    --output "rerank_grounding/results/stage2_v2_${1:-both}_rerank.json" \
    --mode "${1:-both}" \
    --score-batch-size "${2:-8}" \
    --token-threshold "${3:-0.5}" \
    --token-window "${4:-1}" \
    --max-token-candidates "${5:-24}"
  echo "Finished: $(date)"
} 2>&1 | tee "$LOG"

echo "$LOG" > rerank_grounding/latest_rerank_logpath
