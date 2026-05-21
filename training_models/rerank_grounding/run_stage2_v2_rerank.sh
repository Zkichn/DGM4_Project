#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/DGM4_Project/training_models
STAMP=$(date +"%Y%m%d_%H%M%S")
if [[ "${1:-}" =~ ^(token|box|both)$ ]]; then
  # Backward-compatible form:
  #   run_stage2_v2_rerank.sh both 12 0.5 1 24
  RUN_KIND="full"
  MODE="${1:-both}"
  BATCH_SIZE="${2:-8}"
  TOKEN_THRESHOLD="${3:-0.5}"
  TOKEN_WINDOW="${4:-1}"
  MAX_TOKEN_CANDIDATES="${5:-24}"
else
  # New form:
  #   run_stage2_v2_rerank.sh smoke both 8 0.5 1 24
  #   run_stage2_v2_rerank.sh full both 12 0.5 1 24
  RUN_KIND="${1:-full}"
  MODE="${2:-both}"
  BATCH_SIZE="${3:-8}"
  TOKEN_THRESHOLD="${4:-0.5}"
  TOKEN_WINDOW="${5:-1}"
  MAX_TOKEN_CANDIDATES="${6:-24}"
fi

if [ "$RUN_KIND" = "smoke" ]; then
  LIMIT=5
  OUTPUT="rerank_grounding/results/stage2_v2_${MODE}_rerank_smoke5.json"
  EXTRA_ARGS=(--limit "$LIMIT" --print-score-debug --score-debug-examples 5)
else
  LIMIT=0
  OUTPUT="rerank_grounding/results/stage2_v2_${MODE}_rerank.json"
  EXTRA_ARGS=()
fi

LOG="rerank_grounding/rerank_stage2_v2_${RUN_KIND}_${STAMP}.log"
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
  echo "Run kind: $RUN_KIND"
  echo "Mode: $MODE"
  echo "Limit: $LIMIT"
  echo "================================================"
  /root/miniconda3/bin/python rerank_grounding/rerank_stage2_v2.py \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --test-data "$TEST_DATA" \
    --media-dir "$MEDIA_DIR" \
    --input-eval "$INPUT_EVAL" \
    --output "$OUTPUT" \
    --mode "$MODE" \
    --score-batch-size "$BATCH_SIZE" \
    --token-threshold "$TOKEN_THRESHOLD" \
    --token-window "$TOKEN_WINDOW" \
    --max-token-candidates "$MAX_TOKEN_CANDIDATES" \
    "${EXTRA_ARGS[@]}"
  echo "Finished: $(date)"
} 2>&1 | tee "$LOG"

echo "$LOG" > rerank_grounding/latest_rerank_logpath
