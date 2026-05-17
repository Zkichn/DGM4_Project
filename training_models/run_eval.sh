#!/bin/bash
# DGM4 Evaluation Launcher — guaranteed shutdown via trap
# Usage: bash run_eval.sh [adapter_path] [batch_size] [limit]

ADAPTER="${1:-outputs/qwen3-vl-8b/dgm4-instruct/lora-sft}"
BATCH_SIZE="${2:-24}"
LIMIT="${3:-0}"

cleanup() {
    local code=$?
    if [ ${code} -eq 0 ]; then
        autodl notify "DGM4 评测成功 (batch=${BATCH_SIZE})" 2>&1 || true
    else
        autodl notify "DGM4 评测失败 (exit=${code}, batch=${BATCH_SIZE})" 2>&1 || true
    fi
    sleep 30
    autodl halt 2>&1 || shutdown -h now 2>&1 || true
}
trap cleanup EXIT

echo "================================================"
echo "  DGM4 Evaluation | Adapter: ${ADAPTER}"
echo "  batch_size=${BATCH_SIZE} | limit=${LIMIT}"
echo "  Started: $(date)"
echo "================================================"

cd /root/autodl-tmp/DGM4_Project/training_models

HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/root/miniconda3/bin/python3 eval_model_batch.py \
    --adapter "${ADAPTER}" \
    --batch-size "${BATCH_SIZE}" \
    --limit "${LIMIT}"
