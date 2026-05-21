#!/bin/bash
# DGM4 curriculum training + final 12-metric evaluation launcher.
# Usage: bash run_curriculum_train_eval.sh [eval_batch_size] [eval_limit]
# Before first use on AutoDL: autodl key <your_sendkey>

set -euo pipefail

EVAL_BATCH_SIZE="${1:-24}"
EVAL_LIMIT="${2:-0}"

PROJECT_DIR="/root/autodl-tmp/DGM4_Project"
LOG_DIR="${PROJECT_DIR}/training_models/logs"
mkdir -p "${LOG_DIR}"
if [ -z "${DGM4_LOG_REDIRECTED:-}" ]; then
    export DGM4_LOG_REDIRECTED=1
    LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_run_curriculum_train_eval.log"
    echo "${LOG_FILE}" > "${LOG_DIR}/run_curriculum_train_eval.latest.logpath"
    echo "$$" > "${LOG_DIR}/run_curriculum_train_eval.pid"
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi
TRAIN_DIR="${PROJECT_DIR}/training_models/src/LLaMA-Factory"
CONFIG_DIR="../../configs"
FINAL_ADAPTER="${PROJECT_DIR}/training_models/outputs/qwen3-vl-8b/dgm4-curriculum/stage3-final"
FINAL_OUTPUT="${FINAL_ADAPTER}/eval_dgm4_12metrics.json"

cleanup() {
    local code=$?
    if [ "${code}" -eq 0 ]; then
        autodl notify "DGM4 三阶段训练和12指标测评完成" 2>&1 || true
    else
        autodl notify "DGM4 三阶段训练/测评失败 (exit=${code})" 2>&1 || true
    fi
    sleep 30
    autodl halt 2>&1 || shutdown -h now 2>&1 || true
}
trap cleanup EXIT

run_train() {
    local config_name="$1"
    echo "================================================"
    echo "  Training: ${config_name}"
    echo "  Started: $(date)"
    echo "================================================"
    cd "${TRAIN_DIR}"
    HF_HOME=/root/autodl-tmp/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    /root/miniconda3/bin/llamafactory-cli train "${CONFIG_DIR}/${config_name}"
}

run_train "qwen3vl_8b_lora_curriculum_stage1_cls_autodl.yaml"
run_train "qwen3vl_8b_lora_curriculum_stage2_grounding_autodl.yaml"
run_train "qwen3vl_8b_lora_curriculum_stage3_final_autodl.yaml"

echo "================================================"
echo "  Final evaluation: 12 DGM4 metrics"
echo "  Adapter: ${FINAL_ADAPTER}"
echo "  Started: $(date)"
echo "================================================"
cd "${PROJECT_DIR}/training_models"
HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/root/miniconda3/bin/python3 eval_dgm4_qwen3vl.py \
    --adapter "${FINAL_ADAPTER}" \
    --test-data "datasets/dgm4_stage3_final/test.json" \
    --media-dir "/root/autodl-tmp/datasets" \
    --output "${FINAL_OUTPUT}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --limit "${EVAL_LIMIT}"

echo "================================================"
echo "  Done: $(date)"
echo "  Metrics: ${FINAL_ADAPTER}/eval_dgm4_12metrics_summary.json"
echo "================================================"
