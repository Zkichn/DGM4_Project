#!/bin/bash
# Clean Stage1-v2 -> Stage2-v2 chain training and evaluation.
# This preserves previous stage2-grounding-v2 results by writing to *-chain dirs.
# Usage: bash run_curriculum_v2_chain_train_eval.sh [eval_batch_size] [eval_limit] [shutdown_when_done]

set -euo pipefail

EVAL_BATCH_SIZE="${1:-16}"
EVAL_LIMIT="${2:-0}"
SHUTDOWN_WHEN_DONE="${3:-0}"

PROJECT_DIR="/root/autodl-tmp/DGM4_Project"
TRAINING_DIR="${PROJECT_DIR}/training_models"
LOG_DIR="${TRAINING_DIR}/logs"
mkdir -p "${LOG_DIR}"
if [ -z "${DGM4_LOG_REDIRECTED:-}" ]; then
    export DGM4_LOG_REDIRECTED=1
    LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_run_curriculum_v2_chain_train_eval.log"
    echo "${LOG_FILE}" > "${LOG_DIR}/run_curriculum_v2_chain_train_eval.latest.logpath"
    echo "$$" > "${LOG_DIR}/run_curriculum_v2_chain_train_eval.pid"
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

TRAIN_DIR="${TRAINING_DIR}/src/LLaMA-Factory"
CONFIG_DIR="../../configs"
BASE_MODEL="${TRAINING_DIR}/base_models/Qwen3-VL-8B-Instruct"
STAGE1_ADAPTER="${TRAINING_DIR}/outputs/qwen3-vl-8b/dgm4-curriculum/stage1-cls-v2-chain"
STAGE2_ADAPTER="${TRAINING_DIR}/outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding-v2-chain"
STAGE1_OUTPUT="${STAGE1_ADAPTER}/eval_stage1_v2_chain_binary_multilabel.json"
STAGE2_OUTPUT="${STAGE2_ADAPTER}/eval_stage2_v2_chain_binary_multilabel_grounding.json"

cleanup() {
    local code=$?
    if [ "${code}" -eq 0 ]; then
        command -v autodl >/dev/null 2>&1 && autodl notify "DGM4 v2 chain stage1/stage2 training and eval completed" 2>&1 || true
        if [ "${SHUTDOWN_WHEN_DONE}" = "1" ]; then
            sleep 30
            shutdown -h now 2>&1 || true
        fi
    else
        command -v autodl >/dev/null 2>&1 && autodl notify "DGM4 v2 chain failed (exit=${code})" 2>&1 || true
        echo "DGM4 v2 chain failed with exit=${code}. Server left running for debugging." >&2
    fi
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
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    /root/miniconda3/bin/llamafactory-cli train "${CONFIG_DIR}/${config_name}"
}

cd "${TRAINING_DIR}"
echo "================================================"
echo "  Building curriculum v2 datasets"
echo "================================================"
/root/miniconda3/bin/python build_curriculum_v2_datasets.py --root "${TRAINING_DIR}"

run_train "qwen3vl_8b_lora_curriculum_stage1_cls_v2_chain_autodl.yaml"

echo "================================================"
echo "  Evaluating Stage1-v2-chain"
echo "  Adapter: ${STAGE1_ADAPTER}"
echo "================================================"
cd "${TRAINING_DIR}"
HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/root/miniconda3/bin/python eval_curriculum_stage_outputs.py \
    --stage stage1 \
    --adapter "${STAGE1_ADAPTER}" \
    --base-model "${BASE_MODEL}" \
    --test-data "datasets/dgm4_stage1_cls_v2/test.json" \
    --media-dir "/root/autodl-tmp/datasets" \
    --output "${STAGE1_OUTPUT}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --limit "${EVAL_LIMIT}" \
    --max-tokens 80

run_train "qwen3vl_8b_lora_curriculum_stage2_grounding_v2_chain_autodl.yaml"

echo "================================================"
echo "  Evaluating Stage2-v2-chain"
echo "  Adapter: ${STAGE2_ADAPTER}"
echo "================================================"
cd "${TRAINING_DIR}"
HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/root/miniconda3/bin/python eval_curriculum_stage_outputs.py \
    --stage stage2 \
    --adapter "${STAGE2_ADAPTER}" \
    --base-model "${BASE_MODEL}" \
    --test-data "datasets/dgm4_stage2_grounding_v2/test.json" \
    --media-dir "/root/autodl-tmp/datasets" \
    --output "${STAGE2_OUTPUT}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --limit "${EVAL_LIMIT}" \
    --max-tokens 140

echo "================================================"
echo "  Done: $(date)"
echo "  Stage1 metrics: ${STAGE1_ADAPTER}/eval_stage1_v2_chain_binary_multilabel_summary.json"
echo "  Stage2 metrics: ${STAGE2_ADAPTER}/eval_stage2_v2_chain_binary_multilabel_grounding_summary.json"
echo "================================================"
