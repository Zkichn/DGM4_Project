#!/bin/bash
# Continue Stage2-v2-chain with DGM4 field-aware weighted SFT loss for 0.3 epoch, then evaluate on test.
# Usage: bash run_stage2_v2_chain_wloss_train_eval.sh [eval_batch_size] [eval_limit] [shutdown_when_done]

set -euo pipefail

EVAL_BATCH_SIZE="${1:-16}"
EVAL_LIMIT="${2:-0}"
SHUTDOWN_WHEN_DONE="${3:-1}"

PROJECT_DIR="/root/autodl-tmp/DGM4_Project"
TRAINING_DIR="${PROJECT_DIR}/training_models"
LOG_DIR="${TRAINING_DIR}/logs"
mkdir -p "${LOG_DIR}"
if [ -z "${DGM4_LOG_REDIRECTED:-}" ]; then
    export DGM4_LOG_REDIRECTED=1
    LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_run_stage2_v2_chain_wloss_train_eval.log"
    echo "${LOG_FILE}" > "${LOG_DIR}/run_stage2_v2_chain_wloss_train_eval.latest.logpath"
    echo "$$" > "${LOG_DIR}/run_stage2_v2_chain_wloss_train_eval.pid"
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

cleanup() {
    local code=$?
    if [ "${code}" -eq 0 ]; then
        command -v autodl >/dev/null 2>&1 && autodl notify "DGM4 Stage2-v2-chain weighted-loss training and eval completed" 2>&1 || true
    else
        command -v autodl >/dev/null 2>&1 && autodl notify "DGM4 Stage2-v2-chain weighted-loss failed (exit=${code})" 2>&1 || true
        echo "DGM4 Stage2-v2-chain weighted-loss run failed with exit=${code}." >&2
    fi

    if [ "${SHUTDOWN_WHEN_DONE}" = "1" ]; then
        sleep 30
        shutdown -h now 2>&1 || true
    fi
}
trap cleanup EXIT

TRAIN_DIR="${TRAINING_DIR}/src/LLaMA-Factory"
CONFIG_DIR="../../configs"
CONFIG_NAME="qwen3vl_8b_lora_curriculum_stage2_grounding_v2_chain_wloss_0p3ep_autodl.yaml"
BASE_MODEL="${TRAINING_DIR}/base_models/Qwen3-VL-8B-Instruct"
SOURCE_ADAPTER="${TRAINING_DIR}/outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding-v2-chain/checkpoint-6500"
WLOSS_ADAPTER="${TRAINING_DIR}/outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding-v2-chain-wloss-0p3ep"
WLOSS_OUTPUT="${WLOSS_ADAPTER}/eval_stage2_v2_chain_wloss_0p3ep_binary_multilabel_grounding.json"

echo "================================================"
echo "  Stage2-v2-chain weighted-loss 0.3 epoch"
echo "  Config: ${CONFIG_NAME}"
echo "  Source adapter: ${SOURCE_ADAPTER}"
echo "  Started: $(date)"
echo "================================================"

test -s "${SOURCE_ADAPTER}/adapter_model.safetensors"

bash "${TRAINING_DIR}/apply_dgm4_field_weighted_loss_patch.sh"

cd "${TRAIN_DIR}"
HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
/root/miniconda3/bin/llamafactory-cli train "${CONFIG_DIR}/${CONFIG_NAME}"

echo "================================================"
echo "  Evaluating weighted-loss adapter"
echo "  Adapter: ${WLOSS_ADAPTER}"
echo "================================================"

cd "${TRAINING_DIR}"
HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/root/miniconda3/bin/python eval_curriculum_stage_outputs.py \
    --stage stage2 \
    --adapter "${WLOSS_ADAPTER}" \
    --base-model "${BASE_MODEL}" \
    --test-data "datasets/dgm4_stage2_grounding_v2/test.json" \
    --media-dir "/root/autodl-tmp/datasets" \
    --output "${WLOSS_OUTPUT}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --limit "${EVAL_LIMIT}" \
    --max-tokens 140

echo "================================================"
echo "  Done: $(date)"
echo "  Metrics: ${WLOSS_ADAPTER}/eval_stage2_v2_chain_wloss_0p3ep_binary_multilabel_grounding_summary.json"
echo "================================================"
