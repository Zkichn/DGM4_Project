#!/bin/bash
# Resume Stage2-v2-chain from checkpoint-5500, run to step 6500, then evaluate on test.
# Usage: bash run_stage2_v2_chain_resume_eval.sh [eval_batch_size] [eval_limit] [shutdown_when_done]

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
    LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_run_stage2_v2_chain_resume_eval.log"
    echo "${LOG_FILE}" > "${LOG_DIR}/run_stage2_v2_chain_resume_eval.latest.logpath"
    echo "$$" > "${LOG_DIR}/run_stage2_v2_chain_resume_eval.pid"
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

cleanup() {
    local code=$?
    if [ "${code}" -eq 0 ]; then
        command -v autodl >/dev/null 2>&1 && autodl notify "DGM4 Stage2-v2-chain resume and eval completed" 2>&1 || true
    else
        command -v autodl >/dev/null 2>&1 && autodl notify "DGM4 Stage2-v2-chain resume/eval failed (exit=${code})" 2>&1 || true
        echo "DGM4 Stage2-v2-chain resume/eval failed with exit=${code}." >&2
    fi

    if [ "${SHUTDOWN_WHEN_DONE}" = "1" ]; then
        sleep 30
        shutdown -h now 2>&1 || true
    fi
}
trap cleanup EXIT

TRAIN_DIR="${TRAINING_DIR}/src/LLaMA-Factory"
CONFIG_DIR="../../configs"
CONFIG_NAME="qwen3vl_8b_lora_curriculum_stage2_grounding_v2_chain_resume6500_autodl.yaml"
BASE_MODEL="${TRAINING_DIR}/base_models/Qwen3-VL-8B-Instruct"
STAGE2_ADAPTER="${TRAINING_DIR}/outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding-v2-chain"
STAGE2_OUTPUT="${STAGE2_ADAPTER}/eval_stage2_v2_chain_resume6500_binary_multilabel_grounding.json"
RESUME_CHECKPOINT="${STAGE2_ADAPTER}/checkpoint-5500"

echo "================================================"
echo "  Resume Stage2-v2-chain"
echo "  Config: ${CONFIG_NAME}"
echo "  Checkpoint: ${RESUME_CHECKPOINT}"
echo "  Started: $(date)"
echo "================================================"

test -d "${RESUME_CHECKPOINT}"
test -s "${RESUME_CHECKPOINT}/trainer_state.json"
test -s "${RESUME_CHECKPOINT}/adapter_model.safetensors"

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
echo "  Evaluating Stage2-v2-chain resumed final"
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
echo "  Metrics: ${STAGE2_ADAPTER}/eval_stage2_v2_chain_resume6500_binary_multilabel_grounding_summary.json"
echo "================================================"
