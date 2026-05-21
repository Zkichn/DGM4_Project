#!/bin/bash
# Build curriculum v2 datasets, continue Stage1/Stage2 training, then evaluate Stage2-v2.
# Usage: bash run_curriculum_v2_stage12_train_eval.sh [eval_batch_size] [eval_limit] [shutdown_when_done]

set -euo pipefail

EVAL_BATCH_SIZE="${1:-16}"
EVAL_LIMIT="${2:-0}"
SHUTDOWN_WHEN_DONE="${3:-0}"

PROJECT_DIR="/root/autodl-tmp/DGM4_Project"
TRAINING_DIR="${PROJECT_DIR}/training_models"
TRAIN_DIR="${TRAINING_DIR}/src/LLaMA-Factory"
CONFIG_DIR="../../configs"
BASE_MODEL="${TRAINING_DIR}/base_models/Qwen3-VL-8B-Instruct"
STAGE2_V2_ADAPTER="${TRAINING_DIR}/outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding-v2"
STAGE2_V2_OUTPUT="${STAGE2_V2_ADAPTER}/eval_stage2_v2_binary_multilabel_grounding.json"

cleanup() {
    local code=$?
    if [ "${code}" -eq 0 ]; then
        command -v autodl >/dev/null 2>&1 && autodl notify "DGM4 curriculum v2 stage1/stage2 training and eval completed" 2>&1 || true
        if [ "${SHUTDOWN_WHEN_DONE}" = "1" ]; then
            sleep 30
            shutdown -h now 2>&1 || true
        fi
    else
        command -v autodl >/dev/null 2>&1 && autodl notify "DGM4 curriculum v2 failed (exit=${code})" 2>&1 || true
        echo "DGM4 curriculum v2 failed with exit=${code}. Server left running for debugging." >&2
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

run_train "qwen3vl_8b_lora_curriculum_stage1_cls_v2_continue_autodl.yaml"
run_train "qwen3vl_8b_lora_curriculum_stage2_grounding_v2_continue_autodl.yaml"

echo "================================================"
echo "  Evaluating Stage2-v2"
echo "  Adapter: ${STAGE2_V2_ADAPTER}"
echo "================================================"
cd "${TRAINING_DIR}"
HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/root/miniconda3/bin/python eval_curriculum_stage_outputs.py \
    --stage stage2 \
    --adapter "${STAGE2_V2_ADAPTER}" \
    --base-model "${BASE_MODEL}" \
    --test-data "datasets/dgm4_stage2_grounding_v2/test.json" \
    --media-dir "/root/autodl-tmp/datasets" \
    --output "${STAGE2_V2_OUTPUT}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --limit "${EVAL_LIMIT}" \
    --max-tokens 140

echo "================================================"
echo "  Done: $(date)"
echo "  Metrics: ${STAGE2_V2_ADAPTER}/eval_stage2_v2_binary_multilabel_grounding_summary.json"
echo "================================================"
