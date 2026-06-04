#!/bin/bash
# LLaVA-NeXT-Mistral-7B 3-stage chain training and evaluation
# Modeled after run_curriculum_v2_chain_train_eval.sh, with:
#   - skip-completed-stage (top-level adapter_model.safetensors present)
#   - in-stage resume (latest checkpoint-N detected, YAML sed-patched)
# Usage: bash run_llava_chain_train_eval.sh [eval_batch_size] [eval_limit] [shutdown_when_done]

set -euo pipefail

EVAL_BATCH_SIZE="${1:-4}"          # LLaVA anyres needs smaller batch than Qwen3
EVAL_LIMIT="${2:-0}"
SHUTDOWN_WHEN_DONE="${3:-0}"

PROJECT_DIR="/root/autodl-tmp/DGM4_Project"
TRAINING_DIR="${PROJECT_DIR}/training_models"
LOG_DIR="${TRAINING_DIR}/logs"
mkdir -p "${LOG_DIR}"
if [ -z "${DGM4_LOG_REDIRECTED:-}" ]; then
    export DGM4_LOG_REDIRECTED=1
    LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_run_llava_chain_train_eval.log"
    echo "${LOG_FILE}" > "${LOG_DIR}/run_llava_chain_train_eval.latest.logpath"
    echo "$$" > "${LOG_DIR}/run_llava_chain_train_eval.pid"
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

TRAIN_DIR="${TRAINING_DIR}/src/LLaMA-Factory"
CONFIG_DIR="../../configs"
BASE_MODEL="${TRAINING_DIR}/base_models/llava-v1.6-mistral-7b-hf"
STAGE1_ADAPTER="${TRAINING_DIR}/outputs/llava-next-mistral-7b/dgm4-curriculum/stage1-cls"
STAGE1V2_ADAPTER="${TRAINING_DIR}/outputs/llava-next-mistral-7b/dgm4-curriculum/stage1-cls-v2-chain"
STAGE2_ADAPTER="${TRAINING_DIR}/outputs/llava-next-mistral-7b/dgm4-curriculum/stage2-grounding-v2-chain"
STAGE1_OUTPUT="${STAGE1_ADAPTER}/eval_stage1_llava_binary_multilabel.json"
STAGE1V2_OUTPUT="${STAGE1V2_ADAPTER}/eval_stage1_v2_chain_llava_binary_multilabel.json"
STAGE2_OUTPUT="${STAGE2_ADAPTER}/eval_stage2_v2_chain_llava_binary_multilabel_grounding.json"

cleanup() {
    local code=$?
    if [ "${code}" -eq 0 ]; then
        command -v autodl >/dev/null 2>&1 && autodl notify "LLaVA chain DONE" 2>&1 || true
    else
        command -v autodl >/dev/null 2>&1 && autodl notify "LLaVA chain FAILED (exit=${code})" 2>&1 || true
        echo "LLaVA chain failed exit=${code}." >&2
    fi
    if [ "${SHUTDOWN_WHEN_DONE}" = "1" ]; then
        sleep 30
        shutdown -h now 2>&1 || true
    fi
}
trap cleanup EXIT

# Train one stage. Skip if final adapter exists; otherwise resume from latest checkpoint if any.
run_train_with_resume() {
    local config_name="$1"
    local output_dir="$2"
    if [ -f "${output_dir}/adapter_model.safetensors" ]; then
        echo ">>> Skip ${config_name}: finalized adapter exists at ${output_dir}"
        command -v autodl >/dev/null 2>&1 && autodl notify "LLaVA: skip ${config_name}" 2>&1 || true
        return 0
    fi
    local config_path="${CONFIG_DIR}/${config_name}"
    local latest_ckpt
    latest_ckpt="$(ls -d ${output_dir}/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
    if [ -n "${latest_ckpt}" ] && [ -s "${latest_ckpt}/trainer_state.json" ]; then
        local tmp_config="${config_path%.yaml}.resume.yaml"
        cp "${config_path}" "${tmp_config}"
        sed -i "s|resume_from_checkpoint: null|resume_from_checkpoint: ${latest_ckpt}|" "${tmp_config}"
        config_path="${tmp_config}"
        echo ">>> Resume ${config_name} from ${latest_ckpt}"
        command -v autodl >/dev/null 2>&1 && autodl notify "LLaVA: resume ${config_name}" 2>&1 || true
    else
        echo ">>> Train ${config_name} from scratch"
        command -v autodl >/dev/null 2>&1 && autodl notify "LLaVA: train ${config_name} start" 2>&1 || true
    fi
    cd "${TRAIN_DIR}"
    HF_HOME=/root/autodl-tmp/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    /root/miniconda3/bin/llamafactory-cli train "${config_path}"
}

# Eval one stage. Skip if output JSON already exists.
run_eval_stage() {
    local stage="$1"
    local adapter="$2"
    local test_data="$3"
    local out_json="$4"
    local max_tokens="$5"
    local out_summary="${out_json%.json}_summary.json"
    if [ -s "${out_json}" ] && [ -s "${out_summary}" ]; then
        echo ">>> Skip eval ${stage}: ${out_summary} already exists"
        return 0
    fi
    cd "${TRAINING_DIR}"
    HF_HOME=/root/autodl-tmp/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /root/miniconda3/bin/python eval_dgm4_llava.py \
        --stage "${stage}" \
        --adapter "${adapter}" \
        --base-model "${BASE_MODEL}" \
        --test-data "${test_data}" \
        --media-dir "/root/autodl-tmp/datasets" \
        --output "${out_json}" \
        --batch-size "${EVAL_BATCH_SIZE}" \
        --limit "${EVAL_LIMIT}" \
        --max-tokens "${max_tokens}"
}

cd "${TRAINING_DIR}"

echo "================ Stage 1 train ================"
run_train_with_resume "llava_next_mistral_lora_curriculum_stage1_cls_autodl.yaml" "${STAGE1_ADAPTER}"
echo "================ Stage 1 eval ================"
run_eval_stage "stage1" "${STAGE1_ADAPTER}" "datasets/dgm4_stage1_cls/test.json" "${STAGE1_OUTPUT}" 80

echo "================ Pre-Stage1-v2 bs=2 smoke ================"
[ -x /root/autodl-tmp/DGM4_Project/training_models/run_stage2_bs2_smoke_decide.sh ] && bash /root/autodl-tmp/DGM4_Project/training_models/run_stage2_bs2_smoke_decide.sh || true
echo "================ Stage 1-v2-chain train ================"
run_train_with_resume "llava_next_mistral_lora_curriculum_stage1_cls_v2_chain_autodl.yaml" "${STAGE1V2_ADAPTER}"
echo "================ Stage 1-v2-chain eval ================"
run_eval_stage "stage1" "${STAGE1V2_ADAPTER}" "datasets/dgm4_stage1_cls_v2/test.json" "${STAGE1V2_OUTPUT}" 80

echo "================ Pre-Stage2 bs=4 smoke + EVAL_BATCH_SIZE=8 ================"
[ -x /root/autodl-tmp/DGM4_Project/training_models/run_stage2_bs4_smoke_decide.sh ] && bash /root/autodl-tmp/DGM4_Project/training_models/run_stage2_bs4_smoke_decide.sh || true
EVAL_BATCH_SIZE=8
echo "[INFO] EVAL_BATCH_SIZE bumped to ${EVAL_BATCH_SIZE} for subsequent test evals"
echo "================ Stage 2-v2-chain train ================"
run_train_with_resume "llava_next_mistral_lora_curriculum_stage2_grounding_v2_chain_autodl.yaml" "${STAGE2_ADAPTER}"
echo "================ Stage 2-v2-chain eval ================"
run_eval_stage "stage2" "${STAGE2_ADAPTER}" "datasets/dgm4_stage2_grounding_v2/test.json" "${STAGE2_OUTPUT}" 140

echo "================ ALL DONE: $(date) ================"
echo "Stage1 summary:       ${STAGE1_OUTPUT%.json}_summary.json"
echo "Stage1-v2-chain sum.: ${STAGE1V2_OUTPUT%.json}_summary.json"
echo "Stage2-v2-chain sum.: ${STAGE2_OUTPUT%.json}_summary.json"
