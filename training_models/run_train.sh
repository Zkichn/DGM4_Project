#!/bin/bash
# DGM4 Training Launcher
# Usage: bash run_train.sh <config_name>
# Before first use: autodl key <your_sendkey>

set -e
CONFIG_NAME="${1:-qwen3vl_8b_lora_sft_dgm4_instruct_autodl.yaml}"
CONFIG_PATH="../../configs/${CONFIG_NAME}"
PROJECT_DIR="/root/autodl-tmp/DGM4_Project"
LOG_DIR="${PROJECT_DIR}/training_models/logs"
mkdir -p "${LOG_DIR}"
if [ -z "${DGM4_LOG_REDIRECTED:-}" ]; then
    export DGM4_LOG_REDIRECTED=1
    SAFE_CONFIG_NAME="${CONFIG_NAME//[^A-Za-z0-9_.-]/_}"
    LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_run_train_${SAFE_CONFIG_NAME}.log"
    echo "${LOG_FILE}" > "${LOG_DIR}/run_train.latest.logpath"
    echo "$$" > "${LOG_DIR}/run_train.pid"
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

echo "================================================"
echo "  DGM4 Training"
echo "  Config: ${CONFIG_NAME}"
echo "  Started: $(date)"
echo "================================================"

cd "${PROJECT_DIR}/training_models/src/LLaMA-Factory"

HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
/root/miniconda3/bin/llamafactory-cli train "${CONFIG_PATH}" && \
autodl notify "DGM4 训练成功: ${CONFIG_NAME}" && \
autodl halt || \
autodl notify "DGM4 训练失败: ${CONFIG_NAME}"
