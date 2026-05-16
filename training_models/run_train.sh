#!/bin/bash
# DGM4 Training Launcher
# Usage: bash run_train.sh <config_name>
# Before first use: autodl key <your_sendkey>

set -e
CONFIG_NAME="${1:-qwen3vl_8b_lora_sft_dgm4_instruct_autodl.yaml}"
CONFIG_PATH="../../configs/${CONFIG_NAME}"

echo "================================================"
echo "  DGM4 Training"
echo "  Config: ${CONFIG_NAME}"
echo "  Started: $(date)"
echo "================================================"

cd /root/autodl-tmp/DGM4_Project/training_models/src/LLaMA-Factory

HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
/root/miniconda3/bin/llamafactory-cli train "${CONFIG_PATH}" && \
autodl notify "DGM4 训练成功: ${CONFIG_NAME}" && \
autodl halt || \
autodl notify "DGM4 训练失败: ${CONFIG_NAME}"
