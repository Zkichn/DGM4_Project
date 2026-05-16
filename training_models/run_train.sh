#!/bin/bash
# DGM4 Training Launcher — guaranteed shutdown via trap
# Usage: bash run_train.sh <config_name>
# Before first use: autodl key <your_sendkey>

CONFIG_NAME="${1:-qwen3vl_8b_lora_sft_dgm4_instruct_autodl.yaml}"
CONFIG_PATH="../../configs/${CONFIG_NAME}"
EXIT_CODE=1

# trap: guarantees notification + shutdown no matter how script exits
cleanup() {
    local code=$?
    if [ ${code} -eq 0 ]; then
        autodl notify "DGM4 训练成功: ${CONFIG_NAME}" 2>&1 || true
    else
        autodl notify "DGM4 训练失败 (exit=${code}): ${CONFIG_NAME}" 2>&1 || true
    fi
    sleep 30
    autodl halt 2>&1 || shutdown -h now 2>&1 || true
}
trap cleanup EXIT

echo "================================================"
echo "  DGM4 Training | Config: ${CONFIG_NAME}"
echo "  Started: $(date)"
echo "================================================"

cd /root/autodl-tmp/DGM4_Project/training_models/src/LLaMA-Factory

HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
/root/miniconda3/bin/llamafactory-cli train "${CONFIG_PATH}"
