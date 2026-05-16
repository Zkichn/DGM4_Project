#!/bin/bash
# DGM4 Training Launcher with notification + auto-shutdown
# Usage: bash run_train.sh <config_name>
# Example: bash run_train.sh qwen3vl_8b_lora_sft_dgm4_instruct_autodl.yaml

CONFIG_NAME="${1:-qwen3vl_8b_lora_sft_dgm4_instruct_autodl.yaml}"
CONFIG_PATH="../../configs/${CONFIG_NAME}"
LOG_FILE="../../logs/training_$(date +%Y%m%d_%H%M%S).log"
SENDKEY="${SENDKEY:-}"  # Server酱 SendKey, set via env or edit here

cd /root/autodl-tmp/DGM4_Project/training_models/src/LLaMA-Factory

echo "================================================"
echo "  DGM4 Training Launcher"
echo "  Config: ${CONFIG_PATH}"
echo "  Started: $(date)"
echo "================================================"

# Run training
HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
/root/miniconda3/bin/llamafactory-cli train "${CONFIG_PATH}" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    STATUS="✅ 训练成功"
else
    STATUS="❌ 训练失败 (exit ${EXIT_CODE})"
fi

echo "================================================"
echo "  ${STATUS}"
echo "  Finished: $(date)"
echo "================================================"

# Send notification via Server酱 (if SENDKEY is set)
if [ -n "${SENDKEY}" ]; then
    curl -s "https://sctapi.ftqq.com/${SENDKEY}.send" \
        -d "title=DGM4 ${STATUS}" \
        -d "desp=Config: ${CONFIG_NAME} | Exit: ${EXIT_CODE} | Time: $(date)" \
        > /dev/null 2>&1
    echo "Notification sent."
else
    echo "SENDKEY not set, skipping notification."
fi

# Auto shutdown (60s delay to allow log flush)
echo "Shutting down in 60 seconds..."
sleep 60
shutdown -h now
