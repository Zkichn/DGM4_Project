#!/bin/bash
# DGM4-Instruct binary verdict prediction launcher.
# Usage: bash run_predict_binary.sh

set -e

PROJECT_DIR="/root/autodl-tmp/DGM4_Project"
LOG_DIR="${PROJECT_DIR}/training_models/logs"
mkdir -p "${LOG_DIR}"
if [ -z "${DGM4_LOG_REDIRECTED:-}" ]; then
    export DGM4_LOG_REDIRECTED=1
    LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_run_predict_binary.log"
    echo "${LOG_FILE}" > "${LOG_DIR}/run_predict_binary.latest.logpath"
    echo "$$" > "${LOG_DIR}/run_predict_binary.pid"
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

cd "${PROJECT_DIR}/training_models/src/LLaMA-Factory"

HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
/root/miniconda3/bin/llamafactory-cli train ../../configs/qwen3vl_8b_lora_predict_dgm4_instruct_binary_autodl.yaml

cd "${PROJECT_DIR}"

/root/miniconda3/bin/python training_models/scripts/evaluate_binary_verdict.py \
  --pred training_models/outputs/qwen3-vl-8b/dgm4-instruct/binary-test-predict/generated_predictions.jsonl \
  --gold training_models/datasets/dgm4_instruct/test.json \
  --out training_models/outputs/qwen3-vl-8b/dgm4-instruct/binary-test-predict/binary_metrics.json
