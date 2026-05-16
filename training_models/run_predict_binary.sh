#!/bin/bash
# DGM4-Instruct binary verdict prediction launcher.
# Usage: bash run_predict_binary.sh

set -e

cd /root/autodl-tmp/DGM4_Project/training_models/src/LLaMA-Factory

HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
/root/miniconda3/bin/llamafactory-cli train ../../configs/qwen3vl_8b_lora_predict_dgm4_instruct_binary_autodl.yaml

cd /root/autodl-tmp/DGM4_Project

/root/miniconda3/bin/python training_models/scripts/evaluate_binary_verdict.py \
  --pred training_models/outputs/qwen3-vl-8b/dgm4-instruct/binary-test-predict/generated_predictions.jsonl \
  --gold training_models/datasets/dgm4_instruct/test.json \
  --out training_models/outputs/qwen3-vl-8b/dgm4-instruct/binary-test-predict/binary_metrics.json
