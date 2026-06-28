#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/DGM4_Project/training_models || exit 1
export HF_HOME=/root/autodl-tmp/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
TEST=datasets/dgm4_stage3_final/test.json
LLAVA=/root/autodl-tmp/DGM4_Project/training_models/base_models/llava-v1.6-mistral-7b-hf
mkdir -p outputs/qwen3-vl-8b/dgm4-zeroshot-indexed-full outputs/internvl3_5-8b/dgm4-zeroshot-indexed-full outputs/llava-v1.6/dgm4-zeroshot-indexed-full
echo "START $(date -Is)"
echo '=== Qwen indexed full ==='
$PY eval_dgm4_qwen3vl.py --no-adapter --base-model Qwen/Qwen3-VL-8B-Instruct --test-data "$TEST" --limit 0 --output outputs/qwen3-vl-8b/dgm4-zeroshot-indexed-full/eval.json --batch-size 4
echo '=== InternVL indexed full ==='
$PY eval_dgm4_internvl.py --no-adapter --base-model OpenGVLab/InternVL3_5-8B-HF --test-data "$TEST" --limit 0 --output outputs/internvl3_5-8b/dgm4-zeroshot-indexed-full/eval.json
echo '=== LLaVA indexed full ==='
$PY eval_dgm4_llava.py --no-adapter --explicit-prompt --stage stage2 --base-model "$LLAVA" --test-data "$TEST" --limit 0 --output outputs/llava-v1.6/dgm4-zeroshot-indexed-full/eval_explicit.json --batch-size 2
echo "DONE $(date -Is)"
