#!/bin/bash
# Open-source zero-shot DGM4 baselines (run with GPU). Same 2207 test split and
# metric 口径 as the fine-tuned evals. Does NOT auto-halt — inspect results first.
set -uo pipefail
cd /root/autodl-tmp/DGM4_Project/training_models || exit 1
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
TEST=datasets/dgm4_stage3_final/test.json
LLAVA=/root/autodl-tmp/DGM4_Project/training_models/base_models/llava-v1.6-mistral-7b-hf
LIM="${1:-0}"   # optional first arg: limit N samples for a smoke (0 = full)

run() { echo "============ $1 ============"; shift; "$@"; echo "rc=$? ($1)"; }

run "Qwen3-VL-8B zero-shot" \
  $PY eval_dgm4_qwen3vl.py --no-adapter --base-model Qwen/Qwen3-VL-8B-Instruct \
  --test-data "$TEST" --limit "$LIM" --output outputs/qwen3-vl-8b/dgm4-zeroshot/eval.json

run "InternVL3.5-8B zero-shot" \
  $PY eval_dgm4_internvl.py --no-adapter --base-model OpenGVLab/InternVL3_5-8B-HF \
  --test-data "$TEST" --limit "$LIM" --output outputs/internvl3_5-8b/dgm4-zeroshot/eval.json

run "LLaVA-NeXT zero-shot" \
  $PY eval_dgm4_llava.py --no-adapter --stage stage2 --base-model "$LLAVA" \
  --test-data "$TEST" --limit "$LIM" --output outputs/llava-v1.6/dgm4-zeroshot/eval.json

echo "ALL ZERO-SHOT EVALS DONE (limit=$LIM)"
