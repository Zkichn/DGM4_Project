#!/bin/bash
# Full 2207 zero-shot for the two models already verified (LLaVA, InternVL),
# run while Qwen3-VL weights download in parallel.
set -uo pipefail
cd /root/autodl-tmp/DGM4_Project/training_models || exit 1
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/miniconda3/bin/python
TEST=datasets/dgm4_stage3_final/test.json
LLAVA=/root/autodl-tmp/DGM4_Project/training_models/base_models/llava-v1.6-mistral-7b-hf

echo "=== LLaVA-NeXT full ==="
$PY eval_dgm4_llava.py --no-adapter --stage stage2 --base-model "$LLAVA" \
  --test-data "$TEST" --output outputs/llava-v1.6/dgm4-zeroshot/eval.json
echo "rc=$? (llava)"

echo "=== InternVL3.5-8B full ==="
$PY eval_dgm4_internvl.py --no-adapter --base-model OpenGVLab/InternVL3_5-8B-HF \
  --test-data "$TEST" --output outputs/internvl3_5-8b/dgm4-zeroshot/eval.json
echo "rc=$? (internvl)"

echo "LLAVA+INTERNVL FULL DONE"
