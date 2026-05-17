#!/bin/bash
# DGM4 r2 Pipeline: continued fine-tuning + Table 2 eval + auto-halt
# Single unified trap so the machine halts ONCE after both phases.
# Usage: bash run_train_then_eval.sh [config_name] [adapter_path]

CONFIG_NAME="${1:-qwen3vl_8b_lora_sft_dgm4_instruct_autodl_r2.yaml}"
ADAPTER_PATH="${2:-outputs/qwen3-vl-8b/dgm4-instruct/lora-sft-fast-r2}"
PHASE="setup"
TRAIN_EXIT=-1
EVAL_EXIT=-1

cleanup() {
    local code=$?
    autodl notify "DGM4 r2 pipeline end: phase=${PHASE} train=${TRAIN_EXIT} eval=${EVAL_EXIT} exit=${code}" 2>&1 || true
    sleep 30
    autodl halt 2>&1 || shutdown -h now 2>&1 || true
}
trap cleanup EXIT

echo "================================================"
echo "  DGM4 r2 Pipeline"
echo "  Config:  ${CONFIG_NAME}"
echo "  Adapter: ${ADAPTER_PATH}"
echo "  Started: $(date)"
echo "================================================"

# ── Phase 1: Training ──
PHASE="train"
cd /root/autodl-tmp/DGM4_Project/training_models/src/LLaMA-Factory

HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
/root/miniconda3/bin/llamafactory-cli train "../../configs/${CONFIG_NAME}"
TRAIN_EXIT=$?

echo ""
echo "================================================"
echo "  Training finished. exit=${TRAIN_EXIT} at $(date)"
echo "================================================"

if [ ${TRAIN_EXIT} -ne 0 ]; then
    autodl notify "DGM4 r2 训练失败 (exit=${TRAIN_EXIT}), 跳过评测" 2>&1 || true
    exit ${TRAIN_EXIT}
fi

autodl notify "DGM4 r2 训练完成, 启动 Table 2 评测 (batch=24)" 2>&1 || true

# ── Phase 2: Eval ──
PHASE="eval"
cd /root/autodl-tmp/DGM4_Project/training_models

HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/root/miniconda3/bin/python3 eval_model_batch.py \
    --adapter "${ADAPTER_PATH}" \
    --batch-size 24 \
    --oom-fallback-batch 16
EVAL_EXIT=$?

echo ""
echo "================================================"
echo "  Eval finished. exit=${EVAL_EXIT} at $(date)"
echo "================================================"

exit ${EVAL_EXIT}
