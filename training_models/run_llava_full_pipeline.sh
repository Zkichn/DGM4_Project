#!/bin/bash
# Full LLaVA pipeline: chain train+eval -> git commit + push -> autodl notify -> shutdown
# Designed to run autonomously (~10-15h). Per CLAUDE.md trap-EXIT pattern: any
# failure still triggers WeChat notify + shutdown to avoid wasting GPU.
# Usage: nohup bash run_llava_full_pipeline.sh > /root/autodl-tmp/dl_logs/full_pipeline.log 2>&1 &

set -euo pipefail

PROJECT_DIR=/root/autodl-tmp/DGM4_Project
TRAINING_DIR=${PROJECT_DIR}/training_models
LOG_DIR=${TRAINING_DIR}/logs
mkdir -p "${LOG_DIR}"

if [ -z "${DGM4_FULL_PIPE_REDIRECTED:-}" ]; then
    export DGM4_FULL_PIPE_REDIRECTED=1
    LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_run_llava_full_pipeline.log"
    echo "${LOG_FILE}" > "${LOG_DIR}/run_llava_full_pipeline.latest.logpath"
    echo "$$" > "${LOG_DIR}/run_llava_full_pipeline.pid"
    exec > >(tee -a "${LOG_FILE}") 2>&1
fi

PIPELINE_STATE="start"

cleanup() {
    local code=$?
    if [ "${code}" -eq 0 ] && [ "${PIPELINE_STATE}" = "pushed" ]; then
        command -v autodl >/dev/null 2>&1 && autodl notify "LLaVA pipeline DONE: chain+eval done, results pushed to GitHub. Shutting down in 30s." 2>&1 || true
    else
        command -v autodl >/dev/null 2>&1 && autodl notify "LLaVA pipeline FAILED at stage=${PIPELINE_STATE} (exit=${code}). Shutting down anyway in 30s; restart after debug." 2>&1 || true
    fi
    sleep 30
    shutdown -h now 2>&1 || true
}
trap cleanup EXIT

echo "================ LLaVA full pipeline start: $(date) ================"

# ---- Step 1: chain train + eval (no internal shutdown; we handle it) ----
PIPELINE_STATE="chain_train_eval"
echo ""
echo "================ Step 1: chain train+eval (3 stages) ================"
cd "${TRAINING_DIR}"
bash run_llava_chain_train_eval.sh 4 0 0

# ---- Step 2: git stage outputs ----
PIPELINE_STATE="git_stage"
echo ""
echo "================ Step 2: git stage outputs ================"
cd "${PROJECT_DIR}"

# Code (the 6 new files I created during prep)
git add \
    training_models/configs/llava_next_mistral_lora_curriculum_stage1_cls_autodl.yaml \
    training_models/configs/llava_next_mistral_lora_curriculum_stage1_cls_v2_chain_autodl.yaml \
    training_models/configs/llava_next_mistral_lora_curriculum_stage2_grounding_v2_chain_autodl.yaml \
    training_models/configs/llava_next_mistral_lora_curriculum_smoke_autodl.yaml \
    training_models/run_llava_chain_train_eval.sh \
    training_models/eval_dgm4_llava.py

# Per-stage finalized adapter + meta (NOT checkpoints, NOT optimizer states)
for stage in stage1-cls stage1-cls-v2-chain stage2-grounding-v2-chain; do
    out=training_models/outputs/llava-next-mistral-7b/dgm4-curriculum/${stage}
    for f in adapter_model.safetensors adapter_config.json train_results.json all_results.json trainer_state.json training_loss.png eval_results.json; do
        [ -f "${out}/${f}" ] && git add "${out}/${f}" || true
    done
done

# Eval result JSONs (with details + summary)
git add training_models/outputs/llava-next-mistral-7b/dgm4-curriculum/*/eval_*.json 2>/dev/null || true

echo "--- staged file stat ---"
git diff --cached --stat || true
echo "--- staged file list ---"
git diff --cached --name-only || true

# ---- Step 3: commit ----
PIPELINE_STATE="git_commit"
echo ""
echo "================ Step 3: git commit ================"
git commit -m "$(cat <<'CM_END'
add llava-next-mistral chain reproduction (3 stages + eval results)

LLaVA-NeXT-Mistral-7B base. LoRA r=16 alpha=32 dropout=0.05 target=all.
Per-stage hyperparameters mirror Qwen3 chain:
- Stage1 (stage1-cls):           lr=1e-4 warmup=0.1  2.0 ep
- Stage1-v2-chain:               lr=5e-5 warmup=0.05 1.5 ep
- Stage2-v2-chain:               lr=5e-5 warmup=0.05 2.0 ep
LLaVA-NeXT anyres + cutoff_len=4096; per_device_batch=1 with accum 8 or 9.
Vision tower frozen; LoRA only on LLM linear layers.
Eval (eval_dgm4_llava.py) shares metric formulas with eval_dgm4_qwen3vl.py
to keep 12-metric schema 1:1 comparable with Qwen3 chain results.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
CM_END
)"

# ---- Step 4: push ----
PIPELINE_STATE="git_push"
echo ""
echo "================ Step 4: git push ================"
git push origin hammer-small-data

PIPELINE_STATE="pushed"
echo ""
echo "================ ALL DONE: $(date) ================"
echo "Trap cleanup will fire next: WeChat notify + shutdown after 30s"
