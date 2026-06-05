#!/bin/bash
# Run a 5-step bs=2 smoke right before Stage2-v2-chain training.
# Measures GPU peak; if safe, sed-edits the real Stage2 YAML to bs=2+accum=4.
# Called from eval_dgm4_llava.py at the end of Stage1-v2-chain eval.
# Always returns 0 to avoid breaking the pipeline.

set -u

PROJECT_DIR=/root/autodl-tmp/DGM4_Project
TRAINING_DIR=${PROJECT_DIR}/training_models
CONFIG_DIR=${TRAINING_DIR}/configs
LOG_DIR=/root/autodl-tmp/dl_logs
mkdir -p "${LOG_DIR}"

# Sentinel: don't decide twice (launcher may call us, and the eval-script post-hook
# fallback may also call us at Stage 1-v2-chain eval -- only the first one decides).
SENTINEL=/tmp/llava_stage2_bs2_smoke_decided
if [ -f "${SENTINEL}" ]; then
    echo "[skip] smoke already decided previously (${SENTINEL} exists)."
    exit 0
fi

STAMP=$(date +%Y%m%d_%H%M%S)
SMOKE_LOG="${LOG_DIR}/stage2_bs2_smoke_${STAMP}.log"
MEM_LOG="${LOG_DIR}/stage2_bs2_smoke_${STAMP}_mem.log"

SMOKE_YAML=llava_next_mistral_lora_curriculum_stage2_bs2_smoke_autodl.yaml
STAGE1V2_REAL_YAML="${CONFIG_DIR}/llava_next_mistral_lora_curriculum_stage1_cls_v2_chain_autodl.yaml"
STAGE2_REAL_YAML="${CONFIG_DIR}/llava_next_mistral_lora_curriculum_stage2_grounding_v2_chain_autodl.yaml"

THRESHOLD_MIB=28000   # safe peak ceiling = 28 GB (out of 32 GB cap)

echo "================ Stage2 bs=2 smoke + decide ================" | tee -a "${SMOKE_LOG}"
echo "log: ${SMOKE_LOG}" | tee -a "${SMOKE_LOG}"
echo "mem: ${MEM_LOG}" | tee -a "${SMOKE_LOG}"

# wait a bit for parent (eval) process's GPU memory to drain
sleep 5
echo "GPU state before smoke:" | tee -a "${SMOKE_LOG}"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>&1 | tee -a "${SMOKE_LOG}"

# Background GPU memory sampler (every 1s)
(
    while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null >> "${MEM_LOG}"
        sleep 1
    done
) &
MON_PID=$!

# Run the smoke training (max_steps=5, bs=2)
cd "${TRAINING_DIR}/src/LLaMA-Factory"
HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
timeout 720 /root/miniconda3/bin/llamafactory-cli train \
    "${CONFIG_DIR}/${SMOKE_YAML}" >> "${SMOKE_LOG}" 2>&1
SMOKE_EXIT=$?

kill ${MON_PID} 2>/dev/null || true
sleep 1

# Compute peak memory observed
PEAK_MEM=0
if [ -s "${MEM_LOG}" ]; then
    PEAK_MEM=$(sort -n "${MEM_LOG}" | tail -1)
fi

echo "" | tee -a "${SMOKE_LOG}"
echo "================ smoke result ================" | tee -a "${SMOKE_LOG}"
echo "smoke exit code:  ${SMOKE_EXIT}" | tee -a "${SMOKE_LOG}"
echo "peak GPU memory:  ${PEAK_MEM} MiB" | tee -a "${SMOKE_LOG}"
echo "safe threshold:   ${THRESHOLD_MIB} MiB" | tee -a "${SMOKE_LOG}"

# Decide
if [ "${SMOKE_EXIT}" -eq 0 ] && [ "${PEAK_MEM}" -gt 0 ] && [ "${PEAK_MEM}" -lt "${THRESHOLD_MIB}" ]; then
    echo "[DECISION] bs=2 SAFE. Upgrading BOTH Stage1-v2-chain AND Stage2-v2-chain YAMLs to bs=2+accum=4." | tee -a "${SMOKE_LOG}"
    for YAML in "${STAGE1V2_REAL_YAML}" "${STAGE2_REAL_YAML}"; do
        sed -i 's/^per_device_train_batch_size: 1$/per_device_train_batch_size: 2/' "${YAML}"
        sed -i 's/^gradient_accumulation_steps: 9$/gradient_accumulation_steps: 4/' "${YAML}"
        echo "--- updated YAML: ${YAML} ---" | tee -a "${SMOKE_LOG}"
        grep -E '^(per_device_train_batch_size|gradient_accumulation_steps):' "${YAML}" | tee -a "${SMOKE_LOG}"
    done
    command -v autodl >/dev/null 2>&1 && autodl notify "Stage1-v2 + Stage2 bs=2 smoke PASS (peak ${PEAK_MEM}/${THRESHOLD_MIB} MiB). Both train YAMLs upgraded to bs=2+accum=4." || true
else
    echo "[DECISION] bs=2 UNSAFE or smoke failed. Keeping all YAMLs at bs=1." | tee -a "${SMOKE_LOG}"
    command -v autodl >/dev/null 2>&1 && autodl notify "Stage1-v2 + Stage2 bs=2 smoke FAIL/OOM-risk (peak ${PEAK_MEM} MiB, exit ${SMOKE_EXIT}). Keeping bs=1." || true
fi

# Clean smoke output (we don't want it to clutter)
rm -rf "${TRAINING_DIR}/outputs/_smoke_stage2_bs2" 2>/dev/null || true

# Mark sentinel so we don't decide again if called twice
touch "${SENTINEL}"

echo "================ smoke decide done ================" | tee -a "${SMOKE_LOG}"
exit 0
