#!/bin/bash
# Run a 10-step bs=4+accum=2 smoke right before Stage2-v2-chain training.
# Measures GPU peak; if safe, sed-edits ONLY the Stage2-v2-chain YAML (bs=2->4, accum=4->2).
# Does NOT touch Stage1-v2-chain YAML (Stage1-v2-chain is already training).
# Sentinel-guarded so it runs at most once.

set -u

PROJECT_DIR=/root/autodl-tmp/DGM4_Project
TRAINING_DIR=${PROJECT_DIR}/training_models
CONFIG_DIR=${TRAINING_DIR}/configs
LOG_DIR=/root/autodl-tmp/dl_logs
mkdir -p "${LOG_DIR}"

# Distinct sentinel from the bs=2 decide
SENTINEL=/tmp/llava_stage2_bs4_smoke_decided
if [ -f "${SENTINEL}" ]; then
    echo "[skip] Stage2 bs=4 smoke already decided previously (${SENTINEL} exists)."
    exit 0
fi

STAMP=$(date +%Y%m%d_%H%M%S)
SMOKE_LOG="${LOG_DIR}/stage2_bs4_smoke_${STAMP}.log"
MEM_LOG="${LOG_DIR}/stage2_bs4_smoke_${STAMP}_mem.log"

SMOKE_YAML=llava_next_mistral_lora_curriculum_stage2_bs4_smoke_autodl.yaml
STAGE2_REAL_YAML="${CONFIG_DIR}/llava_next_mistral_lora_curriculum_stage2_grounding_v2_chain_autodl.yaml"

THRESHOLD_MIB=29000   # bs=4 推测 peak ~29 GB, threshold 29000 留 ~3GB 余量 (32GB cap)

echo "================ Stage2 bs=4 smoke + decide ================" | tee -a "${SMOKE_LOG}"
echo "log: ${SMOKE_LOG}" | tee -a "${SMOKE_LOG}"
echo "mem: ${MEM_LOG}" | tee -a "${SMOKE_LOG}"

# wait for parent (eval) process's GPU memory to drain
sleep 8
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

# Run smoke training (max_steps=10, bs=4+accum=2)
cd "${TRAINING_DIR}/src/LLaMA-Factory"
HF_HOME=/root/autodl-tmp/hf-cache \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
timeout 900 /root/miniconda3/bin/llamafactory-cli train \
    "${CONFIG_DIR}/${SMOKE_YAML}" >> "${SMOKE_LOG}" 2>&1
SMOKE_EXIT=$?

kill ${MON_PID} 2>/dev/null || true
sleep 1

# Compute peak memory
PEAK_MEM=0
if [ -s "${MEM_LOG}" ]; then
    PEAK_MEM=$(sort -n "${MEM_LOG}" | tail -1)
fi

echo "" | tee -a "${SMOKE_LOG}"
echo "================ Stage2 bs=4 smoke result ================" | tee -a "${SMOKE_LOG}"
echo "smoke exit code:  ${SMOKE_EXIT}" | tee -a "${SMOKE_LOG}"
echo "peak GPU memory:  ${PEAK_MEM} MiB" | tee -a "${SMOKE_LOG}"
echo "safe threshold:   ${THRESHOLD_MIB} MiB" | tee -a "${SMOKE_LOG}"

if [ "${SMOKE_EXIT}" -eq 0 ] && [ "${PEAK_MEM}" -gt 0 ] && [ "${PEAK_MEM}" -lt "${THRESHOLD_MIB}" ]; then
    echo "[DECISION] bs=4 SAFE. Upgrading Stage2-v2-chain YAML to bs=4+accum=2." | tee -a "${SMOKE_LOG}"
    # Stage 2 YAML currently has bs=2 + accum=4 (from previous bs=2 smoke decision)
    sed -i 's/^per_device_train_batch_size: 2$/per_device_train_batch_size: 4/' "${STAGE2_REAL_YAML}"
    sed -i 's/^gradient_accumulation_steps: 4$/gradient_accumulation_steps: 2/' "${STAGE2_REAL_YAML}"
    echo "--- updated Stage2 YAML batch lines ---" | tee -a "${SMOKE_LOG}"
    grep -E '^(per_device_train_batch_size|gradient_accumulation_steps):' "${STAGE2_REAL_YAML}" | tee -a "${SMOKE_LOG}"
    command -v autodl >/dev/null 2>&1 && autodl notify "Stage2 bs=4 smoke PASS (peak ${PEAK_MEM}/${THRESHOLD_MIB} MiB). Stage2 YAML upgraded to bs=4+accum=2." || true
else
    echo "[DECISION] bs=4 UNSAFE or smoke failed. Stage2 YAML stays at bs=2+accum=4." | tee -a "${SMOKE_LOG}"
    command -v autodl >/dev/null 2>&1 && autodl notify "Stage2 bs=4 smoke FAIL/OOM-risk (peak ${PEAK_MEM} MiB, exit ${SMOKE_EXIT}). Stage2 stays at bs=2+accum=4." || true
fi

# Clean smoke output
rm -rf "${TRAINING_DIR}/outputs/_smoke_stage2_bs4" 2>/dev/null || true

touch "${SENTINEL}"

echo "================ Stage2 bs=4 smoke decide done ================" | tee -a "${SMOKE_LOG}"
exit 0
