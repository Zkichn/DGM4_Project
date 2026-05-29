#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/DGM4_Project/training_models/MultiModal-DeepFake
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
PY=/root/miniconda3/bin/python
PORT=29740
EXP_ID="hammer_resampled_v2_resume_20260528_1646_bs75_lrhalf"
RUN_LOG="logs/${EXP_ID}.log"
echo "[INFO] resume latest best start $(date) exp=${EXP_ID} keep_dir=results/log${EXP_ID} save_every=10 delete_previous_periodic=1" | tee -a "$RUN_LOG"
echo "[INFO] checkpoint=results/log${EXP_ID}/checkpoint_best.pth current_best_epoch=7 bs=75 shutdown=0" | tee -a "$RUN_LOG"
$PY train.py \
  --config configs/hammer_resampled_v2_train.yaml \
  --output_dir results \
  --checkpoint "results/log${EXP_ID}/checkpoint_best.pth" \
  --resume True \
  --text_encoder local_models/bert-base-uncased-ms \
  --launcher pytorch \
  --rank 0 \
  --log_num "${EXP_ID}" \
  --dist-url tcp://127.0.0.1:${PORT} \
  --token_momentum \
  --world_size 1 \
  --model_save_epoch 10 2>&1 | tee -a "$RUN_LOG"
echo "[INFO] test best start $(date)" | tee -a "$RUN_LOG"
$PY test.py \
  --config configs/hammer_resampled_v2_test.yaml \
  --output_dir results \
  --text_encoder local_models/bert-base-uncased-ms \
  --launcher pytorch \
  --rank 0 \
  --log_num "log${EXP_ID}" \
  --dist-url tcp://127.0.0.1:$((PORT+1)) \
  --token_momentum \
  --world_size 1 \
  --test_epoch best 2>&1 | tee -a "$RUN_LOG"
echo "[INFO] resume latest best train+test done $(date), results=results/log${EXP_ID}" | tee -a "$RUN_LOG"
