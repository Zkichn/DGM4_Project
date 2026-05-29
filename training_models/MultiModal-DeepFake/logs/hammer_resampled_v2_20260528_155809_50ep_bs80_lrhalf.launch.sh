#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/DGM4_Project/training_models/MultiModal-DeepFake
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
PY=/root/miniconda3/bin/python
PORT=29720
RUN_LOG="logs/hammer_resampled_v2_20260528_155809_50ep_bs80_lrhalf.log"
echo "[INFO] resampled train start \Thu May 28 15:58:09 CST 2026 exp=hammer_resampled_v2_20260528_155809_50ep_bs80_lrhalf" | tee "$RUN_LOG"
echo "[INFO] config=train_resampled_v2 bs=80 lr=1e-5 lr_img=5e-5 warmup=2 shutdown=0" | tee -a "$RUN_LOG"
$PY train.py \
  --config configs/hammer_resampled_v2_train.yaml \
  --output_dir results \
  --checkpoint local_models/ALBEF_4M/ALBEF_4M.pth \
  --text_encoder local_models/bert-base-uncased-ms \
  --launcher pytorch \
  --rank 0 \
  --log_num "hammer_resampled_v2_20260528_155809_50ep_bs80_lrhalf" \
  --dist-url tcp://127.0.0.1:${PORT} \
  --token_momentum \
  --world_size 1 \
  --model_save_epoch 5 2>&1 | tee -a "$RUN_LOG"
echo "[INFO] resampled test best start \Thu May 28 15:58:09 CST 2026" | tee -a "$RUN_LOG"
$PY test.py \
  --config configs/hammer_resampled_v2_test.yaml \
  --output_dir results \
  --text_encoder local_models/bert-base-uncased-ms \
  --launcher pytorch \
  --rank 0 \
  --log_num "loghammer_resampled_v2_20260528_155809_50ep_bs80_lrhalf" \
  --dist-url tcp://127.0.0.1:$((PORT+1)) \
  --token_momentum \
  --world_size 1 \
  --test_epoch best 2>&1 | tee -a "$RUN_LOG"
echo "[INFO] resampled train+test done \Thu May 28 15:58:09 CST 2026, results=results/loghammer_resampled_v2_20260528_155809_50ep_bs80_lrhalf" | tee -a "$RUN_LOG"
