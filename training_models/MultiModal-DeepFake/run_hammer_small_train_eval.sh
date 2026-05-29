#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-/root/miniconda3/bin/python}
EXP_ID=${EXP_ID:-hammer_small_$(date +%Y%m%d_%H%M%S)}
EPOCHS=${EPOCHS:-50}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-10}
BATCH_SIZE_TRAIN=${BATCH_SIZE_TRAIN:-16}
BATCH_SIZE_VAL=${BATCH_SIZE_VAL:-64}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-128}
NUM_GPU=${NUM_GPU:-1}
MODEL_SAVE_EPOCH=${MODEL_SAVE_EPOCH:-100}
SHUTDOWN_WHEN_DONE=${SHUTDOWN_WHEN_DONE:-0}
PORT=${PORT:-29531}

mkdir -p results logs
RUN_LOG="logs/${EXP_ID}.log"

on_exit() {
  status=$?
  if [[ ${status} -ne 0 ]]; then
    echo "[INFO] task failed with status=${status}; server left running for debugging" | tee -a "${RUN_LOG}"
  fi
  exit ${status}
}
trap on_exit EXIT

${PY} prepare_hammer_small_data.py \
  --source_dir ../datasets/dgm4_instruct \
  --output_dir hammer_small_dgm4/metadata | tee -a "${RUN_LOG}"

${PY} - <<PY
from pathlib import Path

def set_top_level_scalar(text, key, value):
    lines = text.splitlines()
    replaced = False
    for idx, line in enumerate(lines):
        if line.startswith(f'{key}:'):
            lines[idx] = f'{key}: {value}'
            replaced = True
            break
    if not replaced:
        lines.append(f'{key}: {value}')
    return '\n'.join(lines) + '\n'

def set_sched_scalar(text, key, value):
    marker = 'schedular:'
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith(marker):
            body = line[len(marker):].strip()
            if body.startswith('{') and body.endswith('}'):
                parts = [p.strip() for p in body[1:-1].split(',') if p.strip()]
                kv = {}
                order = []
                for part in parts:
                    k, v = part.split(':', 1)
                    k = k.strip(); v = v.strip()
                    kv[k] = v; order.append(k)
                if key not in kv:
                    order.append(key)
                kv[key] = str(value)
                lines[idx] = marker + ' {' + ', '.join(f'{k}: {kv[k]}' for k in order) + '}'
                return '\n'.join(lines) + '\n'
    raise RuntimeError('schedular line not found')

for path, val_bs in [('configs/hammer_small_train.yaml', '${BATCH_SIZE_VAL}'), ('configs/hammer_small_test.yaml', '${TEST_BATCH_SIZE}')]:
    p = Path(path)
    text = p.read_text()
    if path.endswith('train.yaml'):
        text = set_top_level_scalar(text, 'batch_size_train', '${BATCH_SIZE_TRAIN}')
    text = set_top_level_scalar(text, 'batch_size_val', val_bs)
    text = set_sched_scalar(text, 'epochs', '${EPOCHS}')
    text = set_sched_scalar(text, 'warmup_epochs', '${WARMUP_EPOCHS}')
    p.write_text(text)
PY
if [[ ! -f local_models/ALBEF_4M/ALBEF_4M.pth ]]; then
  echo "[INFO] downloading ALBEF_4M from ModelScope" | tee -a "${RUN_LOG}"
  ${PY} - <<'PY'
from modelscope import snapshot_download
snapshot_download('Zkichn/ALBEF_4M', local_dir='local_models/ALBEF_4M')
PY
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
HOST=127.0.0.1

echo "[INFO] training ${EXP_ID}: epochs=${EPOCHS}, bs_train=${BATCH_SIZE_TRAIN}, bs_val=${BATCH_SIZE_VAL}, gpu=${NUM_GPU}" | tee -a "${RUN_LOG}"
${PY} train.py \
  --config configs/hammer_small_train.yaml \
  --output_dir results \
  --checkpoint local_models/ALBEF_4M/ALBEF_4M.pth \
  --text_encoder local_models/bert-base-uncased-ms \
  --launcher pytorch \
  --rank 0 \
  --log_num "${EXP_ID}" \
  --dist-url tcp://${HOST}:${PORT} \
  --token_momentum \
  --world_size "${NUM_GPU}" \
  --model_save_epoch "${MODEL_SAVE_EPOCH}" 2>&1 | tee -a "${RUN_LOG}"

echo "[INFO] testing best checkpoint" | tee -a "${RUN_LOG}"
${PY} test.py \
  --config configs/hammer_small_test.yaml \
  --output_dir results \
  --text_encoder local_models/bert-base-uncased-ms \
  --launcher pytorch \
  --rank 0 \
  --log_num "log${EXP_ID}" \
  --dist-url tcp://${HOST}:$((PORT+1)) \
  --token_momentum \
  --world_size "${NUM_GPU}" \
  --test_epoch best 2>&1 | tee -a "${RUN_LOG}"

echo "[INFO] done. Results: results/log${EXP_ID}" | tee -a "${RUN_LOG}"
if [[ "${SHUTDOWN_WHEN_DONE}" == "1" ]]; then
  echo "[INFO] train+test completed successfully; shutting down" | tee -a "${RUN_LOG}"
  /sbin/shutdown -h now || true
fi
