#!/bin/bash
# Download assets required by DGM4-Instruct configs on AutoDL/SeetaCloud.
# Usage:
#   bash training_models/setup_autodl_assets.sh
#
# Expected paths used by configs:
#   DGM4 images/metadata: /root/autodl-tmp/datasets/DGM4
#   Qwen3-VL cache:       /root/autodl-tmp/hf-cache

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets}"
HF_HOME_DIR="${HF_HOME:-/root/autodl-tmp/hf-cache}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://hf-mirror.com}"
DGM4_DIR="${DATA_ROOT}/DGM4"

echo "================================================"
echo "  DGM4-Instruct Asset Setup"
echo "  DATA_ROOT: ${DATA_ROOT}"
echo "  HF_HOME:   ${HF_HOME_DIR}"
echo "  HF_ENDPOINT: ${HF_ENDPOINT_VALUE}"
echo "================================================"

mkdir -p "${DATA_ROOT}" "${HF_HOME_DIR}"

echo "[1/4] Checking tools..."
if ! command -v git >/dev/null 2>&1; then
  echo "git is missing. Installing git..."
  apt-get update
  apt-get install -y git
fi

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs is missing. Installing git-lfs..."
  apt-get update
  apt-get install -y git-lfs
fi

git lfs install

echo "[2/4] Installing Hugging Face utilities if needed..."
python - <<'PY'
import importlib.util
import subprocess
import sys

missing = []
for pkg in ("huggingface_hub", "hf_transfer"):
    if importlib.util.find_spec(pkg) is None:
        missing.append(pkg.replace("_", "-"))

if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *missing, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
PY

echo "[3/4] Downloading DGM4 dataset..."
cd "${DATA_ROOT}"
if [ -d "${DGM4_DIR}/.git" ]; then
  echo "DGM4 repo already exists. Pulling latest files..."
  cd "${DGM4_DIR}"
  git pull
else
  echo "Cloning DGM4 dataset into ${DGM4_DIR}..."
  GIT_LFS_SKIP_SMUDGE=1 git clone "https://huggingface.co/datasets/rshaojimmy/DGM4" "${DGM4_DIR}" || \
  GIT_LFS_SKIP_SMUDGE=1 git clone "https://hf-mirror.com/datasets/rshaojimmy/DGM4" "${DGM4_DIR}"
  cd "${DGM4_DIR}"
fi

git lfs pull

echo "[4/4] Downloading Qwen3-VL-8B-Instruct into HF cache..."
export HF_HOME="${HF_HOME_DIR}"
export HF_ENDPOINT="${HF_ENDPOINT_VALUE}"
export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --repo-type model --resume-download

echo "================================================"
echo "  Done."
echo "  Check dataset:"
echo "    ls ${DGM4_DIR}"
echo "  Check model cache:"
echo "    ls ${HF_HOME_DIR}/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"
echo "================================================"
