#!/bin/bash
# Apply the local DGM4 field-aware weighted SFT loss patch to LLaMA-Factory.
# The patch is intentionally kept outside the LLaMA-Factory submodule so it can
# be enabled for this experiment and skipped/reverted for standard CE training.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/DGM4_Project}"
TRAIN_DIR="${PROJECT_DIR}/training_models/src/LLaMA-Factory"
PATCH_FILE="${PROJECT_DIR}/training_models/patches/dgm4_field_weighted_loss.patch"

if grep -q "use_dgm4_field_weighted_loss" "${TRAIN_DIR}/src/llamafactory/hparams/finetuning_args.py" \
    && grep -q "_compute_dgm4_field_weighted_loss" "${TRAIN_DIR}/src/llamafactory/train/sft/trainer.py"; then
    echo "DGM4 field-aware weighted loss patch already applied."
    exit 0
fi

echo "Applying DGM4 field-aware weighted loss patch..."
cd "${TRAIN_DIR}"
git apply "${PATCH_FILE}"
echo "DGM4 field-aware weighted loss patch applied."
