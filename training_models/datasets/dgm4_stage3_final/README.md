# DGM4 Curriculum Datasets

These datasets are derived from `../dgm4_instruct` without changing the original annotations.

- Stage 1: binary verdict + four atomic labels (FS/FA/TS/TA)
- Stage 2: Stage 1 fields + image/text grounding
- Stage 3: final five-line output with category, grounding, and evidence

Use `train.json` for training and `val.json` for checkpoint selection. `test.json` is provided only for stage-specific sanity checks.
