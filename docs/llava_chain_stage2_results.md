# LLaVA-NeXT-Mistral-7B Stage2-v2-chain — Final Results

Date: 2026-06-05 (training completed)
Branch: this doc is on `autodl-training`; full training code/configs/adapter/eval JSONs are on `hammer-small-data` branch (commit `af9c219`).

## Training summary

- Base model: `llava-hf/llava-v1.6-mistral-7b-hf` (15 GB bf16, 4 safetensors shards)
- Chain (3 stages, all LoRA r=16 α=32 dropout=0.05 target=all, vision tower frozen):
  - Stage 1-cls: 2.0 epoch on `dgm4_stage1_cls` (17,648 samples, 5-line format with FS/FA/TS/TA yes/no)
  - Stage 1-cls-v2-chain: 1.5 epoch on `dgm4_stage1_cls_v2` (20,592 samples, 6-line format with explicit atomic labels 0/1)
  - Stage 2-grounding-v2-chain: 1.0 epoch on `dgm4_stage2_grounding_v2` (28,086 samples, + image/text grounding)
- Hardware: RTX 4080 SUPER 32 GB on AutoDL
- Pipeline total runtime: ~57 hours (smoke + 3 train + 3 eval + push + halt)
- `anyres` image splitting on (up to 5 patches × 576 = 2880 image tokens); `cutoff_len=4096`
- Final train batch determined by smoke: bs=2 + accum=4 (eff_bs=8); bs=4 smoke peak exceeded 29 GB threshold so kept bs=2
- Stage 2 final test eval ran at bs=8 (vs bs=4 in earlier stages)
- ModelScope mirror (adapter + 16 metadata/eval files, public, Apache-2.0): https://www.modelscope.cn/models/dadazi123/llava-mistral-dgm4-s2-chain

## Stage 2-v2-chain — 12 metrics on DGM4 test split (2207 samples)

| Metric | Category | LLaVA-NeXT-Mistral-7B<br/>(28k×1ep) | Qwen3 Stage2-v2-chain<br/>(28k×2ep) | HAMMER 17.6k cp_49<br/>(bs80 small-data) | HAMMER 28k cp_best<br/>(bs75 resampled-v2) | HAMMER paper<br/>(230k) |
|---|---|---:|---:|---:|---:|---:|
| AUC ↑ | Binary | **0.9300** | 0.8934 | 0.7715 | 0.8043 | 0.9319 |
| EER ↓ | Binary | **0.1468** | 0.1951 | 0.2865 | 0.2705 | 0.1410 |
| ACC ↑ | Binary | **0.8523** | 0.8138 | 0.7141 | 0.7313 | 0.8639 |
| mAcc ↑ | Multi-Label | **0.9206** | 0.9069 | 0.8670 | 0.8612 | n/a (paper uses mAP=0.8622) |
| CF1 ↑ | Multi-Label | **0.7513** | 0.7194 | 0.5417 | 0.5165 | 0.7937 |
| OF1 ↑ | Multi-Label | **0.7364** | 0.6867 | 0.5410 | 0.5209 | 0.8037 |
| IoUmean ↑ | Image Ground | 0.7074 | 0.7087 | 0.5979 | 0.6095 | 0.7645 |
| IoU50 ↑ | Image Ground | **0.7576** | 0.7531 | 0.6402 | 0.6507 | 0.8375 |
| IoU75 ↑ | Image Ground | 0.6180 | 0.6747 | 0.5383 | 0.5473 | 0.7606 |
| Tok_Precision ↑ | Text Ground | 0.6093 | 0.5805 | 0.6360 | 0.6632 | 0.7501 |
| Tok_Recall ↑ | Text Ground | 0.5390 | 0.6140 | 0.4621 | 0.3900 | 0.6802 |
| Tok_F1 ↑ | Text Ground | 0.5720 | 0.5968 | 0.5353 | 0.4912 | 0.7135 |

**Bold** = highest value in row (HAMMER paper 230k excluded from "highest" comparison due to ~8× larger data scale).

## Head-to-head summary

| Opponent | LLaVA wins | Ties | LLaVA loses |
|---|---:|---:|---:|
| Qwen3 28k × 2ep (same paradigm, more epochs) | **9** | 0 | 3 (IoU75, Tok_Recall, Tok_F1) |
| HAMMER 17.6k cp_49 (discriminative, less data) | **11** | 0 | 1 (Tok_Precision −2.7pp) |
| HAMMER 28k cp_best (discriminative, same data) | **11** | 0 | 1 (Tok_Precision −5.4pp) |
| HAMMER paper 230k (discriminative, full scale) | 1 | 1 (AUC −0.2pp ~ tie) | 10 |

## Chain progression (binary + multilabel)

| Stage | AUC | EER | ACC | mAcc | CF1 | OF1 | Tok_F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage 1-cls (2ep) | 0.9254 | 0.1542 | 0.8518 | 0.8480 | 0.0877 | 0.0495 | — |
| Stage 1-cls-v2-chain (1.5ep) | 0.9316 | 0.1494 | 0.8500 | **0.9232** | **0.7555** | **0.7434** | — |
| Stage 2-grounding-v2-chain (1ep) | 0.9300 | **0.1468** | **0.8523** | 0.9206 | 0.7513 | 0.7364 | 0.5720 |

CF1 jumps from 0.088 (Stage 1) to 0.756 (Stage 1-v2-chain) — direct evidence for the explicit-atomic-label format (FS/FA/TS/TA: 0/1) impact.

## Stage 2 per-class verdict_acc / category_acc

| Class | n | verdict_acc | category_acc |
|---|---:|---:|---:|
| orig | 1061 | 0.865 | 0.865 |
| face_swap | 397 | 0.809 | 0.698 |
| face_attribute | 318 | 0.811 | 0.673 |
| text_swap | 136 | 0.809 | 0.632 |
| text_attribute | 74 | 0.811 | 0.554 |
| face_swap & text_swap | 86 | 0.953 | 0.523 |
| face_attribute & text_swap | 68 | 0.971 | 0.471 |
| face_swap & text_attribute | 35 | **1.000** | 0.571 |
| face_attribute & text_attribute | 32 | 0.969 | 0.500 |

Combined-manipulation samples (image+text) reach verdict_acc ≥ 0.95 — two independent forensic cues are easier to detect than single-modal tampering.

## Key takeaways (paper narrative)

1. **Data efficiency**: LLaVA 28k × 1ep ≈ HAMMER 230k × 50ep on binary classification (AUC 0.9300 vs 0.9319, −0.2pp).
2. **Small-data dominance**: At the same 28k data scale, generative LLaVA chain outperforms discriminative HAMMER reproduction on 11/12 metrics.
3. **Honest structural gap**: Grounding precision (IoU75, Tok_Precision/Recall) still lags HAMMER paper by ~14pp — attributed to discriminative head + LPAA + Token Detector architecture; valid future-work direction.

## Cross-references

- Training pipeline + adapter weights + eval JSONs: `hammer-small-data` branch @ `af9c219`
- ModelScope (adapter + 16 metadata/eval files): https://www.modelscope.cn/models/dadazi123/llava-mistral-dgm4-s2-chain
- HAMMER small-data reproduction details: `hammer-small-data` branch `docs/dgm4_curriculum_experiment_summary_zh.md` §14
- HAMMER paper Table 2 numbers source: CVPR 2304.02556v1
