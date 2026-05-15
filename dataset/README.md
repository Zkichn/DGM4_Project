# LLaMA-Factory SFT Split Summary

- Source: `C:\Users\Zkichn\Desktop\learn_AI\DGM4_IDEA\versions\v3_2026-05-15\data\qwen_sft_data_v3_grounded_llamafactory.json`
- Output dir: `C:\Users\Zkichn\Desktop\learn_AI\DGM4_IDEA\versions\v3_2026-05-15\splits_llamafactory_8_1_1`
- Seed: `42`
- Ratio: train/val/test = 8/1/1

| split | count | file |
|---|---:|---|
| train | 17648 | `C:\Users\Zkichn\Desktop\learn_AI\DGM4_IDEA\versions\v3_2026-05-15\splits_llamafactory_8_1_1\train.json` |
| val | 2206 | `C:\Users\Zkichn\Desktop\learn_AI\DGM4_IDEA\versions\v3_2026-05-15\splits_llamafactory_8_1_1\val.json` |
| test | 2207 | `C:\Users\Zkichn\Desktop\learn_AI\DGM4_IDEA\versions\v3_2026-05-15\splits_llamafactory_8_1_1\test.json` |

## fake_cls Distribution

### train

| fake_cls | count |
|---|---:|
| `face_attribute` | 2392 |
| `face_attribute&text_attribute` | 260 |
| `face_attribute&text_swap` | 619 |
| `face_swap` | 2876 |
| `face_swap&text_attribute` | 339 |
| `face_swap&text_swap` | 696 |
| `orig` | 8747 |
| `text_attribute` | 501 |
| `text_swap` | 1218 |

### val

| fake_cls | count |
|---|---:|
| `face_attribute` | 299 |
| `face_attribute&text_attribute` | 24 |
| `face_attribute&text_swap` | 89 |
| `face_swap` | 381 |
| `face_swap&text_attribute` | 33 |
| `face_swap&text_swap` | 97 |
| `orig` | 1075 |
| `text_attribute` | 54 |
| `text_swap` | 154 |

### test

| fake_cls | count |
|---|---:|
| `face_attribute` | 318 |
| `face_attribute&text_attribute` | 32 |
| `face_attribute&text_swap` | 68 |
| `face_swap` | 397 |
| `face_swap&text_attribute` | 35 |
| `face_swap&text_swap` | 86 |
| `orig` | 1061 |
| `text_attribute` | 74 |
| `text_swap` | 136 |