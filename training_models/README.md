# Training models workspace

## Folder layout

- `src/LLaMA-Factory/`: LLaMA-Factory source code.
- `datasets/dgm4_instruct/`: DGM4-Instruct train/val/test files for multimodal SFT.
- `configs/`: training YAML configs.
- `base_models/`: optional local base model cache/checkpoints.
- `outputs/`: LoRA/checkpoint outputs.
- `logs/`: training logs.

## Recommended first steps

Open a terminal in:

```powershell
cd training_models/src/LLaMA-Factory
```

Create/activate an environment, then install LLaMA-Factory:

```powershell
pip install -e ".[torch,metrics]"
```

If you want to use ModelScope in China, install it too:

```powershell
pip install modelscope
```

## Train Qwen3-VL-8B with LoRA SFT

Config:

```text
../../configs/qwen3vl_8b_lora_sft_dgm4_instruct.yaml
```

Run:

```powershell
llamafactory-cli train ../../configs/qwen3vl_8b_lora_sft_dgm4_instruct.yaml
```

## Local smoke test on RTX 4060 8GB

Use the debug config to verify the full training path before moving to a server:

```powershell
cd training_models/src/LLaMA-Factory
llamafactory-cli train ../../configs/qwen3vl_8b_lora_sft_dgm4_instruct_debug_local.yaml
```

This config uses 4-bit QLoRA, 16 samples, lower image resolution, and only 5 training steps.

## Image root and system prompt

The dataset stores relative image paths such as:

```json
"images": ["DGM4/origin/..."]
```

The YAML uses `media_dir` to prepend the image root. On this Windows machine it is:

```yaml
media_dir: D:/
```

If the server stores images at `/data/DGM4/...`, change it to:

```yaml
media_dir: /data
```

The shared system prompt is injected by `default_system` in the YAML, so it does not need to be repeated inside every JSON sample.

## Notes

- The training config uses `template: qwen3_vl_nothink`, suitable for structured non-CoT outputs.
- LLaMA-Factory multimodal data requires an `images` list column. The prepared files in `datasets/dgm4_instruct/` already include it.
- Your original metadata fields (`id`, `image`, `fake_cls`, `fake_image_box`, `fake_text_pos`) are preserved for evaluation/debugging.
- Paths in the YAML are relative to `src/LLaMA-Factory`. Keep this workspace layout when moving it to a server.
- The `images` values are relative paths like `DGM4/origin/...`. On the server, place or symlink the `DGM4` image root under the dataset directory unless you later decide to use a different media-root strategy.
