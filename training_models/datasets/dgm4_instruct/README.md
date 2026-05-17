# DGM4-Instruct dataset

This folder contains LLaMA-Factory compatible ShareGPT multimodal files.

- `train.json`
- `val.json`
- `test.json`
- `dataset_info.json`

Each sample keeps metadata fields (`id`, `image`, `fake_cls`, etc.) and adds an `images` column:

```json
"images": ["DGM4/origin/..."]
```

LLaMA-Factory uses `conversations` and `images` according to `dataset_info.json`.
When moving this dataset to a server, place or symlink the `DGM4` image root under this dataset folder so these relative image paths can resolve.
