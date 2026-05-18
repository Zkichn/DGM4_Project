---
library_name: peft
license: other
base_model: /root/autodl-tmp/hf-cache/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
tags:
- base_model:adapter:/root/autodl-tmp/hf-cache/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: lora-sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# lora-sft

This model is a fine-tuned version of [/root/autodl-tmp/hf-cache/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b](https://huggingface.co//root/autodl-tmp/hf-cache/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b) on the dgm4_instruct_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.5329

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.1
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.6762        | 0.2267 | 500  | 0.7081          |
| 0.6871        | 0.4533 | 1000 | 0.6352          |
| 0.6442        | 0.6800 | 1500 | 0.6037          |
| 0.5576        | 0.9066 | 2000 | 0.5827          |
| 0.5627        | 1.1333 | 2500 | 0.5713          |
| 0.5393        | 1.3599 | 3000 | 0.5573          |
| 0.5379        | 1.5866 | 3500 | 0.5474          |
| 0.5015        | 1.8132 | 4000 | 0.5373          |
| 0.4338        | 2.0399 | 4500 | 0.5397          |
| 0.4368        | 2.2665 | 5000 | 0.5389          |
| 0.4222        | 2.4932 | 5500 | 0.5361          |
| 0.4249        | 2.7199 | 6000 | 0.5344          |
| 0.4156        | 2.9465 | 6500 | 0.5328          |
| 0.4236        | 3.0    | 6618 | 0.5329          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.6.0
- Pytorch 2.5.1+cu124
- Datasets 4.0.0
- Tokenizers 0.22.2