# DGM4 小数据指令微调与分阶段训练实验记录

本文档记录 Qwen3-VL-8B 在 DGM4 小规模数据上的训练策略、数据集改造、阶段评测结果和 rerank 尝试结论。指标统一采用当前项目中的 DGM4 评测脚本口径：二分类使用 AUC/EER/ACC，多标签分类使用 mAcc/CF1/OF1，图像定位使用 IoUmean/IoU50/IoU75，文本定位使用 Token Precision/Recall/F1。

## 1. 指标口径说明

原论文 Table 2 的多标签分类指标是四个原子 manipulation label：

- `FS`: face swap
- `FA`: face attribute
- `TS`: text swap
- `TA`: text attribute

早期我们曾用 9 类组合类别做整体分类，例如 `face_swap&text_swap`，这和论文的多标签分类口径不一致。因此后续统一改成原子标签指标。

另外，生成式模型没有论文 HAMMER 那样的多标签分类 head，因此不能严格计算基于概率分数的 mAP。当前脚本将原先的 mAP 替换为 `mAcc`，即四个原子标签的宏平均准确率。

## 2. 原论文与复现背景

论文 HAMMER 的 Table 2 结果大致为：

| 指标 | 论文 HAMMER |
|---|---:|
| AUC | 0.9319 |
| EER | 0.1410 |
| ACC | 0.8639 |
| mAP | 0.8622 |
| CF1 | 0.7937 |
| OF1 | 0.8037 |
| IoUmean | 0.7645 |
| IoU50 | 0.8375 |
| IoU75 | 0.7606 |
| Text Precision | 0.7501 |
| Text Recall | 0.6802 |
| Text F1 | 0.7135 |

官方 HAMMER 不是从随机初始化训练出来的。官方 `train.sh` 加载了 `ALBEF_4M.pth`，即大规模图文对齐预训练权重；配置中还包含 momentum queue、多任务 head、bbox L1/GIoU loss、token classification loss 等。复现时观察到：从头训练 HAMMER 效果很差，加载作者预训练权重后才能接近论文水平。

本项目当前路线是使用 Qwen3-VL-8B 作为多模态大模型底座，进行 LoRA 指令微调。Qwen3-VL 本身具备强视觉语言预训练能力，但输出是生成式文本，和 HAMMER 的分类/定位 head 存在结构差异。

## 3. auto 分支三组五行训练结果

这一组 `3ep / r2 / 5ep-resume` 来自 `autodl-training` / auto 分支的训练结果，不属于当前 `main` 分支新做的四个端到端实验。它们主要用于早期判断“继续训练”和“断点续训”是否有效，以及发现旧 token grounding 统计口径偏高的问题。

最初使用统一五行格式训练：

```text
Verdict: [REAL or FAKE]
Category: [category]
Fake Image Box: [box or []]
Fake Text Pos: [positions or []]
Evidence: [concise grounded explanation]
```

auto 分支三种训练结果含义：

- `3ep baseline`: 原始五行格式训练 3 epoch。
- `r2 adapter 3+2ep`: 在 3ep adapter 基础上更换学习率/优化器参数继续训练 2 epoch，属于二次微调。
- `5ep-resume 3+2ep`: 在 3ep 基础上完整继承 optimizer/scheduler 等状态继续训练 2 epoch，属于断点续训。

早期表格结果：

| 指标 | 3ep baseline | r2 adapter 3+2ep | 5ep resume 3+2ep |
|---|---:|---:|---:|
| AUC | 0.7980 | 0.7618 | 0.7859 |
| EER | 0.2730 | 0.3110 | 0.2949 |
| ACC | 0.7625 | 0.7839 | 0.7801 |
| mAP/旧口径 | 0.3258 | 0.3285 | 0.3311 |
| CF1 | 0.5170 | 0.5230 | 0.5249 |
| OF1 | 0.6510 | 0.6670 | 0.6677 |
| IoUmean | 0.6705 | 0.6966 | 0.6872 |
| IoU50 | 0.8162 | 0.8582 | 0.8457 |
| IoU75 | 0.5551 | 0.5782 | 0.5706 |
| Tok Precision | 0.8518 | 0.8591 | 0.8612 |
| Tok Recall | 0.8632 | 0.8646 | 0.8678 |
| Tok F1 | 0.8574 | 0.8618 | 0.8645 |

注意：早期 Token F1 偏高，主要原因是旧脚本对 `orig` 样本的空 token list 处理会放大结果。后续严格口径下，旧 3ep 的 Token F1 约为 0.55，而不是 0.86。

对 `lora-sft_epoch0_3` 使用当前严格脚本重新计算得到：

| 指标 | 旧 3ep 严格重算 |
|---|---:|
| AUC | 0.8171 |
| EER | 0.2665 |
| ACC | 0.7612 |
| mAcc | 0.8988 |
| OF1 | 0.6088 |
| CF1 | 0.6441 |
| IoUmean | 0.6882 |
| IoU50 | 0.7218 |
| IoU75 | 0.6615 |
| Tok Precision | 0.5379 |
| Tok Recall | 0.5658 |
| Tok F1 | 0.5515 |

## 4. main 分支端到端训练记录

当前 `main` 分支在进入三阶段训练前，先做过端到端五行范式训练。端到端训练的特点是：从输入图文对直接生成最终五行输出，不显式拆分“分类阶段”和“grounding 阶段”。

已在仓库中归档并可追溯的端到端结果：

| 实验 | 输出目录 | 数据范式 | 训练轮次 | 备注 |
|---|---|---|---:|---|
| E2E-3ep | `training_models/outputs/qwen3-vl-8b/dgm4-instruct/lora-sft_epoch0_3` | 原始五行输出 | 3 epoch | 当前仓库已保存 adapter、训练日志、eval 结果和重算 12 指标 |

`lora-sft_epoch0_3` 的训练日志摘要：

| 指标 | 数值 |
|---|---:|
| epoch | 3.0 |
| train_loss | 0.5627 |
| eval_loss | 0.5329 |
| train_runtime | 18572.20s |
| eval_runtime | 214.44s |

使用当前严格 DGM4 评测脚本重算得到：

| 指标 | E2E-3ep 严格重算 |
|---|---:|
| AUC | 0.8171 |
| EER | 0.2665 |
| ACC | 0.7612 |
| mAcc | 0.8988 |
| OF1 | 0.6088 |
| CF1 | 0.6441 |
| IoUmean | 0.6882 |
| IoU50 | 0.7218 |
| IoU75 | 0.6615 |
| Tok Precision | 0.5379 |
| Tok Recall | 0.5658 |
| Tok F1 | 0.5515 |

和后续 Stage2-v2 对比，端到端 3ep 的主要差距是：

- 二分类能力较弱：AUC `0.8171 -> 0.8934`，ACC `0.7612 -> 0.8138`。
- 多标签 F1 较弱：CF1 `0.6441 -> 0.7194`，OF1 `0.6088 -> 0.6867`。
- 图像定位略弱：IoUmean `0.6882 -> 0.7087`，IoU75 `0.6615 -> 0.6747`。
- 文本定位略弱：Tok F1 `0.5515 -> 0.5968`。

说明：你提到 main 分支曾训练过四个端到端实验；当前仓库中可直接追溯的完整 summary 主要是 `lora-sft_epoch0_3`。其余端到端实验如果后续补充对应的 `eval_result.json` / `eval_dgm4_12metrics_summary.json`，应继续填入本节，作为三阶段训练前的端到端 baseline。

## 5. 第一版三阶段 Curriculum

第一版分阶段思路：

1. `Stage 1`: 检测 + 原子多标签分类。
2. `Stage 2`: grounding 专项，输出分类字段 + image/text grounding。
3. `Stage 3`: 回到完整五行格式，补充 evidence。

对应训练轮次：

| 阶段 | 训练目标 | 训练轮次 |
|---|---|---:|
| Stage 1 | 二分类 + 原子多标签 + category | 2 epoch |
| Stage 2 | 分类 + image/text grounding | 1.5 epoch |
| Stage 3 | 完整五行 + evidence | 0.5 epoch |

### Stage 1 结果

checkpoint: `outputs/qwen3-vl-8b/dgm4-curriculum/stage1-cls/checkpoint-3922`

Stage 1 只评估二分类和原子多标签，不强行计算 grounding。

| 指标 | Stage 1 |
|---|---:|
| AUC | 0.8714 |
| EER | 0.2130 |
| ACC | 0.7775 |
| mAcc | 0.8406 |
| OF1 | 0.3761 |
| CF1 | 0.3506 |

Stage 1 分类能力已经提升，但多标签 F1 仍低，说明只输出 `Verdict/Category` 的信息不足以稳定学习四个原子标签。

### Stage 2 结果

checkpoint: `outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding/checkpoint-2942`

| 指标 | Stage 2 |
|---|---:|
| AUC | 0.8812 |
| EER | 0.2064 |
| ACC | 0.7916 |
| mAcc | 0.9036 |
| OF1 | 0.6562 |
| CF1 | 0.6991 |
| IoUmean | 0.6816 |
| IoU50 | 0.7227 |
| IoU75 | 0.6371 |
| Tok Precision | 0.5062 |
| Tok Recall | 0.6279 |
| Tok F1 | 0.5605 |

Stage 2 是第一版 curriculum 中最强阶段。它明显提升了分类和 grounding，但 token precision 偏低，说明模型倾向于多标 token。

### Stage 3 结果

checkpoint: `outputs/qwen3-vl-8b/dgm4-curriculum/stage3-final`

| 指标 | Stage 3 |
|---|---:|
| AUC | 0.8736 |
| EER | 0.2179 |
| ACC | 0.7852 |
| mAcc | 0.8889 |
| OF1 | 0.5945 |
| CF1 | 0.6363 |
| IoUmean | 0.6698 |
| IoU50 | 0.7091 |
| IoU75 | 0.6208 |
| Tok Precision | 0.4974 |
| Tok Recall | 0.6070 |
| Tok F1 | 0.5468 |

Stage 3 相比 Stage 2 出现回落。结论是：完整五行 + evidence 格式会增强最终表达能力，但如果训练过重，会冲掉 Stage 2 学到的分类和 grounding 能力。

## 6. Stage1-v2 / Stage2-v2 数据集重构

第一版 Stage1 只训 `Verdict` 和 `Category`，对论文四个原子标签的监督不够直接。因此 v2 数据重构为显式原子标签格式。

### Stage1-v2 输出格式

```text
Verdict: FAKE
FS: 0
FA: 1
TS: 1
TA: 0
Category: face_attribute&text_swap
```

Stage1-v2 只负责分类，不包含 grounding 字段。

### Stage2-v2 输出格式

Stage2-v2 在 Stage1-v2 的基础上增加 grounding，并拆成三类子任务：

1. image grounding only
2. text grounding only
3. image + text grounding

示例：

```text
Verdict: FAKE
FS: 0
FA: 1
TS: 1
TA: 0
Category: face_attribute&text_swap
Fake Image Box: [244, 55, 308, 144]
Fake Text Pos: [0, 1, 2, 3, 8, 9, 11, 12, 16]
```

### 重采样策略

Stage2-v2 的目标是提升 fake、text、combined 类和 token grounding，因此 train split 进行了离线重采样：

| 类别 | 采样权重 |
|---|---:|
| orig | 0.8x |
| face_swap | 1.0x |
| face_attribute | 1.2x |
| text_swap | 1.5x |
| text_attribute | 2.5x |
| face_swap&text_swap | 2.0x |
| face_attribute&text_swap | 2.5x |
| face_swap&text_attribute | 3.0x |
| face_attribute&text_attribute | 3.0x |

额外权重：

- 有 `Fake Text Pos` 的样本额外 `1.5x`
- 同时包含 image + text manipulation 的样本额外 `1.5x`

### v2 数据集规模

Stage1-v2:

| split | samples |
|---|---:|
| train | 20592 |
| val | 2206 |
| test | 2207 |

Stage2-v2:

| split | samples |
|---|---:|
| train | 28086 |
| val | 2206 |
| test | 2207 |

Stage2-v2 train 中 grounding 子任务分布：

| grounding task | samples |
|---|---:|
| full | 20220 |
| text only | 4469 |
| image only | 3397 |

## 7. Stage2-v2 继续训练结果

Stage2-v2 从旧 Stage2 checkpoint 继续训练 1 epoch：

- 起点：`outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding/checkpoint-2942`
- 输出：`outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding-v2/checkpoint-3121`
- train samples: 28086
- optimization steps: 3121
- per-device batch size: 3
- gradient accumulation: 3
- effective batch size: 9

Stage2-v2 结果：

| 指标 | 旧 Stage2 | Stage2-v2 | 变化 |
|---|---:|---:|---:|
| AUC | 0.8812 | 0.8934 | +0.0122 |
| EER | 0.2064 | 0.1951 | -0.0113 |
| ACC | 0.7916 | 0.8138 | +0.0222 |
| mAcc | 0.9036 | 0.9069 | +0.0033 |
| OF1 | 0.6562 | 0.6867 | +0.0306 |
| CF1 | 0.6991 | 0.7194 | +0.0203 |
| IoUmean | 0.6816 | 0.7087 | +0.0271 |
| IoU50 | 0.7227 | 0.7531 | +0.0304 |
| IoU75 | 0.6371 | 0.6747 | +0.0376 |
| Tok Precision | 0.5062 | 0.5805 | +0.0743 |
| Tok Recall | 0.6279 | 0.6140 | -0.0139 |
| Tok F1 | 0.5605 | 0.5968 | +0.0363 |

和 Stage3-final 对比：

| 指标 | Stage3-final | Stage2-v2 |
|---|---:|---:|
| AUC | 0.8736 | 0.8934 |
| ACC | 0.7852 | 0.8138 |
| mAcc | 0.8889 | 0.9069 |
| OF1 | 0.5945 | 0.6867 |
| CF1 | 0.6363 | 0.7194 |
| IoUmean | 0.6698 | 0.7087 |
| IoU75 | 0.6208 | 0.6747 |
| Tok F1 | 0.5468 | 0.5968 |

结论：Stage2-v2 是当前最强 checkpoint。显式原子标签、重采样、grounding 子任务拆分都与指标提升一致，尤其改善了 token precision 和 image grounding。

但需要注意：Stage2-v2 相比旧 Stage2 也多训练了 1 epoch。因此严格消融还需要补一个 control：从旧 Stage2 checkpoint 继续使用旧 Stage2 数据/格式再训 1 epoch，用来区分“v2 数据策略收益”和“继续训练收益”。

## 8. Rerank 优化尝试与终止结论

为了把生成式 grounding 改得更接近判别式 head，设计了推理侧 rerank：

- Token rerank: 对候选 token 单独判断 `0/1`，用 `P(1)` 决定是否保留。
- Box rerank: 对原预测框附近的候选框单独判断 `0/1`，选 `P(1)` 最高的框。

第一版 rerank 脚本发现所有 score 都是 `0.5`，原因是 tokenizer 对 `" 0"` 和 `" 1"` 的第一个 token 可能都是空格 token，导致实际比较的是同一个 token：

```text
P(space) vs P(space) -> 0.5
```

已修复为完整序列 log probability：

```text
log p("1" | prompt)
log p("0" | prompt)
score = softmax([logp_1, logp_0])
```

修复后继续做了两类小样本验证：

1. numeric scoring: `1` vs `0`
2. word scoring: token 使用 `manipulated` vs `unchanged`，box 使用 `correct` vs `incorrect`

### 8.1 numeric rerank 验证

numeric rerank 修复后不再全是 `0.5`，概率可以正常变化。但在 15 条样本上，和真实 label 对比后发现：直接使用原始 Stage2-v2 grounding 仍然最好。

Token 策略扫描：

| 策略 | Precision | Recall | F1 |
|---|---:|---:|---:|
| 原始预测 | 0.4118 | 0.8235 | 0.5490 |
| 全候选 th=0.03 | 0.3889 | 0.8235 | 0.5283 |
| 只保留原始预测 top8 | 0.4400 | 0.6471 | 0.5238 |
| 原始预测过滤 th=0.03 | 0.4138 | 0.7059 | 0.5217 |
| th=0.5 | 0.5000 | 0.1176 | 0.1905 |

Box 策略扫描：

| 策略 | IoUmean | IoU50 | IoU75 |
|---|---:|---:|---:|
| 原始 box | 0.6682 | 0.6667 | 0.6667 |
| rerank 最高分 box | 0.6131 | 0.6667 | 0.4667 |

结论：numeric rerank 的概率输出已正常，但概率和真实 grounding 正确性不够对齐。高阈值会删掉大量真 token，低阈值又无法超过原始预测；box 最高分选择反而降低 IoU。

### 8.2 word rerank 验证

为了避免 `0/1` 形式过于抽象，又尝试了语义标签判别：

```text
token: manipulated vs unchanged
box: correct vs incorrect
```

20 条样本上的 token 策略扫描：

| 策略 | Precision | Recall | F1 |
|---|---:|---:|---:|
| 原始预测 | 0.5532 | 0.8667 | 0.6753 |
| word 低阈值过滤 | 0.5532 | 0.8667 | 0.6753 |
| 全候选极低阈值 | 0.4918 | 1.0000 | 0.6593 |
| 原始 top8 | 0.5758 | 0.6333 | 0.6032 |

Box 策略扫描：

| 策略 | IoUmean | IoU50 | IoU75 |
|---|---:|---:|---:|
| 原始 box | 0.6596 | 0.6500 | 0.6000 |
| word rerank box | 0.6148 | 0.6500 | 0.4500 |

结论：word rerank 也没有超过原始 Stage2-v2 grounding。token 分数整体过低，box 分数虽有区分度，但“选最高分框”不等于选最高 IoU 框。

### 8.3 最终决定

推理侧 rerank 路线暂时终止，不再作为后续主要优化方向。当前最稳策略是保留 Stage2-v2 原始 grounding 输出。

原因：

- 生成式模型可以给候选项打出不同概率，但这些概率没有可靠校准到 token/box 是否正确。
- token rerank 会在 precision/recall 之间做无效交换，整体 F1 不如原始预测。
- box rerank 的最高分候选框不如原始生成框。
- 继续全量 rerank 会消耗大量显卡时间，但没有指标收益。

后续如果要提升 grounding，不应继续做推理侧 rerank，而应回到训练侧改造：

- Stage2-v3 增加更直接的 token mask 监督；
- 构造更干净的 text grounding 样本；
- 训练时显式惩罚多标 token；
- 或者引入轻量判别式 token/box head，而不是用 prompt 后处理。

## 9. 当前结论

1. 旧五行全量训练可以学到格式，但分类和 grounding 不够稳。
2. 第一版三阶段训练中，Stage2 明显强于 Stage3，说明 grounding 专项训练有效。
3. Stage3 完整五行 + evidence 会造成能力回落，后续 Stage3 只能短训、低学习率、混合 Stage1/2 样本。
4. Stage2-v2 是目前最强方案，说明显式原子标签、重采样和 grounding 子任务拆分是有效方向。
5. 与论文 HAMMER 差距主要集中在 OF1、Text F1、IoU75。结构原因包括：HAMMER 使用判别式 head 和 ALBEF_4M 图文对齐预训练，而当前 Qwen 路线是生成式输出再解析。
6. 后续要想进一步提升后几个指标，应优先尝试：
   - Stage2-control +1ep 消融；
   - Stage3-v2 轻量 evidence 对齐；
   - Stage2-v3 判别式 token mask / 更强 text grounding 数据；
   - 如资源允许，探索轻量判别式 token/box head。

## 10. 关键文件索引

数据构建：

- `training_models/build_curriculum_v2_datasets.py`
- `training_models/datasets/dgm4_stage1_cls_v2/`
- `training_models/datasets/dgm4_stage2_grounding_v2/`

训练配置：

- `training_models/configs/qwen3vl_8b_lora_curriculum_stage1_cls_v2_continue_autodl.yaml`
- `training_models/configs/qwen3vl_8b_lora_curriculum_stage2_grounding_v2_continue_autodl.yaml`
- `training_models/run_curriculum_v2_stage12_train_eval.sh`

评测：

- `training_models/eval_dgm4_qwen3vl.py`
- `training_models/eval_curriculum_stage_outputs.py`

Stage2-v2 结果：

- `training_models/outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding-v2/eval_stage2_v2_binary_multilabel_grounding_summary.json`
- `training_models/outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding-v2/eval_stage2_v2_binary_multilabel_grounding.json`
- `training_models/outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding-v2/checkpoint-3121/trainer_state.json`

Rerank 相关代码已从仓库移除；本路线保留为负结果记录，不再作为后续默认优化方向。

## 11. Stage1-v2-chain 到 Stage2-v2-chain 结果

前面的 Stage2-v2 是从第一版 `stage2-grounding/checkpoint-2942` 继续训练得到的，没有经过 Stage1-v2 显式原子标签阶段。为了验证“先学原子标签，再学 grounding”的链式训练是否有效，后续补做了 clean chain 实验：

```text
stage1-cls/checkpoint-3922
  -> stage1-cls-v2-chain 继续训练 1.5 epoch
  -> stage2-grounding-v2-chain 继续训练到 6500 step
  -> 使用最后 adapter 在 test split 上推理和计算 12 个指标
```

其中 `stage2-grounding-v2-chain` 原计划训练约 2 epoch，中途因为磁盘空间不足在接近 6000 step 时中断。清理 checkpoint 后，从 `checkpoint-5500` 恢复训练，并设置 `max_steps=6500`，最终使用 `stage2-grounding-v2-chain` 根目录 adapter 进行 test 评测。

### 11.1 训练配置摘要

| 项目 | 内容 |
|---|---|
| Stage1 起点 | `stage1-cls/checkpoint-3922` |
| Stage1-v2-chain 输出 | `stage1-cls-v2-chain/checkpoint-3432` |
| Stage1-v2-chain 训练量 | 1.5 epoch |
| Stage2-v2-chain 起点 | `stage1-cls-v2-chain` 最终 adapter |
| Stage2-v2-chain 数据 | `dgm4_stage2_grounding_v2`，包含重采样和 grounding 子任务拆分 |
| Stage2-v2-chain 训练量 | 继续到 `global_step=6500`，约 2.08 epoch |
| 恢复方式 | 从 `stage2-grounding-v2-chain/checkpoint-5500` resume |
| checkpoint 策略 | chain 实验 1000 step 保存一次，最终只保留最后 checkpoint |

训练日志摘要：

| 指标 | 数值 |
|---|---:|
| global_step | 6500 |
| epoch | 2.0827 |
| train_loss | 0.0124 |
| eval_loss | 0.1319 |
| train_runtime | 3028.95s |

### 11.2 Stage2-v2-chain test 结果

结果文件：

- `training_models/outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding-v2-chain/eval_stage2_v2_chain_resume6500_binary_multilabel_grounding_summary.json`
- `training_models/outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding-v2-chain/eval_stage2_v2_chain_resume6500_binary_multilabel_grounding.json`

| 指标 | Stage2-v1 | Stage2-v2 | Stage2-v2-chain resume6500 |
|---|---:|---:|---:|
| AUC | 0.8812 | 0.8934 | 0.8868 |
| EER | 0.2064 | 0.1951 | 0.1957 |
| ACC | 0.7916 | 0.8138 | 0.8097 |
| mAcc | 0.9036 | 0.9069 | 0.9065 |
| OF1 | 0.6562 | 0.6867 | 0.6874 |
| CF1 | 0.6991 | 0.7194 | 0.7179 |
| IoUmean | 0.6816 | 0.7087 | 0.7039 |
| IoU50 | 0.7227 | 0.7531 | 0.7512 |
| IoU75 | 0.6371 | 0.6747 | 0.6724 |
| Tok Precision | 0.5062 | 0.5805 | 0.6659 |
| Tok Recall | 0.6279 | 0.6140 | 0.6321 |
| Tok F1 | 0.5605 | 0.5968 | 0.6486 |

### 11.3 结果解释

Stage2-v2-chain 对文本 grounding 的提升最明显。相比 Stage2-v2，`Tok Precision` 从 `0.5805` 提升到 `0.6659`，`Tok Recall` 从 `0.6140` 提升到 `0.6321`，`Tok F1` 从 `0.5968` 提升到 `0.6486`。这说明 `Stage1-v2 显式原子标签 + Stage2-v2 重采样和 grounding 子任务拆分 + 更长 Stage2 训练` 对 token grounding 是有效的，尤其明显改善了 token precision。

但 image grounding 和二分类略有回落。相比 Stage2-v2，AUC 从 `0.8934` 降到 `0.8868`，ACC 从 `0.8138` 降到 `0.8097`，IoUmean 从 `0.7087` 降到 `0.7039`，IoU75 从 `0.6747` 降到 `0.6724`。这说明继续拉长 Stage2 训练会更偏向文本定位能力，但可能轻微牺牲整体检测和图像定位。

相比 Stage2-v1，Stage2-v2-chain 在 ACC、OF1、CF1、IoU 和 Tok F1 上仍然全面更强。因此 v2 数据范式和重采样策略整体有效；只是 Stage2-v2 与 Stage2-v2-chain 的取舍不同：前者更均衡，后者更偏文本 grounding。

## 12. 最新结果总表与 HAMMER 论文对比

下表把当前关键阶段结果放在同一张表中，最后一行加入 HAMMER 原论文 Table 2 的结果，便于观察差距。

注意：我们当前生成式评测使用 `mAcc` 替代原论文的 `mAP`；HAMMER 行中的对应列仍是论文 `mAP`。因此该列只能作为大致参考，不能视为完全相同指标。

| 模型/阶段 | AUC | EER | ACC | mAcc/mAP | CF1 | OF1 | IoUmean | IoU50 | IoU75 | Tok Precision | Tok Recall | Tok F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage2-v1 | 0.8812 | 0.2064 | 0.7916 | 0.9036 | 0.6991 | 0.6562 | 0.6816 | 0.7227 | 0.6371 | 0.5062 | 0.6279 | 0.5605 |
| Stage2-v2 | 0.8934 | 0.1951 | 0.8138 | 0.9069 | 0.7194 | 0.6867 | 0.7087 | 0.7531 | 0.6747 | 0.5805 | 0.6140 | 0.5968 |
| Stage2-v2-chain resume6500 | 0.8868 | 0.1957 | 0.8097 | 0.9065 | 0.7179 | 0.6874 | 0.7039 | 0.7512 | 0.6724 | 0.6659 | 0.6321 | 0.6486 |
| Stage2-v2-chain + field weighted loss 0.3ep | 0.8843 | 0.2023 | 0.8034 | 0.9060 | 0.7116 | 0.6813 | 0.7044 | 0.7494 | 0.6733 | 0.6738 | 0.6062 | 0.6382 |
| HAMMER small-data bs80 test best | 0.7926 | 0.2888 | 0.7118 | 0.5752 | 0.5302 | 0.5332 | 0.0669 | 0.0403 | 0.0036 | 0.6464 | 0.3843 | 0.4820 |
| HAMMER resampled-v2 bs75 test best | 0.8043 | 0.2705 | 0.7313 | 0.5844 | 0.5165 | 0.5209 | 0.6095 | 0.6507 | 0.5473 | 0.6632 | 0.3900 | 0.4912 |
| HAMMER resampled-v2 bs75 test epoch49 | 0.7976 | 0.2731 | 0.7218 | 0.5785 | 0.5285 | 0.5365 | 0.5975 | 0.6434 | 0.5356 | 0.6718 | 0.4119 | 0.5107 |
| HAMMER 原论文 Table 2 | 0.9319 | 0.1410 | 0.8639 | 0.8622 | 0.7937 | 0.8037 | 0.7645 | 0.8375 | 0.7606 | 0.7501 | 0.6802 | 0.7135 |

### 12.1 和 HAMMER 的差距

以当前文本 grounding 最强的 `Stage2-v2-chain resume6500` 对比 HAMMER：

| 指标 | Stage2-v2-chain | HAMMER | 差距 |
|---|---:|---:|---:|
| AUC | 0.8868 | 0.9319 | -0.0451 |
| ACC | 0.8097 | 0.8639 | -0.0542 |
| CF1 | 0.7179 | 0.7937 | -0.0758 |
| OF1 | 0.6874 | 0.8037 | -0.1163 |
| IoUmean | 0.7039 | 0.7645 | -0.0606 |
| IoU75 | 0.6724 | 0.7606 | -0.0882 |
| Tok Precision | 0.6659 | 0.7501 | -0.0842 |
| Tok Recall | 0.6321 | 0.6802 | -0.0481 |
| Tok F1 | 0.6486 | 0.7135 | -0.0649 |

当前 Qwen3-VL 生成式路线已经接近 HAMMER 的 token recall，但 token precision、OF1 和高阈值图像定位仍有明显差距。这个差距符合模型结构差异：HAMMER 是判别式多任务模型，直接用分类 head、bbox head 和 token classification head 优化；当前方法是生成式五行输出，再解析为结构化结果。

### 12.2 当前最合理的 checkpoint 选择

如果目标是写论文主结果，建议同时报告两个 checkpoint：

- `Stage2-v2`：作为更均衡的主 baseline，AUC、ACC、CF1、IoUmean/IoU75 更高。
- `Stage2-v2-chain resume6500`：作为 grounding 强化版本，Token Precision/Recall/F1 最高，适合证明 Stage1-v2 + Stage2-v2 链式训练对文本定位有效。

如果后续要继续优化，不建议单纯把 Stage2-v2-chain 再长训 2 epoch。更稳的方向是从 `stage2-grounding-v2-chain/checkpoint-6500` 继续做短程训练，例如 `0.2-0.3 epoch`，用更低学习率和字段加权 loss，目标是保住 AUC/ACC/IoU，同时继续提高 token F1。

## 13. 字段加权 loss 短训实验

为了尝试让模型更重视 DGM4 输出中的关键字段，尤其是原子标签、box 和 text position，在 LLaMA-Factory 的 SFT loss 上增加了字段级 token 权重。原始 SFT 是普通 causal LM cross entropy：

```text
L = - sum_t log p(y_t | x, y_<t)
```

字段加权版本为：

```text
L = - sum_t w_t log p(y_t | x, y_<t) / sum_t w_t
```

其中不同字段使用不同权重，例如格式字段较低，原子标签、category、box、text position 较高。该实验从 `stage2-grounding-v2-chain/checkpoint-6500` 继续短训 0.3 epoch，然后在 test split 上推理和计算 12 个指标。

结果文件：

- `training_models/outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding-v2-chain-wloss-0p3ep/eval_stage2_v2_chain_wloss_0p3ep_binary_multilabel_grounding_summary.json`
- `training_models/outputs/qwen3-vl-8b/dgm4-curriculum/stage2-grounding-v2-chain-wloss-0p3ep/eval_stage2_v2_chain_wloss_0p3ep_binary_multilabel_grounding.json`

### 13.1 与 Stage2-v2-chain 对比

| 指标 | Stage2-v2-chain resume6500 | Field weighted loss 0.3ep | 变化 |
|---|---:|---:|---:|
| AUC | 0.8868 | 0.8843 | -0.0025 |
| EER | 0.1957 | 0.2023 | +0.0066 |
| ACC | 0.8097 | 0.8034 | -0.0063 |
| mAcc | 0.9065 | 0.9060 | -0.0005 |
| CF1 | 0.7179 | 0.7116 | -0.0063 |
| OF1 | 0.6874 | 0.6813 | -0.0061 |
| IoUmean | 0.7039 | 0.7044 | +0.0005 |
| IoU50 | 0.7512 | 0.7494 | -0.0018 |
| IoU75 | 0.6724 | 0.6733 | +0.0009 |
| Tok Precision | 0.6659 | 0.6738 | +0.0079 |
| Tok Recall | 0.6321 | 0.6062 | -0.0259 |
| Tok F1 | 0.6486 | 0.6382 | -0.0104 |

### 13.2 实验结论

字段加权 loss 确实让 token prediction 更保守，Token Precision 从 `0.6659` 提升到 `0.6738`，但 Token Recall 明显下降，最终 Tok F1 从 `0.6486` 降到 `0.6382`。同时 AUC、ACC、CF1、OF1 也有轻微回落。

因此，当前字段加权 loss 不是更好的主结果。它可以作为一次负向消融记录：只靠字段权重会让模型减少多标 token，但会牺牲召回和整体分类稳定性。后续如果继续做 loss 设计，应考虑更细粒度的 token false-positive 惩罚或 focal loss，而不是简单按字段整体加权。

## 14. HAMMER 小数据复现实验记录

本节记录在服务器 `/root/autodl-tmp/DGM4_Project/training_models/MultiModal-DeepFake` 下完成的 HAMMER 判别式模型复现实验，用作生成式路线的同数据规模对照。所有数值均由当前项目脚本输出的百分制结果除以 100 得到。

### 14.1 已完成实验

| 实验 | 训练数据 | batch size | 起点权重 | 训练轮次 | 结果目录 |
|---|---:|---:|---|---:|---|
| HAMMER small-data bs80 | 17,648 | 80 | `ALBEF_4M.pth` | 50 | `results/loghammer_small_20260528_112017_50ep_bs80_autoshutdown` |
| HAMMER resampled-v2 bs75 | 28,086 | 75 | `ALBEF_4M.pth` + `checkpoint_best` resume | 50 | `results/loghammer_resampled_v2_resume_20260528_1646_bs75_lrhalf` |

`HAMMER resampled-v2 bs75` 使用 `dgm4_stage2_grounding_v2/train.json` 重采样数据转换得到的 HAMMER metadata。训练中途 batch size 从 80 降到 75 以避免显存溢出；磁盘满导致一次 `checkpoint_10.pth` 写入失败，清理中间权重后从当时最新 `checkpoint_best.pth` 继续训练。最终 checkpoint 策略为每 10 epoch 保存一次周期权重，并保留 `checkpoint_best.pth`。

### 14.2 验证集各指标最佳 epoch

HAMMER 当前训练脚本只按 `val_AUC_cls` 保存 `checkpoint_best.pth`，因此分类、图像定位和文本定位的最佳 epoch 不完全一致。下表记录验证集上各指标出现的最佳轮次。

| 指标 | small-data bs80 最佳 epoch | small-data bs80 | resampled-v2 bs75 最佳 epoch | resampled-v2 bs75 |
|---|---:|---:|---:|---:|
| AUC | 6 | 0.7966 | 32 | 0.8145 |
| ACC | 28 | 0.7248 | 32 | 0.7353 |
| EER | 28 | 0.2744 | 34 | 0.2644 |
| mAP | 29 | 0.5822 | 12 | 0.5956 |
| OF1 | 49 | 0.5559 | 42 | 0.5776 |
| CF1 | 49 | 0.5517 | 42 | 0.5722 |
| mACC | 16 | 0.8736 | 31 | 0.8764 |
| IoUmean | 42 | 0.6249 | 24 | 0.6304 |
| IoU50 | 42 | 0.6618 | 24 | 0.6691 |
| IoU75 | 42 | 0.5603 | 24 | 0.5612 |
| IoU95 | 27 | 0.5018 | 24 | 0.4828 |
| Tok F1 | 36 | 0.5833 | 27 | 0.6037 |

从验证集看，重采样版本在 AUC、ACC、EER、mAP、OF1/CF1 和 Tok F1 上均优于原始 small-data 版本，说明长尾类别重采样对小数据 HAMMER 是有效的。图像定位的 IoUmean/IoU50/IoU75 也略有提升，但幅度不大。

### 14.3 Test split 结果

| 模型/checkpoint | AUC | EER | ACC | mAP | CF1 | OF1 | mACC | IoUmean | IoU50 | IoU75 | IoU95 | Tok P | Tok R | Tok F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small-data bs80 `checkpoint_best` | 0.7926 | 0.2888 | 0.7118 | 0.5752 | 0.5302 | 0.5332 | 0.8711 | 0.0669 | 0.0403 | 0.0036 | 0.0000 | 0.6464 | 0.3843 | 0.4820 |
| small-data bs80 `checkpoint_49` | 0.7715 | 0.2865 | 0.7141 | 0.5741 | 0.5417 | 0.5410 | 0.8670 | 0.5979 | 0.6402 | 0.5383 | 0.4345 | 0.6360 | 0.4621 | 0.5353 |
| resampled-v2 bs75 `checkpoint_best` / epoch32 | 0.8043 | 0.2705 | 0.7313 | 0.5844 | 0.5165 | 0.5209 | 0.8612 | 0.6095 | 0.6507 | 0.5473 | 0.4295 | 0.6632 | 0.3900 | 0.4912 |
| resampled-v2 bs75 `checkpoint_49` | 0.7976 | 0.2731 | 0.7218 | 0.5785 | 0.5285 | 0.5365 | 0.8653 | 0.5975 | 0.6434 | 0.5356 | 0.4087 | 0.6718 | 0.4119 | 0.5107 |

`checkpoint_best` 是按验证集 AUC 保存的，因此 test 上分类指标更好；`checkpoint_49` 的 OF1、CF1、四个原子标签 F1 和 Tok F1 更好，但 AUC/ACC/IoU 略低。后续如果继续跑 HAMMER 对照，建议同时保存 `best_auc`、`best_iou`、`best_tokf1` 和 `best_cf1`，避免多任务模型被单一 AUC 选择策略限制。

### 14.4 与生成式 Stage2-v2 的关系

当前最强生成式结果 `Stage2-v2-chain resume6500` 在同一 test 口径下达到 AUC `0.8868`、ACC `0.8097`、CF1 `0.7179`、OF1 `0.6874`、IoUmean `0.7039`、Tok F1 `0.6486`。相比本节 small-data HAMMER 和 resampled-v2 HAMMER，对小规模数据更稳定，尤其是多标签分类和文本定位明显更强。

因此可以把 HAMMER 小数据复现作为判别式小数据 baseline：在约 2-3 万训练样本规模下，即使加载 `ALBEF_4M.pth`，HAMMER 仍明显低于 23w 规模论文结果；而当前生成式 Stage2-v2 系列在同样小数据规模下表现更接近论文 HAMMER。这一对比可以支撑“生成式 MLLM 路线在小规模、长尾、多粒度 DGM4 数据上具备更好的数据效率”的论点。
