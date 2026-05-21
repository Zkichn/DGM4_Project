# DGM4 小数据指令微调与分阶段训练实验记录

本文档记录 Qwen3-VL-8B 在 DGM4 小规模数据上的训练策略、数据集改造、阶段评测结果和后续 rerank 优化方向。指标统一采用当前项目中的 DGM4 评测脚本口径：二分类使用 AUC/EER/ACC，多标签分类使用 mAcc/CF1/OF1，图像定位使用 IoUmean/IoU50/IoU75，文本定位使用 Token Precision/Recall/F1。

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

## 8. Rerank 优化尝试

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

修复后的脚本：

- `training_models/rerank_grounding/rerank_stage2_v2.py`
- `training_models/rerank_grounding/run_stage2_v2_rerank.sh`

建议下次先跑 5 条 smoke：

```bash
cd /root/autodl-tmp/DGM4_Project/training_models
git pull
bash rerank_grounding/run_stage2_v2_rerank.sh smoke both 8 0.5 1 24
```

重点检查输出：

```json
"all_equal_0_5": false
```

确认概率正常后再跑全量：

```bash
bash rerank_grounding/run_stage2_v2_rerank.sh full both 12 0.5 1 24
```

## 9. 当前结论

1. 旧五行全量训练可以学到格式，但分类和 grounding 不够稳。
2. 第一版三阶段训练中，Stage2 明显强于 Stage3，说明 grounding 专项训练有效。
3. Stage3 完整五行 + evidence 会造成能力回落，后续 Stage3 只能短训、低学习率、混合 Stage1/2 样本。
4. Stage2-v2 是目前最强方案，说明显式原子标签、重采样和 grounding 子任务拆分是有效方向。
5. 与论文 HAMMER 差距主要集中在 OF1、Text F1、IoU75。结构原因包括：HAMMER 使用判别式 head 和 ALBEF_4M 图文对齐预训练，而当前 Qwen 路线是生成式输出再解析。
6. 后续要想进一步提升后几个指标，应优先尝试：
   - Stage2-control +1ep 消融；
   - 修复后的 token/box rerank；
   - Stage3-v2 轻量 evidence 对齐；
   - 如 rerank 有效，再构建 Stage2-v3 判别式 token 数据。

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

Rerank：

- `training_models/rerank_grounding/rerank_stage2_v2.py`
- `training_models/rerank_grounding/run_stage2_v2_rerank.sh`
- `training_models/rerank_grounding/monitor_shutdown_after_rerank.sh`
