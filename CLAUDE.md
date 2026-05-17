# CLAUDE.md — Repository Boundaries

仅记录硬约束（rules / boundaries）。**易变状态、训练结果、余额、待办等放
`docs/DEVELOPMENT_LOG.md`**。

新对话上手：先读 `docs/DEVELOPMENT_LOG.md` 最新一节获取当前状态，再读本文件
获取边界。

---

## 1. 训练断点续训约束（Training Resume — MUST HOLD）

**任何对训练参数 / 脚本 / 配置的修改，都必须保证训练在中断后可以从最近的
checkpoint 恢复，继续完成剩余 epoch / step。**

适用范围：
- `training_models/configs/*.yaml`
- `training_models/run_train.sh` 及任何启动脚本
- 云端 `/root/autodl-tmp/DGM4_Project/training_models/` 对应文件

必须满足：

1. `save_steps` / `save_strategy` 不允许设为 `no` 或 0；默认 ≤ 500 step 一次
2. 不允许 `save_only_model: true`（会丢失 optimizer / scheduler / RNG）
3. `resume_from_checkpoint` 字段保留（值可为 `null`）
4. `trainer_state.json` 必须随 checkpoint 落盘
5. `autodl halt` / `shutdown` 必须放在训练进程退出 *之后*，预留 ≥ 30s 缓冲
6. `trap cleanup EXIT` 模式优先于 `&&` 链
7. `logging_dir` 不允许被启动脚本清空

违反须 commit message 显式声明 `BOUNDARY-OVERRIDE: training-resume` + 用户书面同意。

---

## 2. 代码提交位置（Commit Location — MUST HOLD）

**训练 / 评测相关代码、配置、脚本的 git 提交默认在云端 AutoDL 主机上完成。**

适用范围：
- `training_models/configs/*.yaml`
- `training_models/*.py`、`training_models/*.sh`
- `training_models/src/` 下任何 LLaMA-Factory 改动
- 云端 `/root/autodl-tmp/DGM4_Project/` 内所有改动

流程：
1. 改动在云端文件直接编辑（或本地编辑后 `scp` 上传）
2. 在云端执行 `git add` / `git commit` / `git push`
3. 本地通过 `git pull` 拉取，**不在本地 commit 训练相关文件**

例外（允许本地 commit）：
- `manuscript/`、`docs/`、`notes/`、`references/`
- 本地元数据 (`CLAUDE.md`、`AGENTS.md`、`.codex/`、`scripts/` 下本地工具)
- 不上传到云端的纯本地分析代码

---

## 3. Canonical 目录与脚本契约（path conventions & launcher contract）

新对话处理训练 / 评测时，**必须**遵守以下目录布局与脚本接口契约。
具体模型名 / config 文件名 / 哪份 adapter 是"当前最佳" / 当前 run_name —
**全部**属于状态信息 → 见 `docs/DEVELOPMENT_LOG.md` 最新一节。

新增其它模型（LLaVA / InternVL / BLIP / 其他 VLM）时只需遵循下述
模式落 yaml + 复用三个启动脚本，**不需要**修改本文件。

### 3.1 目录布局（cloud: `/root/autodl-tmp/DGM4_Project/training_models/`）

```
training_models/
├── configs/                                            # 所有 yaml 集中放
│   └── <model><size>_<method>_<task>_<env>.yaml
├── outputs/<model_family>/<task>/<run_name>/           # 训练产出
│   └── eval_results__<run_name>.json                   # 评测结果（按 adapter 名自动）
├── logs/<model_family>-<task>-<run_name>/              # 训练日志（被 .gitignore）
├── eval_model_batch.py                                 # 主评测脚本（batched）
├── eval_model.py                                       # 旧单样本评测（对照保留）
├── run_train.sh                                        # 训练启动器
├── run_eval.sh                                         # 评测启动器
└── run_train_then_eval.sh                              # 一体化 train→eval→halt
```

举例：
- Qwen3-VL：`outputs/qwen3-vl-8b/dgm4-instruct/lora-sft-fast/`
- 假设加 LLaVA-v1.6：`outputs/llava-v1.6/dgm4-instruct/lora-sft-fast/`
  （无需改本文件，按模式即可）

### 3.2 启动脚本接口契约（interface, not implementation）

- `run_train.sh [config_name.yaml]`
  - 加载 `configs/<config_name.yaml>` 调 llamafactory-cli train
  - 自带 `trap cleanup EXIT`：进程退出后 ≥ 30s buffer → `autodl halt`
  - 默认 config 名只是占位；**任意**模型 yaml 都应能被传入

- `run_eval.sh [adapter_path] [batch_size] [limit]`
  - 调 `eval_model_batch.py --adapter <path> --batch-size <N> --limit <L>`
  - 自带相同 trap EXIT shutdown 链
  - canonical 默认 `batch_size=24, oom_fallback_batch=16`
  - 环境：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

- `run_train_then_eval.sh [config_name] [adapter_path]`
  - 单 trap 包住 train + eval，避免内层脚本提前 halt
  - 训练失败时跳过评测仍 halt

### 3.3 评测脚本契约（`eval_model_batch.py` 必须实现）

- 全 12 项 Table 2 指标（Binary AUC/EER/ACC、Multi-Label mAP/CF1/OF1、
  Image IoUmean/IoU50/IoU75、Text Tok_P/R/F1）
- `--output` 默认 `None` → 自动 `eval_results__<adapter_basename>.json`
  存到 `os.path.dirname(adapter)/`，**避免跨实验互相覆盖**
- 三段式 OOM 回退：primary batch → `--oom-fallback-batch` chunks → single-sample
- 输出 per-class 准确率（verdict_acc / category_acc）
- stdout 打印进度 / 解析率 / GPU peak

### 3.4 续训 / 二次微调入口字段（YAML，与模型无关）

- `adapter_name_or_path: <path>` → 二次微调起点（从已有 adapter 继续）
- `resume_from_checkpoint: <checkpoint-N>` → 断点续训起点（从某 step 继续）
- 两者**互斥**，不允许同时设置

### 3.5 当前活跃配置 → 永远查 DEV log

具体哪份 yaml 是当前主用、哪份 adapter 是当前最佳、哪个 run_name 在跑、
钱包余额、最近 loss/指标，**全部**从 `docs/DEVELOPMENT_LOG.md` 最新
一节读取。不要把当前活跃实验钉死在本文件里。

---

## 4. 状态文件分工（State File Layout）

| 文件 | 内容 |
|---|---|
| `docs/DEVELOPMENT_LOG.md` | **运行状态**：训练 run 结果、loss、余额、待办、最近时间线 |
| `docs/PROJECT_GOALS.md` | 论文定位 + 技术路线（静态） |
| `docs/AUTODL_ENVIRONMENT.md` | SSH / 硬件 / 路径（静态） |
| `docs/REPOSITORY_OPERATIONS.md` | git / push 助手（静态） |
| `CLAUDE.md`（本文件） | 边界 + canonical 文件清单（静态） |

修改本文件时只动**规则**；状态信息一律写 `docs/DEVELOPMENT_LOG.md`。

---

## 5. State Sync & Push Triggers（3-tier 仓库架构）

```
   Cloud (active work)  ─push (after approval)─▶  GitHub remote
   /root/autodl-tmp/DGM4_Project/                  Zkichn/DGM4_Project
         │                                         @ autodl-training
         │ state sync (no approval)
         ▼
   Local (handoff for new conversations)
   C:\Users\Administrator\Desktop\Paper-DGM4\
```

### 5.1 触发本地同步（无需审批）

以下任一发生即更新本地 `docs/DEVELOPMENT_LOG.md`：

- 训练运行结束（任意结果）
- 评测运行结束
- 用户做出关键决策（config / lr / dataset / route 分叉）

同步内容必须包含：对应云端 commit hash + 简要总结 + 关键指标 / 参数。

### 5.2 触发 GitHub push（**必须用户书面审批**）

§5.1 任一事件发生 **AND** 用户明确说 "push" / "推" / "确认上传"。

### 5.3 Pre-push checklist（每次 push 前 MUST DO）

1. `git status --short` 列出工作区
2. `git diff --cached --stat` 列出已暂存
3. 把这份清单给用户看
4. 等用户明确 OK
5. 才执行 `git push`

**禁止** `git add .` / `git add -A`。一律 `git add <具体路径>`。

### 5.4 模型权重 push 政策

- 训练完成 + 用户审批后 → `adapter_model.safetensors` 可 push（~167 MB / 个）
- `adapter_config.json` / `training_args.bin` / `*_results.json` / `trainer_state.json` /
  `*_loss.png` 同上，皆为小元数据可入库
- 以下永久 `.gitignore` 屏蔽，永不入库：`optimizer.pt` / `scheduler.pt` /
  `rng_state.pth` / `scaler.pt` / `global_step*` / `checkpoint-*/` 子目录 /
  与 base 模型重复的 tokenizer 文件
