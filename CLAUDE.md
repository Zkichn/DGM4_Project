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

## 3. 训练 / 评测 pipeline 规约 → 见 `docs/TRAINING_PIPELINE.md`

目录布局、启动脚本接口契约、评测脚本契约、续训入口字段全部记录在
`docs/TRAINING_PIPELINE.md`。新对话处理训练 / 评测前先读该文件。
当前活跃 yaml / adapter / run_name 见 `docs/DEVELOPMENT_LOG.md` 最新一节。

---

## 4. 状态文件分工（State File Layout）

| 文件 | 内容 |
|---|---|
| `docs/DEVELOPMENT_LOG.md` | **运行状态**：训练 run 结果、loss、余额、待办、最近时间线 |
| `docs/PROJECT_GOALS.md` | 论文定位 + 技术路线（静态） |
| `docs/AUTODL_ENVIRONMENT.md` | SSH / 硬件 / 路径（静态） |
| `docs/TRAINING_PIPELINE.md` | 训练 / 评测脚本与目录契约（静态） |
| `docs/REPOSITORY_OPERATIONS.md` | 3-tier 同步与 push 规约 + 本地 helper（静态） |
| `CLAUDE.md`（本文件） | 边界 + 状态文件入口（静态，极简） |

修改本文件时只动**规则**与**入口指向**；具体内容一律放对应状态文件。

---

## 5. 仓库同步与 push 规约 → 见 `docs/REPOSITORY_OPERATIONS.md`

3-tier 架构（cloud / local / GitHub）的状态同步触发、push 审批、pre-push
检查清单、模型权重 push 政策，全部记录在 `docs/REPOSITORY_OPERATIONS.md`。
执行任何 push 或权重提交前先读该文件。
