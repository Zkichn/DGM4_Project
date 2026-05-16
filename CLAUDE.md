# CLAUDE.md — Repository Boundaries for Claude Code

本文件记录该仓库对 Claude 的硬性约束（boundaries）。在修改任何训练 / 评测相关
代码或配置之前，必须先读这一节。

---

## 训练断点续训约束（Training Checkpoint Resume — MUST HOLD）

**任何对训练相关参数、脚本或配置的修改，都必须保证训练在中断后可以从最近的
checkpoint 恢复，继续完成剩余 epoch / step，不允许从头重训。**

适用范围：
- `training_models/configs/*.yaml`
- `training_models/run_train.sh` 及任何启动脚本
- AutoDL 实例上 `/root/autodl-tmp/DGM4_Project/training_models/` 下对应文件
- 任何新增的训练 / 微调 / RLHF / 评测脚本

修改训练参数时必须同时满足以下条件：

1. **checkpoint 定期落盘**
   - `save_steps` / `save_strategy` 不允许设为 `no` 或 0
   - 修改 `save_steps` 时必须保留 ≤ 500 step 一次的频率（除非用户明确放宽）
   - 不允许设置 `save_only_model: true` 而丢失 optimizer / scheduler / RNG 状态
     —— 这会导致中断后只能继续推理、无法续训

2. **resume 入口保持可用**
   - 不允许删除或注释 `resume_from_checkpoint` 字段
   - 启动脚本必须暴露续训路径（环境变量或 CLI 参数），默认从 `output_dir`
     最近的 `checkpoint-*` 目录恢复
   - 修改 `output_dir` 时必须同时迁移已有 checkpoint，或保留兼容路径

3. **gradient_checkpointing 与续训互不冲突**
   - 启用 / 关闭 `gradient_checkpointing` 不影响 checkpoint 内容，可以自由切换，
     但切换后第一次启动必须先验证能从旧 checkpoint 加载成功

4. **关机前必须等 checkpoint 写完**
   - `autodl halt` / `shutdown` 必须放在训练进程退出 *之后*，并预留 ≥ 30s 缓冲
   - `trap cleanup EXIT` 模式优先，避免 `&&` 链在异常退出时漏掉续训点保存

5. **日志保留**
   - `logging_dir` 不允许被启动脚本清空
   - `trainer_state.json` 必须随 checkpoint 一同落盘（HF Trainer 默认行为，
     不要手动关闭）

违反以上任意一条都视为破坏仓库边界。如确需例外，必须在 commit message 中显式
声明 "BOUNDARY-OVERRIDE: training-resume" 并附带用户书面同意。

---

## 当前生效的训练配置参考

`qwen3vl_8b_lora_sft_dgm4_instruct_autodl.yaml` 中与续训直接相关的字段：

```yaml
gradient_checkpointing: true     # 显存优化，重算激活值；不影响 checkpoint
save_steps: 500                   # 每 500 step 落盘一次
save_only_model: false            # 必须 false，否则丢失优化器状态
resume_from_checkpoint: null      # 默认 null = 从 output_dir 自动找最新
```

修改这些字段时，参照本文件第一节的硬性约束。
