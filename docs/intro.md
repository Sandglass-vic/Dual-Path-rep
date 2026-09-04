# DualPath 复现实验概览

本项目在 GAIA DGX H100 集群上，以 Qwen2.5-32B、SGLang、Mooncake、Slurm/Pyxis/Enroot 和 Lustre 分阶段复现 *DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference* 的关键趋势。目标是验证可复现的性能趋势，不承诺复现论文内部框架和 3FS 的绝对数值。

## 复现边界

- 可直接验证：Qwen2.5-32B、TP=8、PD 分离、多轮长上下文、外部 KV I/O 瓶颈。
- 近似复现：用 Lustre 替代 3FS，用 SGLang HiCache 管理外部 KV-cache。
- 需要开发：storage→Decode→Prefill 的第二读取路径、路径调度、layer-wise streaming 和流量隔离。
- SGLang 原生 Mooncake PD 仅提供 Prefill→Decode KV transfer，不等价于 DualPath。

## 当前状态

- Qwen2.5-32B base checkpoint 和 SGLang 26.01 基础镜像已固定。
- Mooncake `0.3.13.post1` + Router 0.3.1 派生镜像已构建；包含当前 SGLang
  所需的 DFS replica 配置回移。
- 1P1D、1P2D、2P2D 跨节点 TP=8 Mooncake RDMA 短请求 smoke 已通过。
- 32K、48K、64K 各 500 条 length-only fake trajectory 已生成。
- 离线 fake trajectory replay 已验证并发 agent、精确生成长度和真实输出累积。
- HiCache file backend 已通过单节点 Lustre L3 写入和服务重启后命中验证。
- 1P1D 已验证 Decode 写 Lustre L3、服务重启后 Prefill 跨节点恢复前缀。
- 1P1D+file L3 已完成 94 轮历史命中归因实验；其中逐轮 flush/wait 不计作性能结果。
- 1P2D 已验证共享 Lustre L3 写入、全服务重启后两个独立前缀恢复，以及两个 Decode 参与恢复请求。
- 1P2D+L3 已完成 4 agents × 4 turns 并发 replay；全服务重启后 16/16 轮均命中 L3。
- 历史 Job `65032` 已完成 27-run 四轮诊断矩阵，但因使用
  Populate/Measured、flush 和 storage wait，不作为论文性能结果。
- Mooncake Store POSIX DFS 已完成 Lustre 跨节点写后读、长度与 SHA-256 验证
  （Job `65548`），对象完整发布且未发生 short read。
- Qwen2.5-32B 1P1D HiCache + Mooncake DFS 已完成 4-turn 动态 smoke
  （Job `65557`），3/4 turns 自然命中且 DFS 实际产生数据。
- 正式 replay 对齐论文 buffer 语义：Prefill 在每个 PD ACK 后释放未锁定的
  L1/L2 page，跨 turn 走 Mooncake Store；短暂 settle 仅用于可见性，不调用
  全局 `/flush_cache`。
- 禁用逐请求 release 的 L1/L2 保留路径仅作对照，不得当作正式结果。
- Job `65566` 的 32K × 32/64/128-agent 正式矩阵在首轮失败：Decode-1 节点
  `dgx-gaia-53` 高负载下失联；另有历史故障节点 `dgx-gaia-30`、`dgx-gaia-46`。
  正式作业统一排除 `dgx-gaia-12,dgx-gaia-18,dgx-gaia-30,dgx-gaia-46,dgx-gaia-53`。详见
  [平台说明](platform.md)。
- Job `65593` 已按完整排除列表重提同一矩阵，运行节点为 `dgx-gaia-[14-16]`。
- 指标、字段与预期产物见 [指标规范](metrics.md)。
- 长上下文稳态和 DualPath DE-read 尚未完成。

## 关键约束

- 当前复现：SGLang 保留 L1/L2 **池**，正式路径在 PD ACK 后逐请求解锁并驱逐
  未锁定 page，禁止跨 turn 本地命中。
- 独立正确性仍可通过服务重启排除 L1/L2；正式性能路径默认 per-request
  release + L3 reload。
- 多节点作业提交前核对 [已知故障节点](platform.md#已知故障节点临时排除)。
- 每个测试使用 Job 专属 KV 目录，验收和统计完成后自动删除，日志与结构化结果保留。
- 多轮 replay 必须把真实模型输出加入下一轮完整 context，不能将 trajectory 展开成无关请求。
- 代码和文档在 `/labhome/sezhou/dual-path`；大型运行数据在计算节点可见的 `/mnt/lustre/users/sezhou/dual-path`。

## 下一步

1. 使用一键 profile 执行 L3、Decode offload、chunking 和 overlap 消融。
2. 执行达到 32K/48K/64K 实际 context 的 1P1D/1P2D 稳态对照并确认 Lustre 瓶颈。
3. 瓶颈确认后开发 DualPath DE-read 和真正的 layer-wise streaming。

详细入口见 [文档索引](README.md)。
