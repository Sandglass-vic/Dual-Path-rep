# DualPath 复现文档索引

建议按以下顺序阅读：

1. [项目概览](intro.md)：目标、边界、当前状态和下一步。
2. [论文对齐、模型与 KV-cache](paper-and-model.md)：论文统计、模型选择、容量和 block 格式。
3. [目录与环境约定](environment.md)：代码/Lustre 路径、`RESULT_DIR` vs `REPO_RESULT_DIR`。
4. [GAIA 平台](platform.md)：Slurm、H100、IB/NIC 和 Lustre。
5. [容器与派生镜像](containers.md)：SGLang、Enroot、Mooncake 和 Router。
6. [Unified 运行手册](runbook-unified.md)：模型准备与单节点基线。
7. [PD 运行手册](runbook-pd.md)：1P1D、1P2D、2P2D 和已验证结果。
8. [HiCache 与 DualPath](hicache-and-dualpath.md)：L3 命中语义、数据流与消融开关。
9. [Mooncake Store 与 Lustre POSIX DFS](mooncake-store.md)：版本、原子发布、配置、实测和限制。
10. [Fake 数据集](fake_dataset.md)：length-only 轨迹合成、格式和 replay 要求。
11. [Benchmark 规范](benchmark.md)：论文式动态 rollout 和实验矩阵。
12. [指标规范](metrics.md)：唯一 metric 来源——**trajectory_jct**、
    per-agent mean TTFT/TPOT（agent 内先平均）、代表轨迹 vs 全量 fullchain、
    TTFT Timing V3 分解、task/dataset/agents/rep 产物布局。
13. [Dynamic L3 Read Path](dynamic.md)：request-scoped path 决策与交替策略设计。
14. [三条 read path 对比](dualpath-path-comparison.md)：GAIA TP8 实测的 JCT/TTFT/TPOT、
    Lustre 聚合带宽和排队机制。
15. [已修复缺陷的根因](dualpath-fixes.md)：五个 L3 路径缺陷的根因与防回归断言。
16. [pza100 异平台参考](pza100.md)：g5/g6 A100 的差异，以及存储不对称如何反转 dynamic 结论。
17. [里程碑与风险](milestones-and-risks.md)：完成标准、风险和执行顺序。
18. [复现与安全](reproducibility-and-security.md)：结果元数据、Git 和凭据规则。
19. [参考资料](references.md)：论文和官方文档链接。

## 信息唯一来源

- 项目路径：`environment.md`
- 当前镜像与版本：`containers.md`
- 平台硬件与网络：`platform.md`
- PD 命令和实测：`runbook-pd.md`
- Fake trace 算法和统计：`fake_dataset.md`
- KV-cache 层次语义：`hicache-and-dualpath.md`
- Mooncake Store/Lustre backend：`mooncake-store.md`
- 指标、字段、trajectory_jct / per-agent mean / 代表轨迹 / full-chain / Timing V3：
  `metrics.md`（唯一 metric 来源；见其中「口径速查」）
- 三条 read path 的性能结论：`dualpath-path-comparison.md`
- 已修复缺陷的根因：`dualpath-fixes.md`

## 脚本布局（摘要）

```text
scripts/
  pd_alloc.sh                                    # salloc、默认值与角色配置
  pd_benchmark.sh                                # 节点自动识别与启动入口
  tests/runtime/                                 # P/D/client 运行实现细节
  pd_plot.sh                                     # 独立后处理入口
  benchmark/                                    # generate/replay/collect/summarize
  smoke/                                        # Mooncake Store/DFS smoke
  plot/                                         # 画图
  infra/                                        # patch + 镜像构建
  replay_*.py / generate_*.py
```

仓库侧产物：`results/latency-benchmark/<TASK_NAME>/<DATASET>/agents-<N>/rep-<R>/`
（gitignore）。

当前运行方式：`scripts/pd_alloc.sh P D` 只申请节点并固定长期 server 配置；分别
SSH 到 P/D 节点启动 `pd_benchmark.sh prefill|decode` 后，在 client 节点按 run
设置 `DATASET`、`AGENTS`、`REPETITION` 和 `TASK_NAME`。长期 server 和 allocation
级 L3 KV 可被多个 client run 复用；日志/结果按 `TASK_NAME` 隔离。完整流程见
[PD 分离运行手册](runbook-pd.md#当前手工启动流程)。
