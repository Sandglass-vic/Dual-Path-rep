# DualPath 文档索引

建议按以下顺序阅读：

1. [项目概览](intro.md)：目标、边界、当前状态和下一步。
2. [论文对齐、模型与 KV-cache](paper-and-model.md)：论文统计、模型选择、容量和 block 格式。
3. [目录与环境约定](environment.md)：代码/Lustre 路径、`RESULT_DIR` vs `REPO_RESULT_DIR`。
4. [GAIA 平台](platform.md)：Slurm、H100、IB/NIC 和 Lustre。
5. [容器与派生镜像](containers.md)：SGLang、Enroot、Mooncake 和 Router。
6. [Unified 运行手册](runbook-unified.md)：模型准备与单节点基线。
7. [PD 运行手册](runbook-pd.md)：1P1D、1P2D、2P2D 和已验证结果。
8. [HiCache 与 DualPath](hicache-and-dualpath.md)：L3 命中语义、测试清理和 DE-read 设计。
9. [Mooncake Store 与 Lustre POSIX DFS](mooncake-store.md)：版本、原子发布、配置、实测和限制。
10. [Fake 数据集](fake_dataset.md)：length-only 轨迹合成、格式和 replay 要求。
11. [Benchmark 规范](benchmark.md)：论文式动态 rollout 和实验矩阵。
12. [指标规范](metrics.md)：唯一 metric 来源——**trajectory_jct**、
    per-agent mean TTFT/TPOT（agent 内先平均）、代表轨迹 vs 全量 fullchain、
    task/dataset/agents/rep 产物布局。
13. [当前数据流与消融开关](dataflow-and-ablation.md)：L3、P→D、chunk 传输边界和一键 profile。
14. [四轮延迟结果](latency-results.md)：历史 Job `65032`（诊断口径）。
15. [TTFT 全链路历史测例](ttft-fullchain.md)：历史 Job `65144` 数值。
16. [完整轨迹历史测例](trajectory-metrics.md)：历史 Job `65502` KV/带宽现象。
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
- 指标、字段、trajectory_jct / per-agent mean / 代表轨迹 / full-chain 与产物：
  `metrics.md`（唯一 metric 来源；见其中「口径速查」）

## 脚本布局（摘要）

```text
scripts/
  pd_*_benchmark.sbatch + pd_benchmark_lib.sh   # 正式矩阵
  benchmark/                                    # generate/replay/collect/summarize
  smoke/                                        # Mooncake Store/DFS smoke
  plot/                                         # 画图
  infra/                                        # patch + 镜像构建
  replay_*.py / generate_*.py / pd_{prefill,decode}*.sh
```

仓库侧产物：`results/latency-benchmark/<TASK_NAME>/<DATASET>/agents-<N>/rep-<R>/`
（gitignore）。
