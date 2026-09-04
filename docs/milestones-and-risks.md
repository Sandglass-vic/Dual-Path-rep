# 里程碑、风险与执行顺序

> 返回 [文档索引](README.md)

## 结果判定

当前里程碑 M1 的完成标准：

- Qwen2.5-32B 在单节点 `TP=8` 稳定运行；
- 4K 和 32K completion 正确完成；
- 确定性请求可重复；
- 记录模型 revision、镜像 SHA-256、Enroot/Slurm/SGLang 命令；
- 本里程碑不要求 Mooncake、PD 或 SWE-bench。

里程碑 M2 的完成标准：

- 派生镜像可 import `mooncake.engine`、`mooncake.store` 和 `sglang_router`；
- 1P1D 和 1P2D 均通过 32K 请求；
- 1P2D 两个 Decode 实例都有流量；
- Decode 写出的 KV 可持久化到 Lustre；
- 后续 Prefill 能从 Lustre 命中同一 trajectory 的前缀；
- 32K/48K/64K 下能观察到 KV 读取随 context 增长；
- Prefill 节点存储带宽高于 Decode 节点，形成论文所述不平衡。

里程碑 M3 的完成标准：

- DE-read path 可用；
- 单路径与双路径总 GPU 数一致；
- 双路径提高聚合 Lustre 读取带宽并降低 JCT 或 TTFT；
- TPOT 无显著回归；
- 所有结论以实测数据为准，不以理论峰值代替。

## 风险与决策点

- **镜像兼容性**：当前基础镜像为 NGC `26.01-py3`，实测 SGLang `0.5.7+nv26.1`，有 PD/HiCache adaptor 但没有 Mooncake、NIXL 和 Router。当前统一推理直接使用基础镜像；PD 使用独立派生镜像。
- **Mooncake 版本**：CUDA 13 wheel 0.3.9 已通过同节点 Qwen2.5-32B PD 验证；不要未经测试把 0.3.13 与旧 SGLang 固化在一起。
- **PD layout 一致性**：Prefill/Decode 必须固定相同 `tp-size`、`kv-cache-dtype`、page size、模型 revision 和 attention/KV layout；已知不一致会导致连接冲突或越界写。
- **Router 缺失**：基础镜像中没有 `sglang_router`，PD 前必须安装并固定 Router 版本。
- **Lustre backend 语义**：共享目录不自动等于正确的分布式 KV store。需要验证 key namespace、一致性、并发写、eviction 和重启恢复。
- **HiCache L2 强制分配**：当前 SGLang 0.5.7 强制 host pool 大于 device pool，`--hicache-size 0` 不能关闭 L2。本项目接受原生层次缓存；需要归因 L3 时通过重启或清本地 cache 验证。
- **64K workload**：base 模型原生支持；Instruct checkpoint 默认配置不直接支持，避免不经说明地修改 RoPE。
- **PD 与 DualPath 混淆**：PD transfer 是 PE→DE；DualPath 新路径的关键方向是 storage→DE→PE，方向相反。
- **流量隔离权限**：IB VL/QoS 很可能需要管理员协助，先观测干扰，再决定是否申请配置。
- **共享存储噪声**：Lustre 是共享基础设施，必须重复实验并记录并发负载，否则小幅差异不能归因于系统改动。
- **密钥泄露**：历史参考脚本中的明文 NGC key 必须吊销，后续结果包不得包含 `.credentials`。

## 推荐执行顺序

```text
[完成] 平台探测、26.01 镜像和 Qwen2.5-32B revision
[完成] Mooncake + Router 派生镜像
[完成] 1P1D、1P2D、2P2D 短请求 RDMA smoke
[完成] 32K/48K/64K fake length trajectory
[完成] HiCache file backend 单节点 Lustre L3 重启命中
[完成] 1P1D Decode 写 Lustre L3、重启后 Prefill 读 L3
[完成] fake trajectory 并发离线 replay 最小验证
[完成] 1P1D+L3 单条 94 轮 / 24.8K 完整 trajectory
[完成] 1P2D 共享 L3 写入与两个独立前缀重启恢复
[完成] 1P2D+L3 / 4 agents × 4 turns 并发重启恢复
[完成] 流式客户端指标、legacy TTFT proxy 与 full-chain timing
[完成] 1P2D+L3 / 32K、48K、64K × 4、8、16 agents 四轮矩阵
[待做] Unified TP=8 / 32K 稳态
[待做] 1P1D、1P2D / 32K、48K、64K
[下一步] 消融实验与实际长 context 稳态负载
  -> L3 read、Decode offload、chunking 和 overlap 一键消融
  -> fake/论文 trace replay
  -> 单路径瓶颈确认
  -> DualPath DE-read 代码改造
```

当前已完成 Mooncake 派生镜像、1P1D/1P2D/2P2D 短请求 RDMA smoke、
Mooncake Lustre DFS 正确性、流式指标和历史四轮矩阵。Job `65566` 因
`dgx-gaia-53` 失联失败；正式作业统一排除
`dgx-gaia-18,dgx-gaia-30,dgx-gaia-46,dgx-gaia-53` 后重提 32K 完整 rollout 的
32/64/128-agent 矩阵（当前 Job `65593`，节点 `dgx-gaia-[14-16]`）。指标口径见
[指标规范](metrics.md)；节点故障记录见 [平台说明](platform.md)。DualPath
DE-read 与 layer-wise streaming 尚未完成。
