# DualPath 复现实验概览

本项目在 GAIA DGX H100 集群上，以 Qwen2.5-32B、SGLang、Mooncake、Slurm/Pyxis/
Enroot 和 Lustre 分阶段复现 *DualPath: Breaking the Storage Bandwidth Bottleneck
in Agentic LLM Inference* 的关键趋势。目标是验证可复现的性能趋势，不承诺复现
论文内部框架和 3FS 的绝对数值。

异平台（g5/g6 A100）的差异与结论见 [pza100.md](pza100.md)。

## 复现边界

- 可直接验证：Qwen2.5-32B、TP=8、PD 分离、多轮长上下文、外部 KV I/O 瓶颈。
- 近似复现：用 Lustre 替代 3FS，用 SGLang HiCache 管理外部 KV-cache。
- 需要开发：layer-wise streaming 和流量隔离。
- SGLang 原生 Mooncake PD 仅提供 Prefill→Decode KV transfer，不等价于 DualPath。

## 当前状态

**双读取路径已实现并量化。** `DUALPATH_L3_READ_PATH` 支持 `prefill` /
`decode` / `dynamic` 三种模式，dynamic 按请求解析读取方（设计见
[dynamic.md](dynamic.md)）。GAIA TP8、32K、8 agent 实测 dynamic 的 JCT 比静态
Prefill 快 1.53x、TTFT 快 2.49x，Lustre 聚合读带宽从 7.59 翻到 14.79 GB/s；
完整分解见 [dualpath-path-comparison.md](dualpath-path-comparison.md)。

**瓶颈已定位在单引擎的串行服务能力，不在 Lustre。** 单节点 TP8 只拿到
7.6 GB/s，而 fio 标定单节点 8 jobs 即可读 10.77 GiB/s。Prefill 引擎容量约
1.16 req/s，8 agent 时饱和，排队吸收了全部积压；dynamic 把读取分到两侧后容量
翻倍、脱离饱和。

**五个 L3 路径缺陷已修复**，根因与防回归断言见
[dualpath-fixes.md](dualpath-fixes.md)。

**基础设施就绪。** Mooncake `0.3.13.post1` + Router 0.3.1 的 patched 镜像、
32K/48K/64K 各 500 条 length-only fake trajectory、Timing V3 全链路分解、
per-request 指标采集与绘图流水线均已可用。

**尚未完成**：dynamic 偶发的 Prefill 侧读取停顿未定位根因；48K/64K 长上下文
稳态；1P2D/2P2D 下的 dynamic 行为；layer-wise streaming。

## 关键约束

- 正式路径在 PD ACK 后逐请求解锁并驱逐未锁定的 L1/L2 page，禁止跨 turn 本地
  命中；跨轮复用必须走 durable L3。
- 禁用逐请求 release 的 L1/L2 保留路径仅作对照，不得当作正式结果。
- 独立正确性可通过服务重启排除 L1/L2；正式性能路径默认 per-request release +
  L3 reload。
- 多轮 replay 必须把真实模型输出加入下一轮完整 context，不能把 trajectory 展开
  成无关请求。
- 每个 client run 使用 `TASK_NAME` 专属日志和结果目录；长期 server 使用
  allocation 级共享 KV namespace。当前流程不自动删除这些目录。
- 多节点作业提交前核对[已知故障节点](platform.md#已知故障节点临时排除)。
- 代码和文档在 `/labhome/sezhou/dual-path`；大型运行数据在计算节点可见的
  `/mnt/lustre/users/sezhou/dual-path`，登录节点看不到 Lustre。

## 下一步

1. 定位 dynamic 偶发读取停顿的根因（Lustre 锁争用假设待验证）。
2. 把 dynamic 扩展到 1P2D/2P2D，验证多 Decode 下的选路与收益。
3. 48K/64K 长上下文稳态对照。
4. layer-wise streaming。

详细入口见 [文档索引](README.md)。
