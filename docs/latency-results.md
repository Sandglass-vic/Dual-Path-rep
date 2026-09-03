# 1P2D+L3 四轮延迟结果

> 返回 [文档索引](README.md)；指标定义见 [指标规范](metrics.md)。

> 历史诊断结果：本 Job 使用旧 SGLang file backend、Populate/Measured
> 隔离、cache flush 和 storage quiescence wait。它不符合当前论文式动态
> rollout，不得作为正式 JCT 或 Figure 7 复现结果。

## 实验配置

- 最终 Slurm Job：`65032`
- 模型：Qwen2.5-32B，Prefill TP=8，两个 Decode 各 TP=8
- 数据档：32K、48K、64K length-only fake trajectory
- 并发 agent：4、8、16
- 每条 trajectory：仅执行原始前 4 turns
- 每个组合重复 3 次，共 27 个 measured runs、1008 个请求
- 所有 agent 通过 barrier 同时开始；这是一组固定 agent 数的离线并发结果
- 每个 MaxLen 档先 Populate L3，Measured 前重启服务，每个 run 前清空本地 cache

SGLang file backend 并发覆盖文件会产生 `Short read`。最终实现采用单向隔离：
Populate 阶段仅 Decode 向 L3 根目录写入；Measured 阶段 Prefill 只读该不可变
目录，两个 Decode 分别向独立临时 scratch 目录 offload。所有目录在 Job 退出时
自动删除。

## 主要结果

以下均为 3 次重复的中位数。TTFT、TTST 和 TPOT 是所有请求均值的 run-level
中位数。

```text
MaxLen  Agents  JCT(s)  TTFT(ms)  TTST(ms)  TPOT(ms)
32K          4    4.40     335.8     342.4      8.20
32K          8    4.98     443.3     449.5      8.23
32K         16    7.05     779.6     787.1      8.94
48K          4    4.42     247.3     253.6      8.64
48K          8    5.36     442.4     449.4      8.18
48K         16    6.86     597.7     605.8      9.16
64K          4    6.95     409.3     425.4     14.56
64K          8    8.06     492.9     508.9     16.13
64K         16    9.95     759.6     777.5     17.98
```

主要趋势：

- 每个数据档中，agent 从 4 增加到 16 时 JCT 和 TTFT 均明显上升。
- 32K 和 48K 的 TPOT 约为 8–9 ms；64K 为 14.6–18.0 ms。
- 64K 的 16-agent JCT 中位数为 9.95 s，是本矩阵最高值。

## 有效性与解释边界

结构化校验结果：

```text
runs                         27
rounds                     1008
request timing coverage    100%
L3 timing coverage          77.28%
batched token chunks            0
stream retries                  0
failed requests                 0
```

`32K/48K/64K` 是完整源 trajectory 的 MaxLen 分类，不是这四轮请求的实际长度。
前四轮实测 prompt 范围分别为 315–1763、237–1375、176–1181 tokens。因此这些
结果只能比较三类 trajectory 的前四轮，不能解释为 32K/48K/64K 长上下文性能。

本次 `Allocation Wait` 均为 0。`ttft_breakdown.svg` 是可重叠的历史
Prefill-side proxy，不是论文 Figure 12 的精确复现。各段、`Unattributed` 和
full-chain 口径统一见 [TTFT breakdown](metrics.md#6-ttft-breakdown)；Job
`65144` 的具体数值见 [TTFT 全链路历史测例](ttft-fullchain.md)。

## 产物

仓库侧历史图表目录已清理；数值以上表为准。原始逐请求结果与服务日志仍在
Lustre Job `65032` 的 runtime `results/` 与 `logs/`；测试 KV-cache 已清理。
当前正式仓库侧产物示例见 `results/latency-benchmark/65593/`（口径不同，
勿与本页混报）。
