# 论文对齐、模型与 KV-cache 格式

> 返回 [文档索引](README.md)

## 论文配置对齐

论文中与本项目直接相关的设置如下：

- 模型：`Qwen/Qwen2.5-32B`，Dense GQA；
- 默认资源比：1P2D，其中 P/D 均以“节点”为单位；
- SGLang 基线：SGLang commit `19089aa`、HiCache、Mooncake Store、Mooncake Transfer Engine、3FS；
- Qwen 32B 的论文 SGLang 基线采用 `TP=8`；
- DualPath 为 Qwen 32B 分配 320 GB 主机 DRAM KV buffer/pool；
- 关闭 speculative decoding；
- 数据集每档 500 条 trajectory；
- 离线指标为整个任务的 JCT，在线指标为 TTFT、TTST、TPOT；
- 在线 SLO：`TTFT ≤ 4 s`、`TPOT ≤ 50 ms`。
- Figure 7 离线 agent batch size 为 512、1024、2048、4096；超过每档 500 条
  基础 trajectory 时，附录 A.3 通过采样已有 trajectory 并前置随机 `A1`、
  `g1=1` 的 synthetic round 生成额外轨迹。

以上是论文配置与 SLO；本项目的实际测量边界、聚合方法和字段定义统一见
[指标规范](metrics.md)。

论文 trace 的均值：

| Max Agent Length | 平均轮数 | 每轮 Append | 每轮 Gen | Total | 平均 Context |
|---|---:|---:|---:|---:|---:|
| 32K | 60 | 608 | 148 | 28,639 | 17,183 |
| 48K | 106 | 474 | 172 | 42,607 | 25,120 |
| 64K | 157 | 429 | 176 | 55,958 | 32,721 |

论文的每轮 replay 语义为：

```text
Context_(i+1) = A1 + G1 + A2 + G2 + ... + Ai + Gi
Prompt_(i+1)  = Context_(i+1) + A_(i+1)
```

其中 `Ai` 是该轮新增输入，`Gi` 是该轮模型输出。客户端必须保留真实生成结果并在下一轮重新提交完整上下文，不能把 500 条 trajectory 简化为 500 个互不相关的长 prompt。

### SWE-bench 的定位

后续初步计划使用 SWE-bench，建议先选 SWE-bench Verified 做功能和 agent trajectory 采集，再根据成本扩展。它适合产生“读代码—运行命令—修改—测试”的多轮 agent 轨迹，但不等同于论文的生产 agentic RL trace：

- SWE-bench 是任务/正确率数据集，论文实验使用的是轨迹长度统计和 replay；
- 复现系统性能时，应先运行 agent 得到每轮 `append_len`、`gen_len` 和累计 context，再冻结 trace 做 replay；
- 不能在比较 Basic 与 DualPath 时让模型重新做不同的 agent 决策，否则 workload 不一致；
- SWE-bench 需要 Instruct 模型、chat template、tool/terminal sandbox 和 agent harness，这些均不属于当前 Qwen2.5-32B 推理 smoke test。

因此模型分两条轨道：

- 系统性能、论文对齐：`Qwen/Qwen2.5-32B` base checkpoint；
- SWE-bench 功能任务：`Qwen/Qwen2.5-32B-Instruct`。

当前只需选择其中一个 checkpoint 跑通推理。若希望人工检查回答质量，优先下载 Instruct；若希望尽早与论文配置一致，优先 base。

## 模型、精度与容量

### 模型选择

第一选择为论文使用的基础模型：

```text
Qwen/Qwen2.5-32B
```

它具有 64 层、40 个 attention heads、8 个 KV heads、head dimension 128，原生最大上下文为 131,072。若后续需要真实对话模板，可增加 `Qwen/Qwen2.5-32B-Instruct` 对照组，但其默认 `config.json` 最大长度为 32,768；48K/64K 需要额外配置 YaRN，会引入新的实验变量。

因此，系统性能主实验使用 base checkpoint，更接近论文且避免 64K 长度外推配置差异。

### KV-cache 容量核算

Qwen2.5-32B 的逻辑 KV-cache 大小（不计页表、对齐和框架元数据）为：

```text
每 token 字节数
= 64 layers × 8 KV heads × 128 head_dim × 2(K,V) × dtype_bytes
= 256 KiB/token（BF16）
= 128 KiB/token（FP8 KV）
```

本项目 BF16 逻辑 KV、cache hit 和有效带宽公式统一见
[Cache hit 与 KV 数据量](metrics.md#3-cache-hit-与-kv-数据量)。

由此得到：

- 单条 32K context：BF16 约 8 GiB，FP8 约 4 GiB；
- 单条 48K context：BF16 约 12 GiB，FP8 约 6 GiB；
- 单条 64K context：BF16 约 16 GiB，FP8 约 8 GiB；
- 320 GiB 约可容纳 1.31M BF16 KV tokens 或 2.62M FP8 KV tokens。

`320 GB` 在论文中是主机 DRAM 分配，不应误写为单卡 HBM KV 容量。初期先使用 BF16 权重与默认 KV dtype，待基线稳定后再增加 FP8 KV 对照；不能在主对比中同时改变 PD、KV dtype 和存储后端。

### 论文中的 KV-cache 存储格式

以下格式来自论文附录 A.5 及 §4.1，作为后续 Lustre backend 与 DualPath 数据路径实现的规范基线。

论文定义两种 block：

```text
Layer Block: [1, tokens, bytes_per_layer_per_token]
Full Block:  [num_layers, tokens, bytes_per_layer_per_token]
```

- `tokens` 是 block 内 token 数，即 `block_size`；
- `bytes_per_layer_per_token` 是单层、单 token 的 K/V 数据字节数；
- Qwen2.5-32B 有 `num_layers=64`；
- 对 BF16 KV，`bytes_per_layer_per_token = 8 × 128 × 2(K,V) × 2 bytes = 4096 bytes`；
- 因此 Qwen2.5-32B 的 BF16 Full Block 为 `64 × block_size × 4096 bytes`；
- 论文举例在 Decode 阶段每积累一个完整 block（例如 64 tokens）就持久化，此时 Qwen BF16 Full Block 的有效载荷约为 16 MiB，FP8 KV 约为 8 MiB。

布局用途：

- 与持久化存储交互时只使用 **Full Block**；
- PE/DE host buffer 与 PE HBM 之间按层流式搬运时使用 **Layer Block**；
- 连续拼接同一 token 区间的 64 个 Layer Block，可直接形成一个 Full Block，避免额外 layout conversion；
- KV-cache 在持久化存储中按 trie 组织，每个 trie node 对应一个 Full Block；
- trie key 必须由模型/版本、token block、KV dtype、layout、TP shard 等共同确定，避免不同实验配置误命中；
- 尾部不足一个完整 block 的 token 需要保留在内存中，达到 `block_size` 后再持久化；trajectory 结束时是否 flush 尾块必须固定并记录。

项目的第一版 Lustre 格式建议保持与论文的逻辑形状一致，但增加自描述 header：

```text
KVBlockHeader
  magic/version
  model_revision
  num_layers
  block_size
  valid_tokens
  kv_heads
  head_dim
  dtype
  tp_size/tp_rank
  layout
  checksum

payload
  Full Block [num_layers, valid_tokens, bytes_per_layer_per_token]
```

物理文件组织需要通过实际 I/O 测试决定。不能未经测试就采用“一 block 一小文件”，否则 Lustre metadata 开销可能掩盖数据带宽瓶颈。建议同时评估：

1. 每个 Full Block 一个对象/文件，作为正确性基线；
2. 多个 Full Block append 到 segment file，并维护独立索引；
3. SGLang HiCache backend 的原生序列化格式。

无论采用哪种物理格式，结果中都必须记录 `block_size`、字节序、dtype、校验方式、TP layout、segment 大小和 Lustre stripe。
