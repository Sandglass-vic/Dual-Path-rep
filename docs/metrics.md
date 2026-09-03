# 指标、采集与产物规范

> 返回 [文档索引](README.md)。本文是本项目所有 metric 定义的唯一来源。
> workload 和运行方法见 [Benchmark 规范](benchmark.md)，具体 Job 数值见各结果页。

## 1. 口径分层

阅读结果时先区分以下三类：

- **正式性能口径**：Mooncake Store 动态完整 rollout；不预 Populate，不逐轮
  flush/sleep/storage wait，不重试失败请求。
- **当前可自动汇总**：客户端 streaming 指标、request/L3/P→D timing、cache hit、
  聚合 CSV/JSON 和 SVG。
- **历史诊断口径**：Job `65032/65144/65502` 使用旧 file backend、预热或 cache
  隔离；只用于解释机制，不得与正式性能结果混报。

正式 32K × 32/64/128 agents × 3 rep 完整 rollout 属于第一类。Job `65566`
因 Decode-1 节点 `dgx-gaia-53` 失联失败；Job `65592` 仅排除 `53` 后又分到
含历史故障节点 `dgx-gaia-30` 的组，已取消。当前正式作业排除
`dgx-gaia-30,dgx-gaia-46,dgx-gaia-53`（见 [平台说明](platform.md)）。完成后检查两类 **results**（名称相同、语义不同，不可混用）：

```text
# Lustre runtime results：原始 rounds / trajectories / run.json（RESULT_DIR）
/mnt/lustre/users/sezhou/dual-path/results/latency-benchmark/<JOBID>/
/mnt/lustre/users/sezhou/dual-path/logs/latency-benchmark/<JOBID>/

# Labhome 仓库侧产物：汇总 CSV、SVG、full-chain（REPO_RESULT_DIR）
/labhome/sezhou/dual-path/results/latency-benchmark/<JOBID>/
```

当前仓库侧已整理产物示例：`results/latency-benchmark/65593/`。

## 2. 端到端客户端指标

所有 token latency 都由 replay client 的同一个 `time.perf_counter()` 时钟测量，
不存在跨节点时钟相减。

### JCT

- Batch JCT：`summary.json: wall_seconds`。
- 起点：所有 worker 已创建后，barrier 释放。
- 终点：最后一条 trajectory 完成或失败。
- 不包含模型/Router/Mooncake master 启动和 Job 退出清理。
- 多 agent JCT 是最晚完成 agent 的 critical path，不能把各 agent 时间相加。
- 单 trajectory JCT：`trajectories.jsonl: elapsed_seconds`，等于该 trajectory
  所有 `turn_wall_seconds` 之和。

单轮闭合为：

```text
turn_wall_seconds
  = latency_seconds
  + turn_client_runtime_seconds
  + turn_accounting_error_seconds
```

正式路径中的 accounting error 应只剩浮点误差。旧字段
`storage_sync_seconds`、`inter_turn_delay_seconds`、`cache_flush_seconds`
已删除，不再记录为零。

### Request latency

`latency_seconds` 是 client 发出 `/generate` 请求到完整 streaming response
结束的时间，包含 TTFT、后续 decode、SSE 和网络时间。

### TTFT

`ttft_seconds` 是 client 发出请求到收到第一个包含新 output token 的 SSE
payload 的时间。

第一个 token 在当前 PD 路径中由 Prefill forward、logits processor 和 sampler
产生，再作为 PREBUILT token 交给 Decode；它不是 Decode 的第一次常规 forward。

### TTST

`ttst_seconds` 是 client 发出请求到收到第二个 output token 的时间，不是第一、
第二 token 之间的间隔。

### ITL 与 TPOT

- `itl_seconds[i]`：相邻两个 output token 在 client 到达时间的差。
- `tpot_seconds`：该请求第一个 token 之后所有 ITL 的算术均值。
- 只有一个 output token 时 TPOT 为 `null`。

SGLang 固定 `stream_interval=1`。如果单个 SSE payload 同时新增多个 token，
`batched_token_chunks > 0`，该 run 无法提供真实逐 token 到达时间，汇总器会
判为无效。

### 聚合统计

单 run 对 TTFT、TTST、TPOT、ITL 输出 mean/P50/P90/P99。重复实验先在每个 run
内部计算均值，再对同一配置的 run mean 报告 median/min/max。

## 3. Cache hit 与 KV 数据量

### Prefix hit

- `cached_tokens`：SGLang response `meta_info.cached_tokens`。
- `cache_hit_rate = cached_tokens / prompt_tokens`。
- 它表示 HiCache 可复用前缀，可能来自 L1、L2 或 Mooncake Store。
- 正式性能路径不 flush 本地层，因此不能仅凭 `cached_tokens` 将命中归因于
  Lustre DFS。

### L3 read hit

- `l3_prefetched_tokens_logical`：`DUALPATH_L3_TIMING` 中各 TP rank 报告的逻辑
  token 数。
- `l3_hit_rate = l3_prefetched_tokens_logical / prompt_tokens`。
- 只有产生 L3 prefetch marker 的请求才是可观测 L3 read。
- 动态 rollout 不等待 Store 发布，所以早期 turn 的 L3 timing coverage 可以为
  零；coverage 是结果，不是必须达到的正确性阈值。

### Qwen2.5-32B BF16 KV 大小

```text
64 layers × 8 KV heads × 128 head_dim × K/V × 2 bytes
  = 262,144 bytes/token
  = 256 KiB/token
```

- 新增逻辑 KV：`(append_tokens + generated_tokens) × 262144`。
- turn 后逻辑 KV：`(prompt_tokens + generated_tokens) × 262144`。
- 容量图使用 GiB，即除以 `1024³`。

Mooncake Store 当前正式 replay 不在每轮同步扫描 DFS shard，因此不会为了画图
把 Lustre 文件统计加入 JCT。旧 `storage_bytes_after_turn` 和
`l3_physical_snapshot_gib` 只存在于 Job `65502` file-backend 历史数据；新运行
中若没有独立异步采样器，该物理值不可用，不能以零解释成“未写入”。

## 4. 服务端 timing marker

timing 镜像由 `scripts/infra/patch_sglang_benchmark_timing.py` 注入以下
request-scoped 日志，以 request ID 与客户端 `rounds.jsonl` 关联。

- `DUALPATH_REQ_TIMING`
  - 来源：Prefill scheduler。
  - 字段：queue、allocation、prefill/worker launch、forward duration 及关键
    时间戳。
  - 用途：request timing coverage、旧 proxy breakdown、full-chain Prefill 段。
- `DUALPATH_L3_TIMING`
  - 来源：Prefill HiRadix/HiCache storage prefetch。
  - 字段：Prefill engine、TP rank、read start/end/duration、tokens。
  - 用途：L3 read latency、L3 timing coverage、有效加载带宽。
- `DUALPATH_TTFT_P_TRANSFER`
  - 来源：Prefill disaggregation sender。
  - 窗口：第一个实际 KV chunk 提交到最终 `KVPoll.Success`。
  - 字段：start/end/duration、logical tokens。
- `DUALPATH_TTFT_D_RECEIVE`
  - 来源：Decode receiver。
  - 用途：Decode 接收和 KV commit 诊断。
- `DUALPATH_TTFT_D_DISPATCH`
  - 来源：Decode output processor。
  - 用途：KV ready 到 PREBUILT first-token dispatch。
- `DUALPATH_TTFT_HTTP_YIELD`
  - 来源：Decode HTTP server。
  - 用途：first-token dispatch 到首个 SSE payload yield。

Prefill 与 Decode 位于不同节点。原始 `perf_counter` 时间戳不能跨节点相减；
汇总器只计算同节点 duration，再用 request ID 连接链路。

## 5. Request、L3 与 PD timing

### Coverage

- `request_timing_coverage`：有 `DUALPATH_REQ_TIMING` 的请求数 / 总请求数。
- `l3_timing_coverage`：有 `DUALPATH_L3_TIMING` 的请求数 / 总请求数。
- `pd_timing_coverage`：有 `DUALPATH_TTFT_P_TRANSFER` 的请求数 / 总请求数。

正式矩阵要求 request 和 P→D coverage 至少 95%；关闭 Prefill L3 时要求 L3
coverage 严格为零。启用 L3 时不设置 coverage 下限。

### TP=8 L3 聚合

同一 request 必须取得一个 Prefill engine 的 8 个 TP rank timing。请求的
`read_seconds` 取八个 rank 中最大 duration，`l3_slowest_tp_rank` 记录瓶颈
rank；不能只用 rank 0 计算加载带宽。

### P→D timing 与有效带宽

```text
pd_effective_bandwidth_gbps
  = logical_tokens × 262144 / pd_transfer_seconds / 1e9
```

这是“首个 chunk 提交到全部传输完成”窗口上的逻辑有效带宽，不是 IB wire
bandwidth。Job `65144/65502` 的旧镜像从最后一个 chunk 才开始计时，数值不可与
新结果直接比较。

### Prefill L3 有效加载带宽

```text
prefill_l3_effective_bandwidth_gbps
  = l3_logical_tokens × 262144 / max(tp_rank_read_seconds) / 1e9
```

带宽统一使用十进制 GB/s；KV 容量使用二进制 GiB。

## 6. TTFT breakdown

项目存在两套 breakdown，必须明确标注，不能混用。

### Legacy decomposition proxy（已退役）

旧 `plot_latency_benchmark.py` 曾输出 Scheduling / Allocation Wait / L3 KV
Read / Prefill Compute / Unattributed。这些 duration 可能重叠，且
`Unattributed` 含 Router、P→D、Decode dispatch、SSE 等。Job `65032` 使用该
口径；正式路径不再生成 `ttft_breakdown.svg`。

### Full-chain additive breakdown（正式）

**Full-chain** 把客户端 TTFT 切成互斥、可加总分桶（与 legacy proxy 不同：
legacy 段可能重叠，且把 Router/P→D/Decode/SSE 等塞进 `Unattributed`）。

互斥桶：

| 桶 | 含义 |
|---|---|
| Prefill Bootstrap & Scheduling | bootstrap / queue / scheduling |
| L3 KV Read | Prefill 侧 L3 prefetch |
| Prefill Forward + Logits + Sampling | prefill compute + first-token sample |
| Prefill-to-Decode KV Transfer | P→D KV 传输窗口 |
| Decode Prebuilt-token Dispatch | KV ready → PREBUILT first-token dispatch |
| SSE Serialization & Egress | dispatch → 首个 SSE payload yield |
| Router, Client & Runtime Gaps | 闭合 TTFT 的剩余间隙 |

Prefill 侧重叠区间按 L3 read、scheduling、prefill、P→D 的优先级只归属一次。
Decode receive overlap 仅保留为诊断，不重复加入柱状图。所有桶必须满足：

```text
TTFT = Σ full-chain stages
abs(accounting_error_seconds) <= 1e-6
```

#### 产物粒度（重要）

- `fullchain/agents-<N>/fullchain_rounds.jsonl`：**全量 per-request** 明细。
  每一行对应一次 `(trajectory_id, turn_id)` / request，含各互斥桶秒数与
  `ttft_seconds`；不是配置级均值。
- `fullchain/agents-<N>/fullchain_summary.json`：在上述明细上的**聚合统计**
  （mean 等）。
- `ttft_fullchain_by_agents.svg`：按 `#Agents` 堆叠的互斥桶均值对比图。

正式路径**不**再输出单轨迹 fullchain 切片
（例如 `trajectory-source/fullchain_rounds.jsonl`、
`trajectory_ttft_breakdown.svg`）。`collect_benchmark_metrics.py` 默认对每个
`#Agents` 的 rep-1 生成全量 fullchain。

## 7. 当前自动采集流水线

正式提交入口（每种拓扑仅保留一份）：

```text
scripts/pd_1p1d_benchmark.sbatch
scripts/pd_1p2d_benchmark.sbatch
scripts/pd_2p2d_benchmark.sbatch
```

Job 结束时统一调用 `collect_benchmark_metrics.py`。

客户端：

```text
replay_fake_agent_traces.py
  -> rounds.jsonl
  -> trajectories.jsonl
  -> summary.json
```

服务端：

```text
Prefill/Decode logs
  -> DUALPATH_REQ_TIMING
  -> DUALPATH_L3_TIMING
  -> DUALPATH_TTFT_P_TRANSFER
  -> DUALPATH_TTFT_D_RECEIVE
  -> DUALPATH_TTFT_D_DISPATCH
  -> DUALPATH_TTFT_HTTP_YIELD
```

Job 级汇总与作图（脚本布局：`scripts/plot/` 作图，`scripts/infra/` 镜像补丁）：

```text
collect_benchmark_metrics.py
  -> summarize_latency_benchmark.py
       -> merged_rounds.jsonl / runs.csv / aggregate.csv / validation.json
  -> summarize_ttft_fullchain.py
       -> fullchain/agents-<N>/fullchain_rounds.jsonl   # 全量 per-request
       -> fullchain/agents-<N>/fullchain_summary.json   # 聚合
  -> scripts/plot/plot_trajectory_metrics.py
  -> scripts/plot/plot_trajectory_latency.py
       # 仅用代表轨迹的 filtered rounds（trajectory-source/），不作 fullchain 切片
  -> scripts/plot/plot_agent_compare.py
       -> agent_latency_boxplots.svg
       -> ttft_fullchain_by_agents.svg   # 读全量 fullchain_rounds 再按 #Agents 聚合画图
```

`scripts/plot/plot_ttft_fullchain.py` 仍可用于手工对任意
`fullchain_rounds.jsonl` 画堆叠图，但正式 collect 不再为单轨迹调用它。

## 8. 主要文件字段

### `rounds.jsonl`

每行对应一个 trajectory turn，包含：

- 标识：`trajectory_id`、`turn_id`、`request_id`；
- token：`append_tokens`、`prompt_tokens`、`target_generate_tokens`、
  `actual_generate_tokens`、`cached_tokens`；
- 端到端：`latency_seconds`、`ttft_seconds`、`ttst_seconds`、
  `tpot_seconds`、`itl_seconds`；
- streaming 检查：`streamed_token_count`、`batched_token_chunks`；
- JCT 闭合：`turn_wall_seconds`、`turn_client_runtime_seconds`、
  `turn_accounting_error_seconds`。

### `summary.json`

- batch JCT：`wall_seconds`；
- 完成数与错误：`trajectories_*`、`turns_completed`、`errors`；
- latency 分位数：`request_latency_seconds`、`ttft_seconds`、
  `ttst_seconds`、`tpot_seconds`、`itl_seconds`。

### `runs.csv`

每行对应一个 repetition，包含 JCT、客户端 latency mean、request/L3/P→D
timing mean/coverage、cache hit mean、prompt/cached token 范围。

### `aggregate.csv`

每行对应一个 MaxLen/agents 配置，对三个 repetition 的 run-level metric 输出
median/min/max。

## 9. 有效性检查

硬失败条件：

- trajectory 数、turn 数或目标 generation 长度不完整；
- client 收到空流、错误响应或合并 token chunk；
- request timing coverage < 95%；
- P→D timing coverage < 95%；
- no-L3 profile 出现 L3 timing；
- full-chain TTFT 或 request accounting 不闭合；
- 单 trajectory JCT 与各 turn 之和不闭合。

只报告、不强制失败：

- 启用 L3 后的 L3 timing coverage；
- cache hit rate；
- Prefill/Decode 吞吐与有效带宽波动；
- 动态 rollout 早期未命中。

尚未接入当前自动汇总链的硬件指标：

- IB 端口实际 wire bytes、拥塞和错误计数；
- Lustre 客户端实际 read/write bytes、IOPS 和平均请求大小；
- GPU utilization、HBM、功耗；
- CPU、NUMA、DRAM 带宽；
- 每个 Decode 的 unfinished tokens 和 storage queue depth。

这些指标不能从逻辑 timing 推算；未采集时必须标为 unavailable，不能填零。

## 10. 历史结果索引

- Job `65032`：[四轮 latency 诊断结果](latency-results.md)。
- Job `65144`：[TTFT full-chain 最小测例](ttft-fullchain.md)。
- Job `65502`：[完整轨迹 KV/带宽历史结果](trajectory-metrics.md)。
- Job `65548/65557`：[Mooncake Store 与 DFS smoke](mooncake-store.md)。

历史页只保留具体配置、数值和特殊现象；所有 metric 定义以本文为准。
