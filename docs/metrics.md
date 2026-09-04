# 指标、采集与产物规范

> 返回 [文档索引](README.md)。本文是本项目所有 metric 定义的唯一来源。
> workload 和运行方法见 [Benchmark 规范](benchmark.md)，具体 Job 数值见各结果页。

## 1. 口径分层

阅读结果时先区分以下三类：

- **正式性能口径**：Mooncake Store 动态完整 rollout；不预 Populate；Prefill
  在每个 request 的 PD ACK 后释放 radix 引用并立即驱逐所有未被活跃请求引用的
  L1/L2 page；不使用全局 `/flush_cache`。短 `STORAGE_SETTLE_SECONDS` 仅用于
  Decode Store 可见性，不是旧 file-backend 的长静默等待。
- **当前可自动汇总**：客户端 streaming 指标、request/L3/P→D timing、cache hit、
  聚合 CSV/JSON 和 SVG。
- **历史/对照口径**：Job `65032/65144/65502` 旧 file backend / 预热隔离；Job
  `65694` 未 flush、L1/L2 命中主导且 `l3_timing_coverage=0`。均不得与正式
  L3 路径混报。

正式 32K × 32/64/128 agents × 3 rep 完整 rollout 属于第一类。Job `65566`
因 Decode-1 节点 `dgx-gaia-53` 失联失败；Job `65592` 仅排除 `53` 后又分到
含历史故障节点 `dgx-gaia-30` 的组，已取消。当前正式作业排除
`dgx-gaia-18,dgx-gaia-30,dgx-gaia-46,dgx-gaia-53`（见 [平台说明](platform.md)）。完成后检查两类 **results**（名称相同、语义不同，不可混用）：

```text
# Lustre runtime results：原始 rounds / trajectories / run.json（RESULT_DIR）
/mnt/lustre/users/sezhou/dual-path/results/latency-benchmark/<TASK>/
/mnt/lustre/users/sezhou/dual-path/logs/latency-benchmark/<TASK>/

# Labhome 仓库侧产物：整理后的 data/csv/SVG（REPO_RESULT_DIR）
/labhome/sezhou/dual-path/results/latency-benchmark/<TASK>/
```

task 下按 `<DATASET>/agents-<N>/rep-<R>/` 分组；例如
`figure7-main/32K/agents-200/rep-1/`。

## 2. 端到端客户端指标

所有 token latency 都由 replay client 的同一个 `time.perf_counter()` 时钟测量，
不存在跨节点时钟相减。

### 口径速查（常见混淆）

| 名称 | 含义 | 不是 |
|---|---|---|
| **trajectory_jct** | 单条 trajectory 的完成时间：`elapsed_seconds` = Σ `turn_wall_seconds`。`all-trajectories/latency.svg` 的 JCT 面板展示全 agent 分布 | 不是 batch JCT 的「求和」；batch 用 **max** |
| **Per-agent mean TTFT** | **两步**：① 每个 agent 内对多 turn 的 `ttft_seconds` 先平均 → ② 再在 agent 之间画箱线/取分布 | 不是先把所有 turn 混在一起再平均；也不是只画代表轨迹 |
| **Per-agent mean TPOT** | 与 TTFT 相同的两步聚合 | 同上 |
| **longest-trajectory/** | `collect` 选出的 **JCT 最长 trajectory** 的逐 turn latency/KV/L3/P→D 图 | **不是**多 agent 平均；不再重复输出单轨迹 JCT 图 |
| **fullchain_rounds.jsonl** | 当前 run 的**全量** per-request：每行一个 `(trajectory_id, turn_id)` | 不是 longest trajectory 切片；均值 breakdown 在 `all-trajectories/ttft-breakdown.svg` |

### JCT

- Batch JCT：`summary.json: wall_seconds`。
- 起点：所有 worker 已创建后，barrier 释放。
- 终点：最后一条 trajectory 完成或失败。
- 不包含模型/Router/Mooncake master 启动和 Job 退出清理。
- 多 agent 的 batch JCT 是最晚完成 agent 的 critical path，不能把各 agent
  时间相加。
- **trajectory_jct**（正式命名，勿再称 jct_breakdown）：
  `trajectories.jsonl: elapsed_seconds`，等于该 trajectory 所有
  `turn_wall_seconds` 之和。
  - 最长轨迹的逐 turn 图：`longest-trajectory/turn.svg`；它已经展示每轮 wall
    time，因此不再重复输出单轨迹 JCT 图。
  - 跨 agent 箱线：`all-trajectories/latency.svg` 中的 JCT 面板；
    每个点是一个 agent 的 `elapsed_seconds`，**不做** turn 内再平均。
  - task 根目录 `JCT.svg`：每个 `(dataset, #Agents, rep)` 先取所有 agent 的
    trajectory JCT 最大值，再对同一 `(dataset, #Agents)` 的 repetitions 取最大；
    每个 dataset 一个面板、横轴为 agents、柱高为上述最坏 JCT；不再生成单独的
    `trajectory_jct_max_by_agents.svg`。

单轮闭合为：

```text
turn_wall_seconds
  = latency_seconds
  + turn_client_runtime_seconds
  + turn_accounting_error_seconds
```

正式路径中的 accounting error 应只剩浮点误差。`storage_settle_seconds` 按轮
记录但**不进入** `turn_wall_seconds` 闭合式。正式路径不调用全局 flush，因此
`cache_flush_seconds=0`、`cache_sync_wait_seconds=0`；两字段仅保留给显式开启
`FLUSH_LOCAL_CACHE_BETWEEN_TURNS=1` 的历史/控制实验。

`DUALPATH_PREFILL_CACHE_RELEASE` 在每个成功 PD ACK 后记录释放前后的 device
evictable tokens。SGLang 已先通过 `release_kv_cache()` 解除当前 request 的
radix lock；随后驱逐只处理 `lock_ref=0` 的叶节点，因此不会释放并发请求仍在
使用的共享 prefix。Prefill 使用 `write_through_selective`，Decode 继续负责把
完整上下文发布到 L3。

自动验收要求 `prefill_cache_release_coverage >= 0.95`，且
`prefill_evictable_after_release_max == 0`；对应字段写入 task
`data/validation.json` 和聚合 CSV。

长上下文并发 PD 传输显式设置 `MC_TRANSFER_TIMEOUT=120` 秒，并使用
`SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES=128` 将过大的 all-layer 同步
transfer 按 KV page index 有序分批；两项均写入 `run.json`。Mooncake 默认超时
为 30 秒；达到阈值不仅会令当前 request abort，还会把对应 RDMA session 标记为
failed，后续 request 无法通过 client 重试恢复。分批限制参考并 backport
SGLang upstream #32758 的修复；上游 1024 是针对其他模型的保守值，当前
Qwen2.5-32B TP8 实测在 agents=8 时仍会触发 session timeout，因此按
`CHUNKED_PREFILL_SIZE=8192` / `page_size=64` 收紧为 128。

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

这与下面的 **per-agent mean** 不同：配置级 mean 是「该 run 全部 turn 的均值」；
箱线图的 per-agent mean 是「先按 agent 再平均」。

### Per-agent mean TTFT / TPOT（箱线图）

`all-trajectories/latency.svg` 中的 **Per-agent mean TTFT**：

1. 按 `trajectory_id`（= 一个 agent）分组，收集该 agent 所有 turn 的
   `ttft_seconds`；
2. **agent 内**对上述 turn 取算术均值（`statistics.fmean`），得到每个 agent
   一个标量 `agent_mean_ttft_seconds`；
3. 箱线图在同一 `#Agents` 并发下展示这些 **agent 级均值** 的分布
   （whiskers=min/max，box=Q1–Q3，线=median，蓝点=跨 agent 的 mean）。

**Per-agent mean TPOT** 同理（只对非 null 的 `tpot_seconds`）。旁侧
**trajectory_jct** 面板直接用每个 agent 的 `elapsed_seconds`，不做 turn 内
再平均。

### 代表轨迹（trajectory-* 图）

`longest-trajectory/` **不是**多 agent 平均，而是
`scripts/benchmark/collect_benchmark_metrics.py` 选出的 **一条最长轨迹**：

1. 读该 run 的 `trajectories.jsonl`；
2. 取 `elapsed_seconds`（trajectory JCT）最大的 `trajectory_id`，再以
   `turns_completed` 和 id 稳定打破并列；
3. 过滤该 id 的 rounds / trajectory 行，写入 `data/longest-trajectory/`，再调用
   `plot_trajectory_latency.py` / `plot_trajectory_metrics.py`。

正式路径**不**为最长轨迹生成 fullchain 切片；full-chain 保留当前 run 的全量
request 明细（见 §6）。

## 3. Cache hit 与 KV 数据量

### Prefix hit

- `cached_tokens`：SGLang response `meta_info.cached_tokens`。
- `cache_hit_rate = cached_tokens / prompt_tokens`。
- 它表示 HiCache 报告的可复用前缀；正式路径在 PD ACK 后驱逐未锁定的 Prefill
  L1/L2 page，因此后续 turn 的命中应主要来自 L3 prefetch，并与
  `DUALPATH_L3_TIMING` / `l3_hit_rate` 对照。
- 若关闭 `DUALPATH_PREFILL_RELEASE_LOCAL_CACHE`，命中可来自 L1/L2，不能直接
  归因于 Lustre DFS。

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

正式 replay 在请求完成并等待 `STORAGE_SETTLE_SECONDS` 后递归采样 Mooncake
DFS shard；采样与 settle 的耗时均不计入 `turn_wall_seconds` / trajectory JCT。

- `l3_apparent_snapshot_gib`：文件 `st_size` 总和。Mooncake 预先创建稀疏 shard，
  因而它可能从首轮起固定为预留容量（Job `66097` 为 256 GiB），**不能**视为
  已写入数据量。
- `l3_physical_snapshot_gib`：`st_blocks × 512` 总和，表示 Lustre 当时已分配
  block，是当前落盘曲线使用的物理快照。写入异步，短 settle 后仍可能滞后或
  批量跳升，不要求逐 turn 与逻辑 KV 相等。
- Job `66097`（1P2D、32K、1 agent）最终物理快照为 6.932 GiB，证明此前的 0
  来自缺少采样，而非 Mooncake 未落盘。

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
- `DUALPATH_TTFT_P_TRANSFER_V2`（解析器兼容旧
  `DUALPATH_TTFT_P_TRANSFER`）
  - 来源：Prefill disaggregation sender。
  - 窗口：第一个实际 KV chunk 提交到最终 `KVPoll.Success`。
  - 字段：start/end/duration、logical tokens、page count、page size 和
    page-aligned tokens。
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
- `pd_timing_coverage`：有 P→D transfer marker 的请求数 / 总请求数。

正式矩阵要求 request 和 P→D coverage 至少 95%；关闭 Prefill L3 时要求 L3
coverage 严格为零。启用 L3 时不设置 coverage 下限。

### TP=8 L3 聚合

同一 request 必须取得一个 Prefill engine 的 8 个 TP rank timing。请求的
`read_seconds` 使用 `max(read_end) - min(read_start)`，覆盖八个 rank 的联合
读取窗口；`l3_slowest_tp_rank` 另记录 duration 最大的瓶颈 rank。不能只用
rank 0 或单纯 `max(rank duration)` 计算节点聚合加载带宽。

### P→D timing 与有效带宽

```text
pd_effective_bandwidth_gbps
  = (page_count × page_size) × 262144 / pd_transfer_seconds / 1e9
```

分子是该 request 实际提交的 page-aligned KV payload（不是简单拿 prompt
tokens，也不是多个 Decode engine 的平均）；分母是 Prefill sender 从首个 chunk
提交到最终 `KVPoll.Success` 的完整窗口。它是单 request 的逻辑有效带宽，不是
IB wire bandwidth；曲线逐 request/turn 作图，summary 报其中位数。Job
`65144/65502` 的旧镜像从最后一个 chunk 才开始计时，数值不可与新结果直接比较。
Job `66097` 的 26 个 request 覆盖率 100%，中位数为 9.483 GB/s。

### Prefill L3 有效加载带宽

```text
prefill_l3_effective_bandwidth_gbps
  = l3_logical_tokens × 262144
    / (max(tp_rank_read_end) - min(tp_rank_read_start)) / 1e9
```

这里分子是一个 Prefill engine 的 TP=8 合计逻辑 payload，分母是该 request
八个 rank 的联合读取窗口，所以结果是**单 Prefill engine 的 TP 聚合有效
带宽**，不是每 rank 平均，也不是跨 engine 平均。Job `66097` 有 L3 read 的 6
个 requests 的旧图使用 `max(rank duration)`，需按新联合窗口重新 collect 后
再引用；该逻辑有效带宽也不能直接解释为 Lustre NIC wire bandwidth。带宽统一
使用十进制 GB/s；KV 容量使用二进制 GiB。

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

- `<DATASET>/agents-<N>/rep-<R>/data/fullchain/fullchain_rounds.jsonl`：
  当前 run 的**全量 per-request** 明细。
  每一行对应一次 `(trajectory_id, turn_id)` / request，含各互斥桶秒数与
  `ttft_seconds`；不是配置级均值。
- 同目录 `fullchain_summary.json`：在上述明细上的**聚合统计**
  （mean 等）。
- `all-trajectories/ttft-breakdown.svg`：当前 run 全部 requests 的互斥桶均值
  堆叠图。

#### 与论文 Figure 10/12 的对照（勿混）

| | 本仓库 `all-trajectories/ttft-breakdown.svg` | 论文 Fig.10 / Fig.12(left) |
|---|---|---|
| 场景 | **离线** barrier 同步 batch（当前矩阵） | **在线** Poisson 到达 |
| 横轴 | `#Agents`（并发 agent 数） | **APS**（agents/s 到达率） |
| 柱上 TTFT 是哪一轮？ | **不是某一 turn**：对该 concurrency 下
  `fullchain_rounds.jsonl` 里 **全部 request**（所有 agent × 所有 turn）
  的 `ttft_seconds` 与各桶做算术均值（请求等权） | 在线流上的请求 TTFT 聚合；Fig.10
  阴影为结束前 **150s** 窗口波动；稳态判据也是 150s 窗相对 30min 前
  TTFT 变化 <5%。每个到达的 agent 从 round 0 跑到最后一轮，因此统计
  **覆盖多 turn 请求流**，不是「只取第 1 turn」 |
| 是否「agent 内先平均再跨 agent」？ | **否**（与箱线图 per-agent mean TTFT 不同） | 文中未写该两步；报告的是服务侧
  TTFT 时间序列/窗口均值 |

因此：随 APS 变化的论文 TTFT **是多 turn（多 request）上的在线聚合**，不是单
turn 特写；本仓库 breakdown 图是 **离线全量 request 均值 × #Agents**，横轴也
不是 APS。当前 `benchmark.md` 明确尚未做 APS sweep。

正式路径不输出单轨迹 fullchain 切片。
`scripts/benchmark/collect_benchmark_metrics.py` 对每个
dataset/agents/rep 分别生成该 run 的全量 fullchain。

## 7. 当前自动采集流水线

正式提交入口（每种拓扑仅保留一份）：

```text
scripts/pd_1p1d_benchmark.sbatch
scripts/pd_1p2d_benchmark.sbatch
scripts/pd_2p2d_benchmark.sbatch
```

Job 结束时统一调用 `scripts/benchmark/collect_benchmark_metrics.py`。

三个入口统一接受 `TASK_NAME`、`DATASETS`、`AGENT_COUNTS`、`REPETITIONS` /
`REP_LIST`。新运行的目录固定为：

```text
<TASK>/<DATASET>/agents-<N>/rep-<R>/
  data/                  # run/summary/rounds/trajectories/fullchain JSON(L)
  csv/                   # longest trajectory 与 all trajectories 表格
  longest-trajectory/    # 最长 JCT agent 的逐 turn 图
  all-trajectories/      # 全 agent latency box plot 与平均 TTFT breakdown
```

`longest-trajectory/` 的文件名保持简短：

- latency：`turn.svg`、`ttft.svg`、`ttst.svg`、`tpot.svg`；JCT 与 turn 图信息
  重复，不单独画；
- KV/cache：`kv-new.svg`、`kv-total.svg`、`cache-hit.svg`；
- L3 read：`l3-read-data.svg` 汇总最长轨迹每个 turn 的 TP 聚合逻辑读取量，不在
  文件名中使用有歧义的 engine 0；`l3-read-time-prefill-<ID>.svg` 才按 Prefill
  replica 区分并展示其 TP ranks。1P* 只有 replica 0；2P2D 的 replica 0/1 共享
  Mooncake DFS namespace，但一个 request 只由 Router 选中的 replica 发起读取；
- bandwidth：`l3-bandwidth.svg`、`pd-bandwidth.svg`。

`all-trajectories/latency.svg` 同时展示每个 agent 的 trajectory JCT、agent 内
多 turn mean TTFT、mean TPOT 三组箱线统计，并明确标注 min/max；
`ttft-breakdown.svg` 对该 run 全部 requests 的 full-chain 互斥桶取均值。

客户端：

```text
scripts/benchmark/replay_fake_agent_traces.py
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
scripts/benchmark/collect_benchmark_metrics.py
  -> scripts/benchmark/summarize_latency_benchmark.py
       -> <TASK>/data/{merged_rounds.jsonl,validation.json}
       -> <TASK>/csv/{runs.csv,aggregate.csv}
  -> scripts/benchmark/summarize_ttft_fullchain.py
       -> <run>/data/fullchain/fullchain_rounds.jsonl   # 当前 run 全量 request
       -> <run>/data/fullchain/fullchain_summary.json
  -> scripts/plot/plot_trajectory_metrics.py
  -> scripts/plot/plot_trajectory_latency.py
       -> <run>/longest-trajectory/{turn,ttft,ttst,tpot}.svg
       -> <run>/longest-trajectory/{kv-new,kv-total,cache-hit}.svg
       -> <run>/longest-trajectory/l3-read-data.svg
       -> <run>/longest-trajectory/l3-read-time-prefill-<ID>.svg
       -> <run>/longest-trajectory/{l3-bandwidth,pd-bandwidth}.svg
  -> scripts/plot/plot_run_summary.py
       -> <run>/all-trajectories/latency.svg
       -> <run>/all-trajectories/ttft-breakdown.svg
  -> scripts/plot/plot_task_jct.py
       -> <TASK>/JCT.svg / JCT.json
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
