# 指标、采集与产物规范

> 返回 [文档索引](README.md)。本文是本项目所有 metric 定义的唯一来源。
> workload 和运行方法见 [Benchmark 规范](benchmark.md)，具体 Job 数值见各结果页。

> **正式 TTFT breakdown 口径为 [Timing V3](#6-ttft-breakdown)。**
> `DUALPATH_TTFT_V3` JSON 是新结果的唯一阶段数据源；下文旧 marker 定义只用于
> 历史结果 fallback，不得与 V3 分桶混合。

## 1. 口径分层

阅读结果时先区分以下三类：

- **正式性能口径**：共享 Lustre file-backend 动态完整 rollout；不预 Populate；Prefill
  在每个 request 的 PD ACK 后释放 radix 引用并立即驱逐所有未被活跃请求引用的
  L1/L2 page；不使用全局 `/flush_cache`。同一 trajectory 的下一 turn 等待
  上一 turn 全部 Decode TP rank 的 L3-complete marker。
- **当前可自动汇总**：客户端 streaming 指标、request/L3/P→D timing、Decode
  Host→GPU timing、cache hit、聚合 CSV/JSON 和 SVG。
- **历史/对照口径**：Job `65032/65144/65502` 旧 file backend / 预热隔离；Job
  `65694` 未 flush、L1/L2 命中主导且 `l3_timing_coverage=0`。均不得与正式
  L3 路径混报。

正式 32K × 32/64/128 agents × 3 rep 完整 rollout 属于第一类。Job `65566`
因 Decode-1 节点 `dgx-gaia-53` 失联失败；Job `65592` 仅排除 `53` 后又分到
含历史故障节点 `dgx-gaia-30` 的组，已取消。当前正式作业排除
`dgx-gaia-12,dgx-gaia-18,dgx-gaia-30,dgx-gaia-46,dgx-gaia-53`（见 [平台说明](platform.md)）。完成后检查两类 **results**（名称相同、语义不同，不可混用）：

```text
# Lustre runtime results：原始 rounds / trajectories / run.json（RESULT_DIR）
/mnt/lustre/users/sezhou/dual-path/test/<allocation-id>/<TASK>/logs/

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

#### wall_seconds 比 trajectory_jct 大一个 sync wait

两者不可互换，差值不是噪声。`wait_for_decode_l3()` 在 `turn_ended` 打点之后
执行，所以 `decode_l3_sync_wait_seconds` 不进入 `turn_wall_seconds`，也就不进
入由它累加而成的 `trajectory_jct`；但它是真实墙上时间，`wall_seconds` 包含它。
因此：

```text
wall_seconds ≈ max_over_agents(elapsed_seconds + Σ decode_l3_sync_wait_seconds)
```

Job `69341` 的 1P1D TP8、32K、4 agent、O_DIRECT 三个 run 验证该式，残差 0.02 s，
即真实收尾开销可忽略（实测序列化并写 609 KiB `rounds.jsonl` 到 Lustre 仅 0.03 s）：

| run | max trajectory_jct | + 该 agent sync wait | wall_seconds | 残差 |
|---|---:|---:|---:|---:|
| Prefill path | 96.726 | 99.586 | 99.607 | 0.021 |
| Decode path | 100.851 | 103.770 | 103.788 | 0.017 |
| Dynamic path | 86.683 | 89.956 | 89.975 | 0.019 |

每 agent 的 sync wait 累计约 2.9--3.7 s（26 turn，每轮约 0.12 s）。两点后果：

- 正式对比一律用 **trajectory_jct**，不要混用 `wall_seconds`，两者相差约 3 s。
- JCT 最大的 agent 未必是 wall 最大的那个。上表 Dynamic run 中 JCT 最大的是
  `32k-0001`（86.683），加 sync wait 后最大的却是 `32k-0002`（89.956）。

该口径系统性地把 Decode L3 sync 等待排除在 JCT 之外。保持现状是为了与既有
JCT 数据可比；若要改为计入，只需把 `turn_ended` 移到 `wait_for_decode_l3()`
之后，但此后新旧 JCT 不能直接比较。

单轮闭合为：

```text
turn_wall_seconds
  = latency_seconds
  + turn_client_runtime_seconds
  + turn_accounting_error_seconds
```

正式路径中的 accounting error 应只剩浮点误差。
`decode_l3_sync_wait_seconds` 按轮记录、但**不进入** `turn_wall_seconds`
闭合式；`storage_settle_seconds=0`。正式路径不调用全局 flush，因此
`cache_flush_seconds=0`、`cache_sync_wait_seconds=0`；两字段仅保留给显式开启
`FLUSH_LOCAL_CACHE_BETWEEN_TURNS=1` 的历史/控制实验。

`DUALPATH_PREFILL_CACHE_RELEASE` 在每个成功 PD ACK 后记录释放前后的 device
evictable tokens。SGLang 已先通过 `release_kv_cache()` 解除当前 request 的
radix lock；随后驱逐只处理 `lock_ref=0` 的叶节点，因此不会释放并发请求仍在
使用的共享 prefix。Prefill 设置为 L3 read-only，不产生 write-through/backup，
所以 PD ACK 后无需等待存储写入即可立即驱逐 L1/L2。Decode 根据 Prefill metadata
中的 `cached_tokens`，只发布该 page-aligned L3 前缀之后的新 KV。

自动验收要求 `prefill_cache_release_coverage >= 0.95`、
`prefill_evictable_after_release_max == 0`，并在 deferred eviction 开启时要求
`prefill_l3_cached_token_coverage >= 0.99`。最后一个指标是所有存在 cached prefix
的轮次中 `min(L3 read tokens, cached tokens)` 与 `cached tokens` 的加权比值，
用于发现 PD ACK 时仍被 HiCache 临时锁住、稍后解锁却留在 L1/L2 的 page。
`evictable_before` 仅作为诊断：
短请求或已由请求生命周期提前释放的 page 可以为 0。对应字段写入 task
`data/validation.json` 和聚合 CSV。启用 Prefill L3 与 Decode offload 的多轮
run 还要求观测到跨 turn L3 timing，防止“本地释放成功但 L3 未发布”的假阳性。
Job `66987` 中 turn 1 由 Decode 从 `offload_start=0` 发布 1856 tokens；turn 2
的 Prefill 从 L3 读取 1856 tokens，随后 Decode 从 `offload_start=1856` 发布
1984 个新 tokens，是当前机制的通过样例。

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

SGLang 固定 `stream_interval=1`，但 HTTP/SSE 仍可能把多个 token 合并到一个
payload。此时这些 token 共享同一个 client 到达时间，并通过
`batched_token_chunks > 0` 明确标记。只要 `streamed_token_count` 与实际生成
token 数一致，run 仍然有效：TTFT、TTST 和端到端 TPOT 可用；逐 token ITL 的
时间分辨率受 SSE 合并影响，会包含零值，不应用于分析 server 内部逐步 decode
间隔。只有 streaming token 总数不一致时，汇总器才判定 run 无效。

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
  归因于 Lustre file backend。

### L3 read hit

- `l3_prefetched_tokens_logical`：`DUALPATH_L3_TIMING` 中各 TP rank 报告的逻辑
  token 数。
- `l3_hit_rate = l3_prefetched_tokens_logical / prompt_tokens`。
- 只有产生 L3 prefetch marker 的请求才是可观测 L3 read。
- 下一 turn 在 request-scoped Decode L3 completion 屏障后启动；若仍无 L3
  marker，应视为 file backend 命中或 timing 采集问题。

### Qwen2.5-32B BF16 KV 大小

```text
64 layers × 8 KV heads × 128 head_dim × K/V × 2 bytes
  = 262,144 bytes/token
  = 256 KiB/token
```

- 新增逻辑 KV：`(append_tokens + generated_tokens) × 262144`。
- turn 后逻辑 KV：`(prompt_tokens + generated_tokens) × 262144`。
- 容量图使用 GiB，即除以 `1024³`。

正式 replay 不再递归扫描 Lustre file-backend 目录，也不再生成逐轮 apparent
bytes、allocated blocks 或 file-count snapshot。该扫描会随文件数量和并发 agent
数快速增长，并且 `st_blocks` 受文件系统分配粒度影响，不能作为请求逻辑 KV 的
准确替代。L3 数据量和带宽只使用服务端 request-scoped read timing marker；
Decode 持久化完成仍由 8-rank completion marker 精确同步。

Job `67478`（2026-09-07，1P2D、32K、1 agent、26 turns）验证删除后完整
replay 成功：26/26 turns、0 errors，request/P→D timing coverage 均为 100%，
Prefill L3 cached-token coverage 为 100%，总作业耗时 5:44；生成的 run metadata、
rounds、CSV、summary 和图均不再包含 snapshot/allocated-block 字段。

- Job `66097` 是历史 Mooncake DFS 结果，不与当前 file-backend 容量混报。

## 4. 服务端 timing marker

### Timing V3 正式 marker

`DUALPATH_TTFT_TIMING=1` 时，每个 request、每个 Prefill/Decode TP rank 各输出
一条 `DUALPATH_TTFT_V3 {json}`。默认 `DUALPATH_TTFT_TIMING_DETAIL=0`，不会输出
下面的逐阶段 legacy marker。V3 字段、路径时间线、raw/critical/overlap 语义和
TP=8 聚合规则见 [TTFT Timing V3](#ttft-timing-v3-实现细节)。

以下 marker 均为 legacy/detail 口径。

timing 镜像由 `scripts/infra/patch_sglang_benchmark_timing.py` 注入以下
request-scoped 日志，以 request ID 与客户端 `rounds.jsonl` 关联。

- `DUALPATH_REQ_TIMING`
  - 来源：Prefill scheduler。
  - 字段：queue、allocation、prefill/worker launch、forward duration 及关键
    时间戳。
  - 用途：request timing coverage、旧 proxy breakdown、full-chain Prefill 段。
- `DUALPATH_L3_TIMING`
  - 来源：Prefill HiRadix/HiCache storage prefetch。
  - 字段：Prefill engine、node、TP rank、read start/end/duration、tokens。
  - `read_end` 在该 rank 首次观察到本地 storage IO 完成时、TP 完成共识
    `all_reduce` 之前记录，因此每个 rank 的 duration 不包含等待慢 rank 的时间。
  - 用途：Prefill storage prefetch 的端到端总窗口，包括 worker 排队和
    scheduler 观察完成延迟；不再作为 Lustre 读取带宽分母。
- `DUALPATH_PREFILL_KV_READ_BREAKDOWN`
  - 来源：Prefill cache-controller storage worker。
  - `lustre_read_s` 只累计 `storage_backend.get()` / `batch_get()`；
    `scatter_s` 只累计 `set_from_flat_data_page()` 写入 HiCache Host pool。
  - 同一 request/rank 可能有多个 storage batch，汇总时按 rank 求和。
- `DUALPATH_DECODE_L3_READ`
  - 来源：被 Router 选中的 Decode engine，在反向传输前将 file-backend page
    直接读入独立 PD Host staging。
  - 字段：Decode engine、node、TP rank、read start/end/duration、
    `lustre_read_s`、`scatter_s`、tokens。
  - 用途：Decode 侧 L3 数据量、时间和有效带宽；每个 request 必须只有一个
    Decode engine，并有 8/8 TP rank timing。
- `DUALPATH_D2P_REVERSE`
  - 来源：Prefill bootstrap scheduler。
  - 窗口：Prefill 发起 reverse-fetch 控制消息，到该 TP rank 收到所需 Decode
    session 的全部 completion。
  - 字段：request ID、Prefill engine、node、TP rank、start/end/duration、tokens。
  - TP=8 使用 `min(rank start)` 到 `max(rank end)` 的同节点 monotonic-clock union，
    表示 Decode Host staging → Prefill HBM 的端到端关键路径，而不是仅统计 Decode
    侧 `_send_kvcache_generic()` 调用。
- `DUALPATH_PREFILL_PREP_BREAKDOWN`
  - 来源：Prefill scheduler 构造 EXTEND batch。
  - `batch_init_end → hicache_end` 对应
    `ready_to_load_host_cache()` / `cache_controller.start_loading()`，即 Prefill
    HiCache prefix 的 Host→GPU restore launch 窗口。
  - `hicache_consumer >= 0` 表示该 batch 确实创建了 HiCache load producer；空
    load queue 返回 `-1`，不得把调用开销误算为 Host→GPU。
  - TP=8 使用同节点 `min(rank start)` 到 `max(rank end)` 的 union；仅在
    `cached > 0` 且 producer 有效时计入 TTFT。该窗口不额外同步 CUDA finish
    event，后续逐层完成可与 Prefill forward 重叠，因此不会为了计时改变原执行
    行为。
- `DUALPATH_DECODE_HANDOFF_BREAKDOWN`
  - 来源：被选中的 Decode engine bootstrap receiver。
  - `prealloc_start → prealloc_end` 是 `_pre_alloc()` 的 Host wall-clock 窗口，包含
    request slot、paged HBM KV slots 和 request-to-token mapping 分配。
  - TP=8 使用同一 Decode node 上的 union；raw duration 始终保留为 diagnostic，
    additive TTFT 只计入 Decode-path bootstrap envelope 中不与其他阶段重叠的部分。
- `DUALPATH_TTFT_P_TRANSFER_V2`（解析器兼容旧
  `DUALPATH_TTFT_P_TRANSFER`）
  - 来源：Prefill disaggregation sender。
  - 窗口：第一个实际 KV chunk 提交到最终 `KVPoll.Success`。
  - 字段：Prefill engine、node、start/end/duration、logical tokens、page count、
    page size 和 page-aligned tokens。
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
- `l3_timing_coverage`：有 Prefill 或 Decode L3 timing 的请求数 / 总请求数。
- `pd_timing_coverage`：有 P→D transfer marker 的请求数 / 总请求数。

正式矩阵要求 request 和 P→D coverage 至少 95%；关闭 Prefill L3 时要求 L3
coverage 严格为零。启用 L3 时不设置 coverage 下限。

### TP=8 L3 聚合

同一 request 必须取得一个 Prefill 或 Decode engine 的 8 个 TP rank timing。请求的
`read_seconds` 使用 TP union 总窗口；另保留
`tp_union_read_seconds = max(read_end) - min(read_start)` 作为 rank 启动偏斜
诊断。纯 Lustre 带宽使用总窗口最慢 TP rank 对应的 `lustre_read_s`，scatter
单独报告；不能只使用 rank 0。

逻辑读量使用 8 个 TP rank 报告的最小 `tokens`，避免把 TP=8 的同一份 KV
重复计算。汇总器同时保留 `node/engine` 归属，在
`kv-cache-monitor-scopes.json` 输出每个 Prefill/Decode node-engine 的 tokens、GiB
和 request 数，并在有数据时生成对应的 `l3-read-data-<side>-<node>-<engine>.svg`；
聚合图为 `kv-cache-monitor.svg`。

`kv-cache-monitor.svg` 的 P→D 曲线使用 Prefill sender 报告的逻辑 payload；在
Decode-read 分支应接近每轮新增 KV，在 Prefill-read 分支应接近每轮 total KV。
P→D 的 node/engine 分项与总量同样记录在 `kv-cache-monitor-scopes.json`。

### P→D timing 与有效带宽

正式路径不是 Prefill GPU 直接写 Decode GPU。每个 Decode TP rank 使用独立的
40 GB `layer_first` pinned Host staging pool，并把 Host 指针注册给 Mooncake
Transfer Engine。收到全部 Prefill completion ACK 后，Decode 在独立 CUDA
stream 上把 request pages 搬入 HBM，完成后归还 staging pages，才向 scheduler
返回 `KVPoll.Success`。后续新 KV 的 D2H→Lustre 使用另一套
`page_first_direct` HiCache Host pool；两套 pool 不共享 allocator。

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

正式 DGX H100 配置将 TP rank 0–7 一一绑定到八条 400 Gb/s compute-fabric
HCA；该值仍是应用层逻辑有效带宽，不能用 `8 × 400 Gb/s` 直接替代实测。

### Prefill L3 有效加载带宽

```text
prefill_l3_effective_bandwidth_gbps
  = l3_logical_tokens × 262144
    / slowest_tp_rank_lustre_read_s / 1e9
```

这里分子是一个 Prefill engine 的 TP=8 合计逻辑 payload，分母是总 Prefetch
窗口最慢 rank 在 `storage_backend.get()` 中累计的时间，所以结果是**单 Prefill engine 的 TP 聚合
有效带宽**，不是每 rank 平均，也不是跨 engine 平均。该逻辑有效带宽不能直接
解释为 Lustre NIC wire bandwidth。带宽统一使用十进制 GB/s；KV 容量使用
二进制 GiB。

## 6. TTFT breakdown

### 正式 V3 口径

Prefill path 的堆叠顺序为：Prefill before Lustre Read、Prefill KV Read
(Lustre)、Prefill Scatter、Prefill Host→GPU、Prefill Computation、P→D、
Decode Host→GPU、Decode dispatch、Misc。

Decode path 的堆叠顺序为：Decode pre-allocation、Decode before Lustre Read、
Decode KV Read (Lustre)、Decode Scatter、D→P、Prefill Computation、P→D、
Decode Host→GPU、Decode dispatch、Misc。

V3 不跨节点相减绝对时间戳。`misc_seconds = client TTFT - sum(critical stages)`；
闭合容差为 `max(2ms, 0.5% TTFT)`，超限写入 error flag。三个 TTFT 绘图入口
共同读取 `scripts/plot/ttft_components.py` 的 path-aware registry。

### Legacy full-chain（仅历史结果 fallback）

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
| Prefill Bootstrap & Scheduling | 两种 L3 路径统一为通用 Prefill 本地 bootstrap 与 queue/scheduling；不包含 Prefill L3 read、Decode L3 read、Decode-specific handoff 或 D→P reverse transfer |
| Prefill L3 KV Read | Prefill 侧 L3 prefetch |
| Decode L3 KV Read | Decode 侧从 Lustre 读取 durable prefix；TTFT 使用同一 Decode node/engine 上 `min(rank start)` 到 `max(rank end)` 的 TP union wall time，并限制为 Prefill bootstrap envelope 中可归属的关键路径时间；最慢单 rank duration 另存为带宽诊断 |
| Decode HBM Pre-allocation | Decode `_pre_alloc()` 的 TP union；从 Decode-path bootstrap envelope 中归因，raw duration 单独保留 |
| Decode L3 Bootstrap Handoff | Prefill bootstrap entry 到 D→P 开始之间，扣除 Decode L3 TP union 和 Decode HBM pre-allocation 后的 Decode-path-only 等待；包括其余 receiver/metadata handoff、Prefill prefix page 分配和 TP 共识 |
| Decode-to-Prefill KV Transfer | Decode Host staging → Prefill HBM；Prefill 时钟域端到端计时，包含 reverse-fetch 控制与 TP completion 等待 |
| Prefill HiCache Host-to-GPU | `start_loading()` 的 Host-observed prefix restore launch 窗口；不额外同步 CUDA finish，逐层完成与 forward 的重叠不重复计入 |
| Prefill Forward + Logits + Sampling | prefill compute + first-token sample |
| Prefill-to-Decode KV Transfer | P→D KV 传输窗口 |
| Decode Host-to-GPU Copy | Decode pinned Host staging → HBM；八个 TP rank 取最慢 duration |
| Decode Prebuilt-token Dispatch | KV ready → PREBUILT first-token dispatch |
| SSE Serialization & Egress | dispatch → 首个 SSE payload yield |
| Server Runtime Residual | `Decode HTTP handler_entry→yield - Σ直接观测服务器阶段`；表示服务器内部尚未细分或跨组件协调时间 |
| Router / Client Boundary | `client TTFT - Decode HTTP handler_entry→yield`；覆盖 Router、客户端和边界网络/序列化，仅作为边界残差，不是 Router 进程内部的直接计时 |

Prefill 侧重叠区间按 Prefill L3 read、D→P、scheduling、prefill、P→D 的优先级
只归属一次。Decode L3 与 Prefill 时钟不同，因此使用 Decode 同节点 TP union
duration，并从已排除 D→P 的 bootstrap envelope 余额中扣除；超出该余额的部分
仅保留为 diagnostic，避免跨节点并发或时钟不可比导致重复归因。Decode 分支在
D→P 开始前剩余的 path-specific wait 归入 `Decode L3 Bootstrap Handoff`，而不是
通用 `Bootstrap & Scheduling`。
Decode receive wait 与 P→D 窗口重叠，仅保留为诊断；P→D ACK 后的 Host→GPU copy
单独计入 additive breakdown。所有桶必须满足：

```text
TTFT = Σ full-chain stages - overlap_excess
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

### TTFT Timing V3 实现细节

> 口径；旧 marker 仅用于历史结果 fallback。

### 开关与日志

benchmark 默认设置：

```bash
DUALPATH_TTFT_TIMING=1
DUALPATH_TTFT_TIMING_DETAIL=0
```

总开关为 `0` 时，不创建 request timing state 或额外 CUDA timing event，也不输出
V3 JSON。detail 开关只用于人工诊断旧 marker，正式 benchmark 保持为 `0`。

每个 request 的 Prefill 和 Decode 各 TP rank 分别输出一次：

```text
DUALPATH_TTFT_V3 {json}
```

因此 TP=8 时正常为 8 条 Prefill 加 8 条 Decode。JSON 包含 schema version、
`rid/path/side/engine/node/tp_rank`、raw/critical/overlap 字段、本地诊断窗口、
tokens 和 pages。环境变量在模块导入/对象初始化时缓存，热路径不重复查询。

### 四种时间

- **Raw duration**：该操作自身观测到的完整持续时间，可能与其他阶段重叠。
- **Critical contribution**：按固定归属规则去重后，计入 TTFT 堆叠柱的时间。
- **Overlap removed**：`raw - critical`，用于解释流水重叠，不画成额外阶段。
- **Misc**：客户端 `TTFT - sum(critical stages)`。它是剩余端到端时间，不是某个
  单独 server 函数的计时。

CPU 操作用 `perf_counter_ns()` 累计。GPU H2D 和 Prefill computation 只在 timing
开启时各增加常数个 CUDA event，不增加逐层 event，也不新增 `cuda.synchronize()`；
event 只在现有完成/ACK 路径读取。

### Prefill Read Path

真实时间线和绘图顺序：

```text
Prefetch worker dequeue
  -> hash + batch_exists + TP hit consensus       [Prefill before Lustre Read]
  -> storage_backend.get/batch_get                [Prefill Lustre Read]
  -> page data -> layer_first Host pool           [Prefill Scatter]
  -> Host pool -> Prefill HBM                      [Prefill Host->GPU]
  -> EXTEND forward + first-token sampling        [Prefill Computation]
  -> first P->D send ... wait-send success         [P->D transfer]
  -> Decode staging Host -> HBM                    [Decode Host->GPU]
  -> PREBUILT ready -> rank-0 output dispatch      [Decode dispatch]
  -> client TTFT remainder                         [Misc]
```

H2D 和 computation 使用同 GPU timeline。重叠优先归给 H2D，computation 只保留
独占贡献。P→D raw 从第一次实际 send 到 Prefill wait-send 成功；chunked transfer
与 Prefill pipeline 重叠的部分从 P→D critical contribution 中扣除。

### Decode Read Path

真实时间线和绘图顺序：

```text
Decode _pre_alloc()                               [Decode pre-allocation]
  -> Decode Host staging allocation / metadata    [diagnostic, Misc]
  -> worker dequeue, hash, exists, TP consensus   [Decode before Lustre Read]
  -> storage_backend.get/batch_get                [Decode Lustre Read]
  -> page data -> Decode PD Host staging          [Decode Scatter]
  -> synchronous Mooncake reverse send + wait     [D->P transfer]
  -> Prefill computes Lustre-miss suffix           [Prefill Computation]
  -> Prefill sends suffix and waits for completion [P->D transfer]
  -> Decode PD Host staging -> HBM                 [Decode Host->GPU]
  -> PREBUILT ready -> rank-0 output dispatch      [Decode dispatch]
  -> client TTFT remainder                         [Misc]
```

`_pre_alloc()` 分配 request slot、HBM pages 和 req-to-token mapping，确实早于
Decode hash/read。独立 pinned Decode Host staging 的分配不属于 pre-allocation；
它作为 diagnostic 输出并留在 Misc。D→P 在 Decode 端用真实 `rid` 记录同步
Mooncake send 的完整窗口。Prefill 只计算 miss suffix，随后 P→D 只发送 suffix。

Decode 的 storage prefetch 生命周期由 `DecodePreallocQueue` 持有，结构与 Prefill
的 scheduler-owned prefetch 一致：scheduler 创建 `PrefetchOperation`，共享的
`HiCacheController.prefetch_thread_func` 完成 hash/exists，
`prefetch_io_aux_func` 完成 Lustre Read/Scatter，scheduler 按 FIFO 检查完成并做
TP consensus。`MooncakeKVReceiver` 不再轮询 storage operation，只接收 scheduler
已经确认的 `hit_tokens`、staging indices 和 timing snapshot。

Decode 当前一次只提交一个有序 storage prefetch operation。I/O 仍在后台线程异步
执行，但下一个 request 不会在前一个 request 的 TP query/finish consensus 完成前
进入共享 query stream。取消操作不阻塞 scheduler；worker 发布 `io_complete` 后才
释放独立 Decode staging，避免 I/O 或 RDMA 尚在引用时复用 buffer。

### TP=8 聚合

- read pipeline 选择 `before + read + scatter` raw 总和最大的 TP rank，并从同一
  rank 取三个分量，禁止分别挑 rank。
- Prefill H2D/computation 选择两者 critical contribution 总和最大的同一 rank。
- Decode pre-allocation、D→P、P→D、Decode H2D 各取最慢 TP rank。
- Decode dispatch 只采用负责输出的 rank 0。
- 不比较不同节点 `perf_counter` 的绝对时间戳，只合并 duration。

每轮同时输出 `*_raw_seconds`、`*_seconds` 和
`*_overlap_removed_seconds`。若闭合误差或 critical sum 超出 TTFT 的量超过
`max(2ms, 0.5% TTFT)`，汇总器设置 `accounting_error_exceeds_tolerance=true`；
当前镜像交付阶段只标记，不自动执行集群验证。

### Misc 中仍未拆分的开销

- Prefetch 和 I/O auxiliary queue wait。
- Prefill radix insertion、HBM page allocation。
- Decode Host staging allocation、page-index 和 metadata 准备。
- I/O 完成后的 scheduler polling/final consensus。
- Admission/backpressure、bootstrap 和控制消息。
- Router、tokenizer、detokenizer、SSE 和客户端网络。
- Prefill path 中未单列的 Decode pre-allocation。

V3 JSON 保留 pre-read 本地窗口和 Decode Host allocation diagnostic，便于用户
验证后决定是否继续拆分。旧 `DUALPATH_L3_TIMING`、read breakdown、handoff、
receive 和 HTTP marker 仅在 detail 模式输出；parser 对旧镜像结果保留 fallback。

## 7. 当前独立采集流水线

正式运行入口（所有拓扑共用一份）：

```text
scripts/pd_benchmark.sh
scripts/tests/runtime/pd_runtime.sh
```

`scripts/pd_alloc.sh P D` 负责建立 allocation、节点角色、镜像和长期 server 静态配置；
dataset、agents、repetition、task/log/result 路径由每次 client invocation 配置。

运行与绘图彼此独立。客户端角色只生成 Lustre 原始结果；某个 client run 完成后即可通过
`scripts/pd_plot.sh` 调用
`scripts/benchmark/collect_benchmark_metrics.py`：

```bash
scripts/pd_plot.sh <TASK_NAME>
```

allocation 配置提供 `PREFILL_HOSTS`、`DECODE_HOSTS` 和固定 L3 read path；client
接受 `TASK_NAME`、`DATASET`、`AGENTS`、`REPETITION`。P/D 数量由 host list 推导；
TASK 目录不自动追加 Job ID，每个 client run 必须使用唯一名称：

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
- L3 read：`l3-read-data.svg` 在每个 turn 同时画 Prefill 与 Decode TP 聚合逻辑
  读取量，未读取的一侧为 0；`l3-read-time-prefill-<ID>.svg` 和
  `l3-read-time-decode-<ID>.svg` 分别展示实际读取 engine 的 8 个 TP ranks；
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
  -> DUALPATH_DECODE_L3_READ
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
       -> <run>/longest-trajectory/{kv-new,kv-total,kv-cache-monitor,cache-hit}.svg
       -> <run>/longest-trajectory/l3-read-data.svg
       -> <run>/longest-trajectory/l3-read-time-prefill-<ID>.svg
       -> <run>/longest-trajectory/l3-read-time-decode-<ID>.svg
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

- Job `65548/65557`：[Mooncake Store 与 DFS smoke](mooncake-store.md)。
- 当前三条 read path 的完整对比：[dualpath-path-comparison.md](dualpath-path-comparison.md)。

Job `65032`/`65144`/`65502` 的单次诊断页已移除：它们的口径早于 Timing V3，数值
无法与当前结果并列，结论已被上面的对比覆盖。所有 metric 定义以本文为准。
