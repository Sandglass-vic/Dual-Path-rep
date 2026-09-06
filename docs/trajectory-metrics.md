# Job 65502：完整轨迹 KV 与带宽历史测例

## 测试配置

- Job：`65502`
- 架构：1P1D，Prefill/Decode 均为 Qwen2.5-32B、TP=8
- 数据：`dualpath_fake_32k.jsonl` 的 `32k-0250`
- 轨迹：94 turns，真实输出会加入下一轮 context
- 隔离：每轮结束后清空 Prefill 本地 cache，并等待 Lustre L3 写入稳定
- KV 口径：Qwen2.5-32B BF16 的集群逻辑 KV 为
  `64 layers × 8 KV heads × 128 head_dim × K/V × 2 bytes = 262144 bytes/token`

Job 完成后已自动删除其 KV-cache 测试目录；仓库侧旧
`artifacts/trajectory-metrics/65502/` 已清理。下列结论仍有效。

正式口径（勿与本历史页混淆）：当前 `trajectory-*` 图是 **一条代表轨迹**
（非多 agent 平均）；单轨迹 JCT 图名为 **`trajectory_jct`**；跨 agent 箱线的
TTFT 是 **agent 内多 turn 先平均再跨 agent**。定义与产物布局见
[指标规范 · 口径速查](metrics.md#口径速查常见混淆)。

Job `65502` 是旧 file backend 的命中归因实验，不是性能结果。逐轮 flush、
长 storage wait 属于历史实现；当前正式 replay 恢复了论文对齐所需的逐轮
Prefill flush，并在短 `STORAGE_SETTLE_SECONDS` 后采样 Mooncake DFS 的
`st_blocks`。flush、settle 和扫描耗时均不进入 trajectory JCT。Job `65502`
的 JCT、physical snapshot 和旧带宽曲线仍不能与新 Mooncake Store 动态
rollout 混报。

## 命中率锯齿的来源

Decode offload manager 不会在每轮结束时发布一个包含全部新 context 的连续 L3
前缀。它按 admission、page 对齐和异步 backup 进度批量推进可检索前缀。例如
Job 65502 中，turn 16–25 的 `l3_read_tokens` 固定为 5952（`93 × 64`）：
L3 已发布前缀长度不变，而完整 prompt 持续增长，所以命中率下降；turn 26 发布
9280-token 前缀后命中率再次跃升。该锯齿是固定分子除以增长分母，再叠加批量
发布造成的，不表示已经写出的每个 page 都能立即扩展为可检索的连续前缀。

## 五类曲线

1. `trajectory_cache_hit_rate.svg`
   - 可复用前缀命中率：`cached_tokens / prompt_tokens`。
   - L3 读取命中率：`l3_read_tokens / prompt_tokens`。
   - 本测试每轮清空 Prefill L1/L2，因此两条线在出现 L3 命中后重合；两者在未清
     本地 cache 的普通 serving 中不能视为等价。
2. `trajectory_new_kv.svg`
   - 新增逻辑 KV：`(append_tokens + generated_tokens) × 256 KiB`。
   - CSV 另保留 L3 物理增量，即相邻 turns 的 `storage_bytes_after_turn` 差分；
     它受 64-token page、异步 offload 和 TP 文件布局影响，会出现批量尖峰，因而
     不与平滑的逻辑新增量画在同一张图中。
3. `trajectory_total_kv.svg`
   - turn 结束时的逻辑 KV：`(prompt_tokens + generated_tokens) × 256 KiB`。
   - L3 曲线只是 client 采样时看到的 `.bin` 文件总字节数。Decode offload 是异步
     的，而 Job 65502 的等待条件只要求新增八个 TP 文件并静默 1 秒，因此它不是
     “该 turn 全部 KV 已持久化后的 physical footprint”。
   - 例如 turn 10 的采样值只有 0.156 GiB；turn 15 异步 backlog 批量完成后单轮
     跳增约 1.203 GiB。这解释了它为什么不能与前十轮约 0.990 GiB 的逻辑 KV
     累加值逐点比较。图例已改为 `Observed L3 snapshot (async)`。
4. `trajectory_pd_bandwidth.svg`
   - 新口径计算：`page_count × page_size × 256 KiB /
     pd_transfer_seconds`，即实际提交的 page-aligned 逻辑 payload。
   - Job 65502 的旧 timing 镜像从最后一个 Prefill chunk 提交开始计时，因此历史
     曲线仍是“完成窗口逻辑有效带宽”。
   - 新 timing patch 在 `send_kv_chunk()` 第一次真正调用 Mooncake sender 前记录
     起点，并在最终 `KVPoll.Success` 结束，覆盖从首个实际 chunk 提交到全部
     传输完成的完整窗口。
5. `trajectory_l3_bandwidth.svg`
   - 计算：`l3_read_tokens × 256 KiB / l3_read_seconds`。
   - Job 65502 的历史曲线使用 tp_rank=0 的 wall duration。
   - 新 timing patch 记录每个 Prefill Engine 的全部八个 TP rank；新聚合器使用
     TP 完成共识 `all_reduce` 之前采集的 `max(rank local read duration)` 作为
     带宽分母，并输出
     `prefill_engine_<id>_l3_rank_times.svg`。分子仍表示 TP8 合计逻辑 KV payload。

所有曲线横坐标均为 `turn_id`。逐点原始值在
`trajectory_metrics.csv`，聚合结果在 `trajectory_metrics_summary.json`。

## Job 65502 结果

- 94/94 turns 成功，P→D timing 覆盖率 100%。
- 第 16 turn 首次读取 L3，L3 timing 覆盖 79/94 turns（84.04%）。
- 最后一轮命中率为 99.33%。
- 最后一轮结束时逻辑 KV 为 6.063 GiB，L3 物理占用为 6.031 GiB。
- P→D 完成窗口逻辑有效带宽中位数为 2.325 GB/s。
- 有 L3 读取的 turns 中，Prefill L3 逻辑有效加载带宽中位数为 11.481 GB/s。

这些带宽值用于当前实现内部的相对趋势和后续消融对比。若要报告硬件网络物理
带宽，还需按 request 时间窗采集 IB 端口计数器，或在 Mooncake transfer engine
中记录每个 chunk 的实际 wire bytes 与首次提交时间。

## 当前口径

本页公式与字段仅用于解释 Job `65502`。当前正式运行的 JCT、TTFT/TTST/TPOT、
request/L3/P→D timing、八个 TP rank 聚合、cache hit、产物和闭合规则统一见
[指标规范](metrics.md)。

修正版 Job `66097`（1P2D、32K、1 agent、26 turns）验证结果：L3 allocated
snapshot 最终为 6.932 GiB，P→D timing coverage 为 100%，P→D 有效带宽中位数
为 9.483 GB/s；6 个发生 L3 read 的 requests 的 TP=8 聚合加载带宽中位数为
43.837 GB/s。Mooncake 稀疏 shard 的 apparent size 固定为 256 GiB，不表示实际
落盘量。

Job `65502` 是非流式历史结果，因此没有逐轮 TTFT/TTST/TPOT；不得从缺失字段
推导零值。其他历史结果见 Job `65032` 的 [四轮延迟结果](latency-results.md)
和 Job `65144` 的 [TTFT 全链路测例](ttft-fullchain.md)。

