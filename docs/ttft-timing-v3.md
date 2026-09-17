# TTFT Timing V3

> 返回 [文档索引](README.md)。V3 是 2026-09-17 起正式使用的 TTFT breakdown
> 口径；旧 marker 仅用于历史结果 fallback。

## 开关与日志

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

## 四种时间

- **Raw duration**：该操作自身观测到的完整持续时间，可能与其他阶段重叠。
- **Critical contribution**：按固定归属规则去重后，计入 TTFT 堆叠柱的时间。
- **Overlap removed**：`raw - critical`，用于解释流水重叠，不画成额外阶段。
- **Misc**：客户端 `TTFT - sum(critical stages)`。它是剩余端到端时间，不是某个
  单独 server 函数的计时。

CPU 操作用 `perf_counter_ns()` 累计。GPU H2D 和 Prefill computation 只在 timing
开启时各增加常数个 CUDA event，不增加逐层 event，也不新增 `cuda.synchronize()`；
event 只在现有完成/ACK 路径读取。

## Prefill Read Path

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

## Decode Read Path

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

## TP=8 聚合

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

## Misc 中仍未拆分的开销

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
