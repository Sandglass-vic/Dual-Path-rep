# 已修复缺陷的根因记录

> 返回 [文档索引](README.md)

五个缺陷在 2026-09 的 L3 路径重构中被定位并修复，修复已全部合并进 main 并由
`scripts/infra/build_benchmark_timing_image.sbatch` 的构建期断言守住。本页只记录
**根因和防回归措施**，不保留当时的验收流水账；当前性能口径见
[dualpath-path-comparison.md](dualpath-path-comparison.md)。

共同背景：Prefill 的 radix namespace 按请求私有
（`DUALPATH_NO_CROSS_TURN_KV_REUSE=1`），Prefill 不写 L3
（`DUALPATH_HICACHE_READ_ONLY=1`），跨轮复用只能走 durable L3。多数缺陷都源于
这个前提没有在某个环节贯彻到底。

## 1. Prefill 前缀复用率恒为 0

**根因。** `scheduler.py` 的 `_prefetch_kvcache()` 调用 `prefetch_from_storage()`
时只传了 5 个参数，第 6 个 `extra_key` 取默认 `None`。prefetch 完成后
`check_prefetch_progress()` 以 `RadixKey(token_ids=..., extra_key=None)` 把恢复的
page 插进 radix 树，而请求自己用 `dualpath:{rid}:` 去 `match_prefix`，永远匹配不到
挂在 `None` 命名空间下的节点。每轮都退化为整条 prompt 重算，尽管 durable 前缀命中
率有 94.5%。

Decode path 不受影响，它绕开 radix 匹配，直接沿用 D→P 反向传输装好的 prefix
indices。

这是上一轮"namespace 解耦"改过头的结果：durable L3 的 hash 查询不带 namespace 是
对的（跨轮必须共享），但"恢复后插回请求自己的 namespace"这后半句从未接上。

**修复。** 新增幂等 helper `patch_scheduler_prefetch_namespace()`，在全量 patch 和
`Already patched` 两个分支都注入 `extra_key=req.extra_key`。原先标着
`"keep request radix namespace out of durable L3 hash lookup"` 的 `replace_once()`
其 old 与 new 完全相同，是个空操作，已替换。

`extra_key` 只进入 `ongoing_prefetch` 供 `_insert_helper_host()` 构造 RadixKey，
**不出现在** `cache_controller.prefetch()` 的实参里，因此 durable L3 的 hash 查询
仍然无 namespace。

**效果。** Prefill 计算从 3.953 s 降到 0.310 s，平均 TTFT 从 11.664 s 降到 1.874 s。

## 2. `python -O` 让 95 条构建断言全部失效

**根因。** 构建脚本本有一条专防上一条缺陷的断言
`assert "extra_key=req.extra_key" in scheduler`，但镜像内验证跑在 `python -O` 下，
`assert` 被解释器整个剥掉。**该文件当时的 95 条断言全是死代码**，这正是缺陷 1 能
静默进入镜像的原因。

引入 `-O` 原本是为了跳过不适配 partial patch 镜像的旧 marker 断言，而现在的构建
流程是从原始基底全量 patch，这个权宜之计已无必要。

**修复。** 把 `python -O -` 改回 `python -`。实测对全量 patch 的源码树 95 条断言
全部通过，无需重写。并补了三条针对缺陷 1 的断言，其中第三条是关键：

```python
assert "extra_key=req.extra_key," in scheduler
assert "extra_key=extra_key" in hiradix
assert "extra_key" not in hiradix[
    hiradix.index("operation = self.cache_controller.prefetch(") :
][:200]
```

它确保 namespace 只进入本地 radix 插入、不污染 durable L3 的 hash 查询，避免
"修好前缀复用却弄坏跨轮共享"。

> 若将来需要从 partial patch 镜像重建并遇到断言失败，正确做法是把断言分成"必须
> 存在"与"仅全量 patch 才存在"两层，**不要**重新加回 `-O`。

## 3. Prefetch 配额静默跳过请求

**根因。** SGLang 对在飞 storage prefetch 有全局 token 配额
（`managers/cache_controller.py`）：

```python
self.prefetch_capacity_limit = int(
    0.8 * (self.mem_pool_host.size - self.mem_pool_device.size)
)
```

`prefetch_from_storage()` 在入口用 `prefetch_rate_limited()` 检查它，超限就直接
`return`：不发 hash query、不落日志、不重试，该请求本轮退化为全量重算，在日志里
与"存储真的没命中"无法区分。

配额取决于 **host pool 与 device pool 的差**，不是 host pool 本身，所以它对 GPU
的 KV 容量很敏感。

**修复。** 新增 `HICACHE_SIZE_GB`（映射到 `--hicache-size`，非 0 时覆盖
`--hicache-ratio`），并新增 `DUALPATH_L3_PREFETCH_SKIP` 日志，把三种静默跳过
（`rate_limited` / `below_threshold` / `no_host_mem`）显式打出来。

**两个平台的取值方向相反**，见
[hicache-and-dualpath.md](hicache-and-dualpath.md#host-pool-尺寸与-l3-prefetch-配额)：

- A100 40GB：device pool 小，ratio 派生的配额塌到约 26k tokens，比单个 32K 前缀
  还小，必须用绝对值（40 GB/rank，配额 413k）；
- H100 80GB：device pool 有 1,131,200 tokens，绝对 40 GB 只剩 89k 的差值、配额
  71,628，反而**低于** ratio 1.35 给出的 316,736。GAIA 默认 `HICACHE_SIZE_GB=0`。

验收标准：正式请求的 `DUALPATH_L3_PREFETCH_SKIP` 不应出现 `reason=rate_limited`；
健康检查的 `below_threshold requested_tokens=0` 不算。

## 4. 队列等待沉在 misc 残差里

**根因。** `summarize_ttft_fullchain.py` 里 `misc_seconds = max(0, ttft -
critical_sum)` 是残差而非测量值，会同时吸收真实排队和任何打点缺口。实测它占 TTFT
的 35–50%，晚期轮次达 65%，其中 87–97% 其实是排队。

**修复。** 把 queue wait 提升为 TTFT V3 的正式阶段。不需要新增打点：
`RequestTimeStats` 用 `time.perf_counter()`，`dualpath_ttft.now_ns()` 用
`time.perf_counter_ns()`，同一时钟，直接复用调度器已有的时间戳。

| 侧 | 窗口起点 | 窗口终点 |
|---|---|---|
| Prefill | `prefill_bootstrap_queue_entry_time` | `forward_entry_time` |
| Decode | `decode_prealloc_queue_entry_time` | `decode_transfer_queue_entry_time` |

**关键是重叠扣除。** 队列窗口天然包含这个请求自己的 L3 读——它排队期间引擎正在为
它读盘，直接相加会让 `critical_sum` 超过 TTFT。`_v3_queue_wait()` 求队列窗口与读
窗口的**交集**并扣掉（扣交集而非读耗时，因为读可能在请求排到队头之前就开始）：

```python
overlap = max(0.0, min(q_end, r_end) - max(q_start, r_start)) / 1e9
return max(0.0, raw - overlap)
```

两条路径的队列窗口是同一段物理时间，所以只计 owner 侧那一个。修复后 misc 降到
约 0.08 s。

## 5. Decode 侧 L3 写放大 16 倍

**根因。** 两条路径的写方都是 Decode（Prefill 在任何模式下都不写 L3）。
`decode_kvcache_offload_manager` 的 `offload_start` 由 `cached_tokens // page_size`
推出，而 `cached_tokens` 在非 decode 读路径下被硬编码成 0，于是 prefill path 每轮
把整个上下文重写一遍：

| | `offload_start > 0` | 每请求每 rank | L3 写入总量 |
|---|---:|---:|---:|
| prefill path | 0 / 416 | 19,892 token | **505.1 GiB** |
| decode path | 400 / 416 | 1,246 token | 31.6 GiB |

硬编码 0 当初是保守选择，理由是"Router 的 cached_tokens 聚合了多层缓存，不是
durable 前缀边界"。缺陷 1 的 namespace 修复改变了这个前提。

**修复。** 两个开关同时成立时（namespace 请求私有 + Prefill 只读），
`req.cached_tokens` 精确等于 durable L3 前缀，而 Decode 早已通过 P→D metadata
buffer 收到它。新增幂等 helper `patch_decode_offload_prefill_path_boundary()`：

```python
if decode_l3_read:
    dualpath_durable_prefix = getattr(req, "dualpath_decode_l3_hit_tokens", 0)
elif dualpath_prefill_l3_prefix_trusted:      # 上述两个开关都为 1
    dualpath_durable_prefix = getattr(req, "cached_tokens", 0)
else:
    dualpath_durable_prefix = 0               # 保持原有保守行为
```

关掉任一开关就退回全量重写。写量从 8,274,944 降到 518,144 token/run（16 倍）。

**风险有界**：即使边界估大，被跳过的页下一轮在 L3 里查不到，hash 链在该页断开、
前缀命中变短、退化为重算，是性能损失而非正确性问题。

该 guard 运行在 Decode 上，因此**不能读 Prefill 专有的环境变量**
（`DUALPATH_HICACHE_READ_ONLY` 在 Decode 侧未设置），构建断言会检查这一点。

### 附带：decode path 的 `cached_tokens` 重复计数

decode path 的 `cached_tokens` 恰好是实际值的 2 倍，同一批 token 被两处各记一次：

| 写入点 | 性质 |
|---|---|
| `prefill.py` D→P 反向处理里 `req.cached_tokens = hit_tokens`，经 metadata buffer 送给 Decode | DualPath 补丁 |
| `decode_schedule_batch_mixin.py` 的 `req.cached_tokens += pre_len - req.already_computed` | 上游 SGLang |

Decode 从 L3 暂存进来的前缀就是它自己的 `prefix_indices`，上游那个 `+=` 已经算对
了，Prefill 送来的那份多余。修复放在 Decode 的接收点：decode 读路径下不采用
Prefill 传来的值。**这是纯报告口径修复**，不影响 offload 边界。

## 已作废的结论

排查过程中有两个结论后来被证伪，记录下来避免重走：

**"部分请求的 L3 读取结果与 hash 命中不一致"** —— 原始证据是串号的日志。逐一对齐
104 个请求后事实相反：只要 query 发出去了，`read_tokens` 与
`DUALPATH_L3_HIT_CONSENSUS.hit_tokens` 完全一致，88/88 无例外。hash 链、TP 共识、
O_DIRECT 读、page 发布都正常，问题只在"要不要发起预取"这一步（即缺陷 3）。

**用 `if req.last_node.backuped or ...` 作为预取门槛** —— 空操作。
`hiradix_cache.py` 初始化时设了 `self.root_node.host_value = []`，而 `backuped` 的
定义是 `host_value is not None`，故 root 恒为 `True`；又因为 namespace 按请求私有，
`match_prefix` 跨轮永不命中、`last_node` 永远是 root。该门槛从未拦过任何请求。
