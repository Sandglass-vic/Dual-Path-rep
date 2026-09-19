# DualPath Prefill L3 路径调试交接

> 更新时间：2026-09-19
>
> 本文交接 pza100 / 1P1D / Prefill L3 read 的当前调试状态。先读本文，再按需读
> [pza100 环境](pza100.md)、[PD 运行手册](runbook-pd.md)、
> [HiCache 与 DualPath](hicache-and-dualpath.md)、[TTFT Timing V3](ttft-timing-v3.md)。
>
> 上一份交接是 [task-handoff-2026-09-19.md](task-handoff-2026-09-19.md)，其中的
> prefetch 配额问题已修复并验证。本文接续处理新发现的问题。

## 摘要

Prefill L3 read 路径存在一个 **P0 级功能缺陷**：durable L3 的 KV 被正确读回，但因为
radix namespace 对不上，恢复的 page 从未被前缀复用消费，Prefill 每轮都在全量重算。
`#cached-token` 全程为 0。

同镜像同数据的 decode path 对照组复用率 94.6%，证明这不是数据或存储问题，只是
prefill path 的一处参数传递缺失。

**该缺陷已修复并通过新镜像 + 两条 O_DIRECT 路径的全流程验收（§2、§3、§7）。**
prefill path 的 mean TTFT 从 `11.664s` 降到 `1.854s`，JCT 从 `436.1s` 降到 `224.9s`。

在此基础上，基于 487/488 的数据分析又发现并修复了两项，**代码已就绪并通过本地验证，
但尚未重建镜像、尚未跑实验**：

| | 问题 | 状态 |
|---|---|---|
| §4 | 队列等待没有被计量，全部沉在 `misc` 里（占 TTFT 的 35%–49%） | 已实现显式阶段 |
| §5 | prefill path 下 Decode 把整个上下文重写进 L3，写放大 16 倍 | 已修复 |
| §5 | decode path 的 `cached_tokens` 被两侧各记一次，虚高 2 倍 | 已修复（报告口径） |

接手方需要做的是：重建镜像 → 按 §6 跑验收 → 回填结果。

## 当前环境

- 仓库：`/data/home/yongxiao/dual-path`，分支 `pza100`
- 工作区脏（38 个文件有改动），包含此前的实验产物与修改。**不要**使用
  `git reset --hard`、`git checkout --` 或覆盖无关文件
- 平台 `g5`/`g6`，每节点 4 卡 **A100 40GB**（注意不是 H100 80GB，旧文档里的部分
  容量数字按 H100 写的）
- 模型 `/data/home/yongxiao/mydata/models/Qwen2.5-32B`，TP4，1P1D
- 运行时根目录 `/data/home/yongxiao/mydata`

### 当前镜像

```text
/data/home/yongxiao/mydata/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-hostpool40.sqsh
```

SGLang 版本 `0.5.7+31b61bbe`。该镜像由 Slurm job `483` 从原始基底
`sglang:26.01-py3.sqsh` 全量 patch 构建（构建日志 `~/timing-image-483.out`，全部
`Patched:`，无 `Already patched:`）。

该镜像已包含 `DUALPATH_L3_PREFETCH_SKIP` 诊断补丁，但**不含**本文 §2 的修复，
**必须重建**。重建可以用原始基底，也可以用这个镜像——§2/§3 的改动在全量 patch 与
`Already patched` 两条分支里都生效，且已验证幂等。

镜像内源码可用 `unsquashfs` 直接取出核对，这是确认"补丁到底有没有进镜像"的可靠手段：

```bash
unsquashfs -d /tmp/sgimg -n -q <image>.sqsh \
  'usr/local/lib/python3.12/dist-packages/sglang/srt/managers/scheduler.py'
```

### 参考实验

| alloc | 结果目录 | `DUALPATH_L3_READ_PATH` |
|---|---|---|
| 484 | `results/latency-benchmark/32k-a4-r1-prefill-odirect-hostpool40` | `prefill` |
| 485 | `results/latency-benchmark/32k-a4-r1-decode-odirect-hostpool40` | `decode` |

两者镜像相同、数据集相同（32K / agents-4 / rep-1 / 104 轮），只有读路径不同，是干净的
对照组。引擎日志在 `/data/home/yongxiao/mydata/test/<alloc>/pd-servers/`。

## 1. 已完成并验证：prefetch 配额修复

SGLang 对在飞 storage prefetch 有一个全局 token 配额
（`managers/cache_controller.py`）：

```python
self.prefetch_capacity_limit = int(
    0.8 * (self.mem_pool_host.size - self.mem_pool_device.size)
)
```

A100 40GB 上 `max_total_num_tokens=94080`，原 `--hicache-ratio 1.35` 只给出 26368
tokens 的配额，比单个 32K 前缀（约 32256）还小，导致并发时大量请求被静默跳过。

修复：新增 `HICACHE_SIZE_GB`（映射到 `--hicache-size`，GB 整数，非 0 时覆盖
`--hicache-ratio`），默认 40，配额变为 413030 tokens。详见
[HiCache 与 DualPath](hicache-and-dualpath.md#host-pool-尺寸与-l3-prefetch-配额)。

484 验收结果，**此项已关闭**：

- `DUALPATH_L3_HIT_QUERY` 独立 rid 数 = 104 / 104（修复前 88）
- 非首轮 `read_tokens=0` 的请求数 = 0（修复前 16）
- `DUALPATH_L3_PREFETCH_SKIP` 共 40 条，全部是健康检查请求的
  `reason=below_threshold requested_tokens=0`，**无 `reason=rate_limited`**

## 2. P0：prefill path 前缀复用率为 0（已验证修复）

### 现象

484 与 485 的 TTFT 分解（单位秒，104 轮均值）：

| 阶段 | prefill path | decode path | 评价 |
|---|---:|---:|---|
| KV read (Lustre) | 0.594 | 2.363 | 预期内：prefill 读本地 XFS，decode 走 NFS |
| scatter | 0.146 | 0.193 | 接近 |
| **prefill_computation** | **3.953** | **0.361** | **11x，异常** |
| p_to_d_transfer | 1.080 | 0.014 | 见下文，部分属架构差异 |
| decode_host_to_device | 0.058 | 0.055 | 接近 |
| decode_dispatch | 0.015 | 0.014 | 接近 |
| misc | 5.811 | 3.033 | 排队，是上面的后果 |
| **mean TTFT** | **11.664** | **6.146** | |
| **JCT median** | **436.1** | **315.7** | |

### 证据

统计两侧 `prefill-0.log` 的 `Prefill batch` 行：

```bash
python3 - <<'PY'
import re
for a in ("484", "485"):
    nt = ct = 0
    for ln in open(f"/data/home/yongxiao/mydata/test/{a}/pd-servers/prefill-0.log",
                   errors="replace"):
        m = re.search(r"#new-token: (\d+), #cached-token: (\d+)", ln)
        if m:
            nt += int(m.group(1)); ct += int(m.group(2))
    print(a, f"new={nt:,} cached={ct:,} reuse={ct/(ct+nt):.1%}")
PY
```

| | prefill path (484) | decode path (485) |
|---|---:|---:|
| `#new-token` 合计（真正 forward） | 2,050,752 | 111,616 |
| `#cached-token` 合计（前缀复用） | **0** | 1,939,200 |
| 复用率 | **0.0%** | 94.6% |

prefill path 从 L3 读了 **1,939,200** 个 token（与 decode path 复用量完全一致），
然后全部丢弃并重算 205 万 token。那 0.594s 的 Lustre 读加 0.146s 的 scatter 是纯开销。

### 根因

484 镜像 `srt/managers/scheduler.py` 的 `_prefetch_kvcache()` 调用点只传了 5 个参数：

```python
self.tree_cache.prefetch_from_storage(
    req.rid,
    req.last_host_node,
    new_input_tokens,
    last_hash,
    prefix_keys,
)
```

`prefetch_from_storage()` 的第 6 个参数 `extra_key: Optional[str] = None` 因此取默认
`None`。prefetch 完成后，`check_prefetch_progress()` 以
`RadixKey(token_ids=..., extra_key=None)` 把恢复的 page 插进 radix 树。

而 `DUALPATH_NO_CROSS_TURN_KV_REUSE=1` 使每个请求的
`req.extra_key = f"dualpath:{rid}:"`（见 `schedule_batch.py` 的 patch）。于是
`init_next_round_input()` 用 `dualpath:{rid}:` 去 `match_prefix`，永远匹配不到挂在
`None` 命名空间下的节点。

decode path 不受影响，因为它绕开了 radix 匹配：`dualpath_decode_l3_prefix_ready`
会让 `schedule_policy.py` 与 `scheduler.py` 跳过重新匹配，直接沿用 D→P 反向传输装好的
prefix indices。

这是上一轮"namespace 解耦"改过头的结果。当时的意图是：durable L3 的 hash 查询不带
namespace（正确，跨轮必须共享），但恢复后仍插入当前 request 的本地 namespace。后半句
从未接上——调用点根本没把 namespace 传下去。

### 已应用的修复

`scripts/infra/patch_sglang_benchmark_timing.py` 中，原先标签为
`"keep request radix namespace out of durable L3 hash lookup"` 的 `replace_once()`
**old 与 new 完全相同，是个空操作**。已替换为新的幂等 helper
`patch_scheduler_prefetch_namespace()`，在 `patch_decode_l3_scheduler()` 的全量 patch
分支和 `Already patched` 分支里都会调用，生成：

```python
self.tree_cache.prefetch_from_storage(
    req.rid,
    req.last_host_node,
    new_input_tokens,
    last_hash,
    prefix_keys,
    extra_key=req.extra_key,
)
```

安全性已在打完补丁的源码上逐条确认：`prefetch_from_storage()` 里 `extra_key` 只被存进
`ongoing_prefetch`，完成时供 `_insert_helper_host()` 构造
`RadixKey(token_ids=..., extra_key=extra_key)`；它**不出现在**
`cache_controller.prefetch(req_id, host_indices, new_input_tokens, last_hash, prefix_keys)`
的实参里。因此 durable L3 的 hash 查询仍然无 namespace，不会回退到上一轮修掉的问题。
§3 新增的断言会持续守住这一点。

`patch_hiradix()` 的 "Already patched" 分支里那行
`source.replace("extra_key=req.extra_key,\n", "")` 作用于 `hiradix_cache.py`，与本次改动
所在的 `scheduler.py` 是不同文件，已验证不会互相干扰。

### 本地验证结果

用 `--site-packages` 参数可以在本机完整复现构建，不需要 Slurm：

```bash
unsquashfs -d /tmp/sgpristine -n -q \
  /data/home/yongxiao/mydata/images/sglang:26.01-py3.sqsh \
  'usr/local/lib/python3.12/dist-packages/sglang'
python3 scripts/infra/patch_sglang_benchmark_timing.py \
  --site-packages /tmp/sgpristine/usr/local/lib/python3.12/dist-packages
```

已验证：

- 原始基底与 484 镜像两种起点，patch 后均能编译，`extra_key=req.extra_key,` 恰好 1 处
- 连续两次 patch 结果不变（幂等）；第二遍有 26 个文件走 `Already patched` 分支，
  §3 的校验仍然通过
- 人为删掉 `extra_key=req.extra_key,` 复现 484 的缺陷后，§3 的校验以 `AssertionError`
  拦截并返回退出码 1

### 预期效果与不会变的部分

- `prefill_computation` 应从 3.953s 降到 0.36s 量级（与 decode path 同等后缀规模）
- engine 占用大幅下降，排队随之收缩，`misc` 应显著变小
- prefill path 的 TTFT 应反超 decode path（它的 L3 读本来就快约 4 倍）

**`p_to_d_transfer` 不会降到 decode path 的 0.014s。** prefill path 下 Decode 手里没有
prefix，Prefill 必须把整个 32K 上下文推过去；decode path 只需传后缀。1s 量级是架构差异，
不是缺陷。

另外 `p_to_d_transfer_raw_seconds` 与 `prefill_computation_raw_seconds` 几乎相等
（如 6.710 vs 6.703），两者并发；`p_to_d_overlap_removed` 约 5.1s，余下约 1.59s 的尾巴
在 turn 17–26 间基本恒定。怀疑与 `chunked_prefill_size=8192` 的最后一个 chunk 有关。
修完 P0 后再复核，不要在此之前下结论。

## 3. 构建校验在 `python -O` 下全部失效（已验证修复）

`scripts/infra/build_benchmark_timing_image.sbatch` 原本有一条专门防这次回归的断言：

```python
assert "extra_key=req.extra_key" in scheduler
```

484 镜像的 `scheduler.py` 里该字符串出现 **0 次**，断言却没有报错——因为镜像内验证跑在
`python -O` 下，`assert` 语句被解释器整个剥掉。该文件当时的 **95 条 `assert` 全是死
代码**，这正是 §2 的缺陷得以静默进入镜像的原因。

历史上引入 `-O` 是为了跳过不适配 partial patch 镜像的旧 marker 断言。但现在的构建流程
（job 483）是从原始基底全量 patch，这个权宜之计已无必要。实测确认：对全量 patch 后的
源码树，**95 条断言在不带 `-O` 时全部通过**。

因此修复就是把 `python -O -` 改回 `python -`，无需重写这 95 条断言，也无需分层。
同时补了三条针对 §2 的断言：

```python
assert "extra_key=req.extra_key," in scheduler
assert "extra_key=extra_key" in hiradix
assert "extra_key" not in hiradix[
    hiradix.index("operation = self.cache_controller.prefetch(") :
][:200]
```

第三条是关键：它确保 namespace 只进入本地 radix 插入，**不会**污染 durable L3 的 hash
查询，避免"修好前缀复用却弄坏跨轮共享"。

验证方式（本地即可，不需要容器）：

```bash
# 取出校验块，把 root 指向本地已 patch 的源码树后直接跑
python3 <checks>.py            # 全量 patch 树 -> SGLANG_BENCHMARK_TIMING_PATCH_OK
python3 <checks>.py            # 人为删掉 extra_key 的树 -> AssertionError, 退出码 1
```

> 若将来确实需要从 partial patch 镜像重建并遇到断言失败，正确做法是把断言分成"必须
> 存在"与"仅全量 patch 才存在"两层，**不要**重新加回 `-O`。

## 4. 队列等待没有被计量（已实现显式阶段，待验证）

`scripts/benchmark/summarize_ttft_fullchain.py` 里：

```python
output["misc_seconds"] = max(0.0, ttft - critical_sum)
```

`misc` 不是被测量的阶段，而是残差，会同时吸收真实排队和任何打点缺口。484 里它占 TTFT
的 50%（5.81s / 11.66s），晚期轮次达 65%。

把 `misc` 与引擎 `Req Time Stats` 逐轮对齐后：

- 484（prefill path）：`bootstrap_queue_duration + queue_duration` 占 misc 的 **92%**
- 晚期轮次排队整体迁移到 `bootstrap_queue`（12–13s，且全部落在 `other` 分项），说明那时
  瓶颈转到等 Decode 侧就绪
- 485（decode path）：排队总量更大（5.63s vs 5.35s），misc 反而更小（3.03 vs 5.81），
  因为 Decode 的 L3 读与 Prefill 侧排队是同一段物理时间，已计入 `decode_lustre_read`

两个 run 都没有出现钳位（`misc == 0`）或 `accounting_overlap_excess > 0`，分解本身自洽。
但 **misc 的语义在两条路径下不同，不能直接横向比较**。

487/488 上同样成立：prefill path 的 misc `0.644s` 里 87% 是排队，decode path 的 `2.806s`
里 97% 是排队。

### 已实现：`queue_wait` 阶段

排队现在是 TTFT V3 的一个正式阶段，不再沉在 misc 里。

**不需要新增打点。** `RequestTimeStats` 用 `time.perf_counter()`，`dualpath_ttft.now_ns()`
是 `time.perf_counter_ns()`，同一时钟，所以直接复用调度器已有的时间戳：

| 侧 | 窗口起点 | 窗口终点 |
|---|---|---|
| Prefill | `prefill_bootstrap_queue_entry_time` | `forward_entry_time` |
| Decode | `decode_prealloc_queue_entry_time` | `decode_transfer_queue_entry_time` |

新增 helper `dualpath_ttft.queue_window_ns(stats, start_attr, end_attr)` 负责换算，两侧
emit 各输出 `queue_wait_raw_seconds` / `queue_wait_start_ns` / `queue_wait_end_ns`。

**关键是重叠扣除。** 队列窗口天然包含这个请求自己的 L3 读——它排队期间引擎正在为它读
盘。直接相加会让 `critical_sum` 超过 TTFT。`summarize_ttft_fullchain._v3_queue_wait()`
求队列窗口与读窗口的**交集**并扣掉：

```python
overlap = max(0.0, min(q_end, r_end) - max(q_start, r_start)) / 1e9
return max(0.0, raw - overlap)
```

注意扣的是交集而不是读耗时——读可能在请求排到队头之前就开始了。

两条路径的队列窗口是同一段物理时间（Prefill 的 bootstrap 队列要等 Decode 就绪才排空），
所以只计 owner 侧那一个：prefill path 取 Prefill 侧，decode path 取 Decode 侧。

绘图上 `queue_wait` 排在各阶段最前，配色用新增的 `slate`（中性深灰），与 `Misc` 的浅灰
和 `D->P transfer` 的 pink 都能区分。

### 验证结果

用 487/488 的历史日志反推出 `queue_wait` 字段后重跑汇总器：

| | queue_wait | misc（新） | misc（旧） | `critical > ttft` | 超容差 |
|---|---:|---:|---:|---:|---:|
| prefill path | 0.559 s | **0.085 s** | 0.644 s | 0 | 0 |
| decode path | 2.730 s | **0.076 s** | 2.806 s | 0 | 0 |

misc 收敛到 0.08s 量级——正是预期的客户端往返残量，且没有任何账目溢出。

向后兼容也验过：旧日志没有 `queue_wait_*` 字段时 `queue_wait_seconds = 0`、`misc` 与
改动前完全一致（0.644 / 2.806），绘图侧有 `setdefault` 兜底。

## 5. Decode 侧 L3 写放大 16 倍（已修复，待验证）

### 现象

统计 `DUALPATH_DECODE_OFFLOAD_BACKUP_COMPLETE` 的 `offload_tokens`（416 条 = 104 请求 ×
4 rank）：

| | `offload_start > 0` | 每请求每 rank | L3 写入总量（4 rank） |
|---|---:|---:|---:|
| prefill path (487) | **0 / 416** | 19,892 token | **505.1 GiB** |
| decode path (488) | 400 / 416 | 1,246 token | 31.6 GiB |

两条路径的写方**都是 Decode**——Prefill 在任何模式下都不写 L3
（`_inc_hit_count()` 在 `DUALPATH_HICACHE_READ_ONLY=1` 时直接 return）。差别在于
prefill path 每轮把整个上下文重写一遍，decode path 只写新增部分。

### 根因

`decode_kvcache_offload_manager` 里 `offload_start` 由 `cached_tokens // page_size` 推出，
而 `cached_tokens` 在非 decode 读路径下被硬编码成 0：

```python
cached_tokens = min(max(int(
    getattr(req, "dualpath_decode_l3_hit_tokens", 0) if decode_l3_read else 0
), 0), len(req.origin_input_ids))
```

当初这样写是保守选择，因为"Router 的 cached_tokens 聚合了多层缓存，不是 durable 前缀
边界"。§2 的 namespace 修复改变了这个前提。

### 已应用的修复

`DUALPATH_NO_CROSS_TURN_KV_REUSE=1` 让 Prefill 的 radix namespace 按请求私有，所以它唯一
可能的前缀命中就是 L3 prefetch 插入；`DUALPATH_HICACHE_READ_ONLY=1` 又保证 Prefill 不会
本地 backup。两个开关同时成立时，`req.cached_tokens` 就精确等于 durable L3 前缀。

实测佐证：487 的 `rounds.jsonl` 里 `cached_tokens` 合计 `1,939,200`，与 L3 读 token 数
完全一致。而这个值 Decode 早就通过 P→D metadata buffer 收到了
（`decode.py`: `decode_req.req.cached_tokens = cached_tokens[0].item()`）。

新增幂等 helper `patch_decode_offload_prefill_path_boundary()`，在两条分支里都生效：

```python
if decode_l3_read:
    dualpath_durable_prefix = getattr(req, "dualpath_decode_l3_hit_tokens", 0)
elif dualpath_prefill_l3_prefix_trusted:      # 上述两个开关都为 1
    dualpath_durable_prefix = getattr(req, "cached_tokens", 0)
else:
    dualpath_durable_prefix = 0               # 保持原有保守行为
```

关掉任一开关就退回全量重写，不会静默做错误假设。

### 风险评估

即使边界估大了也不会丢数据：被跳过的页下一轮在 L3 里查不到，hash 链在该页处断开，前缀
命中变短、退化为重算，是性能损失而非正确性问题。

### 附带修复：decode path 的 `cached_tokens` 重复计数

decode path 的 `rounds.jsonl` `cached_tokens` 是 `3,878,400`，恰好是实际值的 2 倍。同一批
物理 token 被两处各记了一次，实测两边贡献完全对称：

| 写入点 | 性质 | 贡献 |
|---|---|---:|
| `prefill.py` D→P 反向处理里 `req.cached_tokens = hit_tokens`，经 metadata buffer 送给 Decode | DualPath 补丁 | 1,939,200 |
| `decode_schedule_batch_mixin.py:65` `req.cached_tokens += pre_len - req.already_computed` | **上游 SGLang** | 1,939,200 |

Decode 从 L3 暂存进来的前缀就是它自己的 `prefix_indices`，所以上游那个 `+=` 本身已经
算对了；Prefill 送来的那份是多余的。

修复放在 Decode 的接收点（新增幂等 helper `patch_decode_cached_tokens_double_count()`）：
decode 读路径下不采用 Prefill 传来的值，改由 Decode 自己的 `prepare_for_extend` 计一次。
没有动上游的累加逻辑，也没有动 Prefill 侧的内部状态。

**这是纯报告口径修复**，不影响 offload 边界：decode path 的边界用的是
`dualpath_decode_l3_hit_tokens`，prefill path 用的是 `req.cached_tokens`，而 prefill path
不存在重复计数（487 实测 `1,939,200`，与 L3 读 token 数完全一致，是 1 倍）。

## 6. 接手方要做的事

§4、§5 的代码改动已完成并通过本地验证，剩下的是重建镜像与实验验收。

### 步骤一：重建镜像

沿用 job 486 的方式，覆盖 sbatch 里默认指向 `/labhome/sezhou` 的路径变量
（`CODE_HOME` / `RUNTIME_HOME` / `BASE_IMAGE` / `OUTPUT_IMAGE`）。建议输出名带
`queue-wait` 以便与 `prefix-namespace-fix` 区分。

构建会真正执行全部断言（本轮新增 8 条：`queue_wait` 阶段 4 条、写放大 2 条、原有 2 条）。
**若报 `AssertionError`，说明补丁没有完整落地，不要加回 `-O` 绕过**，按 §3 末尾处理。

构建成功的标志仍是末尾打印 `SGLANG_BENCHMARK_TIMING_PATCH_OK`。

提交 Slurm 前建议先本地验证，几秒即可（见 §8 最后一条）。

### 步骤二：验收实验

两条路径各跑一次冷 cache 的 `32K / agents-4 / rep-1`，`CONTEXT_LENGTH=32768`、
`DUAL_PATH_O_DIRECT_READ=1`。

**先确认 §2、§3 的成果没有回退**（这两项已验证过，属于回归保护）：

1. Prefill 复用率仍在 94.5% 左右，`prefill_computation_seconds` 仍在 0.5s 以下
2. `DUALPATH_L3_HIT_QUERY` 独立 rid 数 = 104，无 `reason=rate_limited` 的 SKIP
3. mean TTFT 与 487/488 基本持平（1.854s / 5.772s）

**再验本轮的两项新改动：**

4. **`queue_wait_seconds` > 0 且 `misc_seconds` 降到 0.1s 以下** —— §4 的核心指标。
   参考值：prefill path queue_wait ≈ 0.56s、decode path ≈ 2.73s
5. `accounting_error_exceeds_tolerance` 全为 False，`accounting_overlap_excess_seconds`
   全为 0。**这一条最关键**：如果出现 `critical > ttft`，说明重叠扣除有问题，
   优先查 `_v3_queue_wait()` 里读窗口的字段名（Prefill 是 `prefetch_before_read_*`，
   Decode 是 `decode_before_read_*`，两者不同）
6. TTFT 分解图里应出现深灰色的 `Queue wait` 段，排在最前
7. **prefill path 的 `offload_start > 0` 应接近 400/416，L3 写入总量从 505 GiB 降到
   30 GiB 量级** —— §5 的核心指标。统计命令见 §5
8. 写放大修复后，prefill path 的复用率**不应下降**。若下降，说明 `cached_tokens` 边界
   估大了，把 `DUALPATH_NO_CROSS_TURN_KV_REUSE` 或 `DUALPATH_HICACHE_READ_ONLY` 置 0
   即可退回旧行为定位
9. **decode path 的 `rounds.jsonl` `cached_tokens` 合计应从 `3,878,400` 降到
   `1,939,200`**（与 L3 read tokens 相等）；prefill path 保持 `1,939,200` 不变

### 步骤三：回填

把实测数字补进 §4 的验证表和 §5 的对比表，并把两节标题改为"已验证"。§7 追加一小节
记录本轮的镜像、allocation 和结果目录。

## 7. 历次验收结果

### 第一轮：prefix namespace 修复（§2/§3）

已从原始基底重建新镜像，构建 job `486`。构建验证使用普通 `python`，没有使用
`python -O` 绕过断言，并成功打印 `SGLANG_BENCHMARK_TIMING_PATCH_OK`：

```text
/data/home/yongxiao/mydata/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-prefix-namespace-fix.sqsh
SHA256: 12a94ec5e0fa5682c07cc799bc6a595d4fb8a7da2ccbe64c77abaf7e39070e52
```

### Prefill path + O_DIRECT

Allocation `487`，冷 cache，完整 `32K / agent4 / rep1`：

```text
TASK_NAME=32k-a4-r1-prefill-odirect-prefixfix
DUALPATH_L3_READ_PATH=prefill
DUAL_PATH_O_DIRECT_READ=1
```

结果目录：

```text
/data/home/yongxiao/mydata/test/487/32k-a4-r1-prefill-odirect-prefixfix
```

- `104/104` turns、`4/4` trajectories 完成，无错误。
- wall time `236.89s`，mean TTFT `1.854s`；旧镜像对应值为 `11.664s`。
- `DUALPATH_L3_HIT_QUERY` 独立 rid `104/104`，consensus `104/104`，无正式请求 skip。
- Prefill `#cached-token` 合计 `1,939,200`，`#new-token` 合计 `112,768`，复用率
  `94.5%`；旧镜像 Prefill 复用率为 `0%`。
- `prefill_computation_seconds` 均值 `0.310s`，降到预期的 0.5s 以下。
- `read_tokens=0` 仅为四个 agent 的首轮冷 cache；非首轮为 0 的请求数为 0。
- TTFT V3 `read_tokens` 与 L3 consensus hit 逐请求一致，accounting error 最大值为 0。

### Decode path + O_DIRECT

由于同一 allocation 内重启服务时旧 Decode launcher step 未及时释放，Slurm 报过
`step creation temporarily disabled`。这不是镜像或代码错误；随后使用独立 allocation
`488` 完成了冷 cache Decode 验收：

```text
TASK_NAME=32k-a4-r1-decode-odirect-prefixfix
DUALPATH_L3_READ_PATH=decode
DUAL_PATH_O_DIRECT_READ=1
```

结果目录：

```text
/data/home/yongxiao/mydata/test/488/32k-a4-r1-decode-odirect-prefixfix
```

- `104/104` turns、`4/4` trajectories 完成，无错误。
- wall time `319.87s`，mean TTFT `5.772s`。
- Decode `DUALPATH_L3_HIT_QUERY` 和 consensus 均为 `104/104`，无正式请求 skip。
- Decode L3 read tokens `1,939,200`，Prefill L3 read tokens 为 0，路径选择正确。
- `read_tokens=0` 仅为四个 agent 的首轮冷 cache，非首轮为 0 的请求数为 0。
- Decode 平均 Lustre/NFS read `2.244s`，Host-to-device `0.055s`，dispatch `0.014s`。
- accounting error 最大值为 0。

两组图表分别位于：

```text
/data/home/yongxiao/dual-path/results/latency-benchmark/32k-a4-r1-prefill-odirect-prefixfix
/data/home/yongxiao/dual-path/results/latency-benchmark/32k-a4-r1-decode-odirect-prefixfix
```

Allocation `487` 和 `488` 均已释放。

### 第二轮：queue wait / offload 修复验收

已从原始基底重建并测试新镜像：

```text
/data/home/yongxiao/mydata/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-queue-wait-offload-fix.sqsh
SHA256: b6658d37e44588cb7e56ba61ea61ae49ee7df89b8a49e0bd3317f220b8f335c0
构建 job: 489
```

四个冷 cache O_DIRECT 测例均完成并生成图表：

| path | agents | alloc | turns | wall | mean TTFT | L3 read tokens |
|---|---:|---:|---:|---:|---:|---:|
| Prefill | 1 | 490 | 26/26 | 190.03s | 1.348s | 485,632 |
| Prefill | 4 | 491 | 104/104 | 235.11s | 1.800s | 1,939,200 |
| Decode | 1 | 492 | 26/26 | 269.97s | 4.758s | 485,632 |
| Decode | 4 | 493 | 104/104 | 328.25s | 6.073s | 1,939,200 |

四组均满足：

- 所有 trajectories 和 turns 完成，无 replay error 或卡死。
- `DUALPATH_L3_HIT_QUERY` / consensus 独立 rid 等于总 turns：agent1 为 26/26，
  agent4 为 104/104。
- 正式请求没有 `DUALPATH_L3_PREFETCH_SKIP`；非首轮 `read_tokens=0` 为 0。
- 四组图表均已写入对应的 `results/latency-benchmark/32k-*-odirect-queuefix/` 目录。

`queue_wait` 已正确出现在图表和汇总中。agent1 的 overlap accounting 为 0；agent4
各有 1 个轻微超容差轮次：Prefill 最大 `0.0095s`，Decode 最大 `0.0124s`，其余轮次
均为 0。`misc` 已降到约 `0.07--0.09s`。

本轮还暴露出 §5 的 offload 修复尚未生效：

- Prefill agent1/agent4 的 `offload_start` 仍全部为 0，分别写入
  `2,072,064` / `8,274,944` tokens per-rank aggregate，仍是全上下文写放大。
- Decode agent1/agent4 的正向 `offload_start` 为 `100/104`、`400/416`，写入量为
  `129,536` / `518,144` tokens per-rank aggregate，说明 Decode path 原有边界正常。
- 因此本轮不能把 §5 标记为已验证；需要继续检查 Prefill path 下 Decode 接收的
  `cached_tokens` 是否在 offload manager 计算边界前已经可见，以及
  `dualpath_prefill_l3_prefix_trusted` 的条件是否在 Decode 进程成立。

另外，四个 `rounds.jsonl` 中 Decode path 的 `cached_tokens` 合计仍为 0，虽然 Decode
L3 read tokens 正确为 `485,632` / `1,939,200`；这说明报告口径修复也尚未达到文档中
预期的数值，需要单独追踪 client 可见字段的生成链路。图表和原始日志保留如下：

```text
/data/home/yongxiao/dual-path/results/latency-benchmark/32k-a1-r1-prefill-odirect-queuefix
/data/home/yongxiao/dual-path/results/latency-benchmark/32k-a4-r1-prefill-odirect-queuefix
/data/home/yongxiao/dual-path/results/latency-benchmark/32k-a1-r1-decode-odirect-queuefix
/data/home/yongxiao/dual-path/results/latency-benchmark/32k-a4-r1-decode-odirect-queuefix
```

## 8. 注意事项

- **32K 实验必须显式设置 `CONTEXT_LENGTH=32768`**，默认 `4096` 会在第 2 或第 3 轮因
  prompt 超限失败
- `HICACHE_SIZE_GB` 必须大于 device KV pool，否则 SGLang 的
  `assert host_pool.size > device_pool.size` 会在启动时失败。默认 40 在 A100 40GB 与
  H100 80GB 上都成立；设为 0 则退回 `HICACHE_RATIO`
- `DUALPATH_L3_READ_PATH`、`DUAL_PATH_O_DIRECT_READ`、`HICACHE_SIZE_GB` 都是 run-time
  参数，不要放进 allocation 节点申请逻辑
- 客户端 `rounds.jsonl` 的 `cached_tokens` 历史上不可靠，两个已知问题都已修：§2 之前
  prefill path 全为 0，§5 之前 decode path 虚高 2 倍。修复后两条路径都应等于 L3 read
  tokens。**验证前仍以 `DUALPATH_TTFT_V3.read_tokens` 和 `Prefill batch` 的
  `#cached-token` 为准**，它们是引擎直出、未经跨侧聚合的数值
- 比对日志务必按完整 rid 匹配。不同 trajectory 的同一 turn 经常有完全相同的
  `requested_pages` 和 `local_hit_tokens`，按字段值匹配极易串号——上一轮排查就因此把
  根因判断反了
- 排查 L3 问题的第一步：数 `DUALPATH_L3_HIT_QUERY` 的独立 rid 数是否等于总轮数。不相等
  说明部分请求没发起 prefetch，看 `DUALPATH_L3_PREFETCH_SKIP`；相等则问题在读回之后的
  消费环节（本文 §2 就属于后者）
- Prefill/Decode 必须在同一 allocation 内运行，共享
  `/data/home/yongxiao/mydata/test/<ALLOC_JOB_ID>/pd-servers/kv-cache/file`
- 不要在 engine 仍运行时清理该 cache directory
- 仓库根目录与 `mydata/test/` 下的构建与实验日志属于实验产物，不要当作源码改动回滚
- 改 patch 脚本时注意 `replace_once()` 要求匹配数恰好为 1，且
  `patch_*()` 函数通常有 "Already patched" 分支。新增改动要在两条分支里都生效，否则从
  partial patch 镜像重建时会漏掉。现有范式见 `patch_scheduler_prefetch_namespace()` 与
  `patch_hiradix_prefetch_skips()`
- 改完 patch 脚本先在本地用 `--site-packages` 跑一遍再提交 Slurm 构建，几秒就能验证，
  比排队等构建快得多

## 9. 第三轮：offload guard / cached_tokens 修复验收

2026-09-19 根据 Claude 提供的两个根因重建镜像。构建使用原始基底
`/data/home/yongxiao/mydata/images/sglang:26.01-py3.sqsh`，Slurm job 为 `494`：

```text
/data/home/yongxiao/mydata/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-offload-cached-fix-20260919.sqsh
SHA256: a6f5238997e03e4d7d4ad76e8bc36710193b9036682069463d6118e76df55d53
```

构建期断言全部通过，包括：

- Prefill 不再写入 `req.cached_tokens = hit_tokens`，改为由上游统一计数；
- Decode offload guard 不再读取 Prefill 专属的
  `DUALPATH_HICACHE_READ_ONLY`；
- `queue_wait` 和 O_DIRECT 相关断言继续通过。

四个冷 allocation 的完整 `32K / rep1 / O_DIRECT` 测试均成功完成：

| path | agents | allocation | turns | wall | mean TTFT | L3 read | cached_tokens | offload positive | offload tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prefill | 1 | 497 | 26/26 | 179.45s | 1.331s | 485,632 | 485,632 | 100/104 | 129,536 |
| Prefill | 4 | 498 | 104/104 | 219.57s | 1.874s | 1,939,200 | 1,939,200 | 400/416 | 518,144 |
| Decode | 1 | 499 | 26/26 | 230.33s | 3.223s | 485,632 | 485,632 | 100/104 | 129,536 |
| Decode | 4 | 500 | 104/104 | 345.78s | 6.771s | 1,939,200 | 1,939,200 | 400/416 | 518,144 |

`offload positive` 和 `offload tokens` 按 Decode log 的 TP-rank 记录汇总；每个正式
request 有 4 个 TP 记录，首轮冷 cache 不产生正向 offload，故 agent1 为 `25*4=100`
条、agent4 为 `100*4=400` 条。写入量已从上一轮 Prefill 的全上下文放大降到与 Decode
路径一致的量级。

本轮 `rounds.jsonl` 的 `cached_tokens` 已恢复正确：两条 path 都与对应的
`DUALPATH_TTFT_V3.read_tokens` 相等，说明重复计数修复生效。

图表输出：

```text
/data/home/yongxiao/dual-path/results/latency-benchmark/32k-a1-r1-prefill-odirect-newfix
/data/home/yongxiao/dual-path/results/latency-benchmark/32k-a4-r1-prefill-odirect-newfix
/data/home/yongxiao/dual-path/results/latency-benchmark/32k-a1-r1-decode-odirect-newfix
/data/home/yongxiao/dual-path/results/latency-benchmark/32k-a4-r1-decode-odirect-newfix
```

补充：allocation `495` 是一次手工编排失败，客户端漏传 `CONTEXT_LENGTH=32768`，在
第 2 轮因默认 `4096` 上限退出；它没有进入正式结果统计。随后补齐参数后，`497--500`
四组均完整通过。
