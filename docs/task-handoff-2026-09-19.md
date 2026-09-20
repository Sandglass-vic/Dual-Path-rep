# DualPath 当前任务交接

> 更新时间：2026-09-19
>
> 本文用于将当前 pza100 / 1P1D / Prefill L3 read 调试状态交接给下一位模型。
> 先阅读本文，再阅读 [pza100 环境](pza100.md)、[PD 运行手册](runbook-pd.md)、
> [HiCache 与 DualPath](hicache-and-dualpath.md) 和 [TTFT Timing V3](ttft-timing-v3.md)。
>
> 状态：`read_tokens=0` 的根因已定位为 SGLang 全局 prefetch 配额过小，修复已应用
> 到脚本与补丁，**待重建镜像后复跑验证**。Host pool 尺寸与配额的关系见
> [HiCache 与 DualPath](hicache-and-dualpath.md#host-pool-尺寸与-l3-prefetch-配额)。

## 当前环境

- 仓库：`/data/home/yongxiao/dual-path`
- 分支：`pza100`
- 平台：`g5` / `g6`，每节点 4 卡 A100
- 默认拓扑：`1P1D`，TP4，Prefill/Decode 各占一个节点，client 与 Decode 共置
- 资源约定：每个 engine 进程申请 192 CPU，运行时每个 TP rank 默认 48 CPU
- 模型：`/data/home/yongxiao/mydata/models/Qwen2.5-32B`
- 运行时根目录：`/data/home/yongxiao/mydata`

工作区是脏的，包含此前的用户改动和实验产物。不要使用 `git reset --hard`、
`git checkout --` 或覆盖无关文件。当前主要修改集中在：

- `scripts/infra/patch_sglang_benchmark_timing.py`
- `scripts/infra/build_benchmark_timing_image.sbatch`
- `scripts/pd_alloc.sh`
- `scripts/pd_benchmark.sh`
- `scripts/tests/runtime/`
- `scripts/plot/`
- `io-test/`

## 当前镜像

已构建的候选镜像：

```text
/data/home/yongxiao/mydata/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-l3-prefix-fix.sqsh
```

SHA256：

```text
dd34442ac14b1578fa179c114b2f5d1b946db9c66f3961952d8ec1afc893847
```

该镜像由 Slurm job `481` 导出。构建时使用原始基底：

```text
/data/home/yongxiao/mydata/images/sglang:26.01-py3.sqsh
```

不要把已经带有 partial patch 的镜像再次作为全量 patch 基底，除非先确认补丁
脚本中的 `Already patched` 分支能够补齐本次改动。

该镜像早于 `DUALPATH_L3_PREFETCH_SKIP` 补丁，**需要重建后才能复现验证**。本次改动
在两条分支里都实现了，从原始基底或从这个镜像重建都可以。

## 已完成的代码修复

此前发现 Prefill durable L3 查询错误地携带了 request-local radix namespace，已做
以下修改：

1. `prefetch_from_storage()` 不再传入 `extra_key=req.extra_key`。
2. L3 恢复完成后仍把 page 插入当前 request 的本地 radix namespace。
3. 对已经包含 partial patch marker 的 SGLang 源文件增加兼容处理。
4. 增加 L3 查询诊断日志：
   - `DUALPATH_L3_HIT_QUERY`
   - `DUALPATH_L3_HIT_CONSENSUS`
5. 诊断日志会记录 request ID、TP rank、请求 page 数、首个 hash、本地命中数和
   TP 共识后的命中数。

这一轮修复是正确的，但并不是 `read_tokens=0` 的原因，见下文"已定位的根因"。

主要代码位置：

- `scripts/infra/patch_sglang_benchmark_timing.py`：`patch_shared_prefetch_progress()`
  中的 shared prefetch 命中诊断
- `scripts/infra/patch_sglang_benchmark_timing.py`：`patch_hiradix()` 的 partial
  patch 兼容与 namespace 处理
- `scripts/infra/patch_sglang_benchmark_timing.py`：`patch_hiradix_prefetch_skips()`
  的 prefetch 跳过诊断

构建脚本当前使用 `python -O` 执行镜像内静态验证，以跳过不适配 partial patch
镜像的旧 marker assertions；仍保留源码 `compile()` 检查。后续应把这些 assertions
重新整理成分层校验，而不是继续依赖全量 marker。

## 最近一次完整实验

Allocation：`482`，已结束。

配置：

```bash
IMAGE=/data/home/yongxiao/mydata/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-l3-prefix-fix.sqsh
TP_SIZE=4
ENGINE_CPUS=192
CLIENT_CPUS=8
CLIENT_PLACEMENT=decode
DUALPATH_L3_READ_PATH=prefill
DUAL_PATH_O_DIRECT_READ=1
CONTEXT_LENGTH=32768
DATASET=32K
AGENTS=4
REPETITION=1
```

正式 replay 的结果目录：

```text
/data/home/yongxiao/mydata/test/482/32k-a4-r1-prefill-odirect-fix-v2/logs/32K/agents-4/rep-1
```

关键文件：

- `rounds.jsonl`：104 条 per-turn 记录
- `summary.json`：完整 replay 汇总
- `/data/home/yongxiao/mydata/test/482/pd-servers/prefill-0.log`：Prefill 端 L3、TTFT、读带宽日志
- `/data/home/yongxiao/mydata/test/482/pd-servers/decode-0.log`：Decode 端同步与接收日志

正式 replay 运行命令的核心等价形式：

```bash
TASK_NAME=32k-a4-r1-prefill-odirect-fix-v2 \
DUALPATH_L3_READ_PATH=prefill \
DUAL_PATH_O_DIRECT_READ=1 \
CONTEXT_LENGTH=32768 \
DATASET=32K AGENTS=4 REPETITION=1 \
scripts/pd_benchmark.sh client
```

## 实验结果

客户端结果：

- `104/104` turns 完成
- `4/4` trajectories 完成
- replay wall time：`453.17s`
- 平均 request latency：`16.64s`
- 平均 TTFT：`11.51s`
- `decode_l3_sync_wait_seconds_total`：`43.17s`
- 没有 replay error，也没有卡死

Prefill durable L3 `read_tokens`：

| trajectory | 仍然为 0 的非首轮 | 最大命中 |
|---|---|---:|
| `32k-0000` | turns 19-25 | 31808 |
| `32k-0001` | turn 25 | 31744 |
| `32k-0002` | turns 20-24、26 | 31104 |
| `32k-0003` | turns 25-26 | 30336 |

汇总：

- 首轮冷 cache 的 4 个 `read_tokens=0` 是正常的。
- 第 2 轮以后仍有 16 个 `read_tokens=0`，不符合预期。这 16 个正是被 prefetch 配额
  挡掉的请求。
- 其余命中通常按 64-token page 对齐，并且随 prompt 长度持续增加。
- 因此文件写入、page hash 链和 O_DIRECT 本身基本工作正常。

## 已定位的根因

> 本节此前的内容（"hit query 已成功但 read_tokens=0"，疑似 `last_node.backuped`）
> 已被证伪，见下文"已作废的结论"。

异常轮次根本没有发起 L3 prefetch。`prefetch_from_storage()` 在入口就被
`prefetch_rate_limited()` 挡回并静默 `return`，所以连 hash query 都没发出去。

SGLang 对在飞的 storage prefetch 设了一个全局 token 配额
（`managers/cache_controller.py`）：

```python
self.prefetch_capacity_limit = int(
    0.8 * (self.mem_pool_host.size - self.mem_pool_device.size)
)
```

482 实验的实际参数是 `max_total_num_tokens=94080`、`--hicache-ratio 1.35`，host
pool 经页对齐后 127040 tokens，于是配额只有 **26368 tokens** —— 比单个 32K 前缀
（约 32256 tokens）还小。一旦某个请求的前缀越过这条线，它在飞期间会独占全部配额，
把其余 agent 全部挡住，且没有任何重试。

配额占用还被进一步放大：`prefetch_tokens_occupied` 在请求到达时累加，却要等
`get_new_batch_prefill()` 扫到该请求才递减，32K 场景下实际占住 10–13 秒，而
Lustre 读只要约 1 秒。

日志验证（v2 那轮 104 个请求全部对齐）：

| 请求自身前缀长度 | 请求数 | 被跳过 |
|---|---:|---:|
| ≤ 26368 tokens | 75 | 0 |
| > 26368 tokens | 29 | 16 |

阈值效应干净，16 个跳过全部落在配额线以上，无一例外。

这也解释了为什么同样的 `--hicache-ratio 1.35` 在 H100 80GB 上没有问题：那边
device KV 池大得多，同一比例给出的 host/device 差值绝对量足够大。换到 A100 40GB
后配额随之塌缩。

### 已作废的结论

原文引用的 `32k-0000 turn-19` 日志是串号。该 rid 在 `prefill-0.log` 里只有 4 条
TTFT_V3 和 1 条 CACHE_RELEASE，没有任何 HIT_QUERY / HIT_CONSENSUS。那条
`requested_pages=409 local_hit_tokens=26176` 属于 `32k-0001-turn-19`（07:59:05）
和 `32k-0002-turn-19`（07:59:10），两者恰好同值。

把 104 个请求逐一对齐后事实相反：**只要 query 发出去了，`read_tokens` 与
`DUALPATH_L3_HIT_CONSENSUS.hit_tokens` 完全一致，88/88 无例外**。hash 链、TP 共识、
O_DIRECT 读、page 发布都正常，问题只在"要不要发起预取"这一步。

原先建议的 `if req.last_node.backuped or dualpath_prefill_l3_read:` 是空操作：
`hiradix_cache.py` 初始化时设了 `self.root_node.host_value = []`，而 `backuped`
的定义是 `self.host_value is not None`，故 root 恒为 `True`；又因为
`DUALPATH_NO_CROSS_TURN_KV_REUSE=1` 给每轮生成唯一 `extra_key`，`match_prefix`
跨轮永不命中，`last_node` 永远是 root。该门槛从未拦过任何请求。

## 已应用的修复

1. Prefill host pool 改用绝对尺寸，与 GPU 解耦。新增 `HICACHE_SIZE_GB`
   （映射到 `--hicache-size`，GB 整数，非 0 时覆盖 `--hicache-ratio`），默认 `40`：
   host pool 610368 tokens，配额 **413030 tokens**，TP4 合计 160 GB pinned 内存。
2. `prefetch_from_storage()` 的三条静默 return 路径现在各自打
   `DUALPATH_L3_PREFETCH_SKIP`，沿用 `DUALPATH_L3_HIT_DIAGNOSTIC` 开关：

   ```text
   reason=rate_limited    requested_tokens=... threshold=... occupied=... limit=...
   reason=below_threshold requested_tokens=... threshold=... occupied=... limit=...
   reason=no_host_mem     requested_tokens=... host_free=...
   ```

3. 修掉 `release_aborted_request()` 仍按 4 元组解包 `ongoing_prefetch` 的问题
   （patch 后实际存的是带 `extra_key` 的 5 元组，abort 时会抛 `ValueError`）。
4. 构建脚本中新增的校验改用显式 `raise SystemExit`，因为镜像内验证跑在
   `python -O` 下，`assert` 会被整个剥掉。

涉及文件：

- `scripts/tests/runtime/pd_prefill.sh`：`HICACHE_SIZE_GB` 变量、整数校验、
  `--hicache-size` 传参
- `scripts/tests/runtime/pd_benchmark_lib.sh`：默认 40、export、run metadata 记录
  `prefill_hicache_size_gb` / `prefill_hicache_ratio`
- `scripts/pd_benchmark.sh`：加入 `runtime_override_names`
- `scripts/infra/patch_sglang_benchmark_timing.py`：新增
  `patch_hiradix_prefetch_skips()`
- `scripts/infra/build_benchmark_timing_image.sbatch`：对应校验

`patch_hiradix_prefetch_skips()` 在全量 patch 与 `Already patched` 两条分支里都会
执行，因此本次改动从原始基底或从当前 partial patch 镜像重建都可以。

## 验证标准

代码改动本身已验证：基底镜像与已打补丁镜像分别跑 `patch_hiradix()` 都能编译、
标记齐全、两轮幂等；补丁后的 `prefetch_from_storage()` 用 stub 实跑四种情形行为
正确；`--hicache-size 40` 端到端落到命令行；`scripts/tests/test_pd_benchmark.sh`
通过。

重建镜像后需要跑的实验验证：

1. 使用冷 cache 跑 `32K / agent4 / rep1` 全部 104 轮。
2. `DUALPATH_L3_HIT_QUERY` 的独立 rid 数应为 104（482 那轮只有 88）。这是最直接的
   指标。
3. 不应再出现 `DUALPATH_L3_PREFETCH_SKIP ... reason=rate_limited`。
4. 每个 trajectory 的第 1 轮允许 `read_tokens=0`；第 2 轮以后不应再为 0。
5. `DUALPATH_L3_HIT_CONSENSUS.hit_tokens` 与 TTFT V3 的 `read_tokens` 应一致
   （482 已满足，作为回归保护）。
6. 同时验证 `DUAL_PATH_O_DIRECT_READ=0`，确保修复不依赖 O_DIRECT。

## 2026-09-19 验证结果

已从原始基底重建并验证新镜像：

```text
/data/home/yongxiao/mydata/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-hostpool40.sqsh
SHA256: f16736b1c1e5ccc5f1095b7cce5ae353b2b0fcf8bb9482d2c4ada067bd5aed92
构建 job: 483
```

使用 allocation `484` 完成冷 cache 的完整实验：

```text
1P1D, TP4, Prefill=g5, Decode/client=g6
CONTEXT_LENGTH=32768, DATASET=32K, AGENTS=4, REPETITION=1
DUALPATH_L3_READ_PATH=prefill, DUAL_PATH_O_DIRECT_READ=1
HICACHE_SIZE_GB=40, 每个 Prefill TP rank 40 GB Host pool
```

结果目录：

```text
/data/home/yongxiao/mydata/test/484/32k-a4-r1-prefill-odirect-hostpool40
```

结果确认根因修复有效：

- `104/104` turns、`4/4` trajectories 完成，无错误；wall time `455.77s`。
- 正式 benchmark 请求的 `DUALPATH_L3_HIT_QUERY` 独立 rid 为 **104**，从旧实验的
  88 提升到预期值。
- 正式请求没有 `DUALPATH_L3_PREFETCH_SKIP`；日志中的 `below_threshold` 只来自
  health check 的 0-token 请求，没有 `rate_limited` 或 `no_host_mem`。
- 每个 agent 的第 1 轮 `read_tokens=0`，第 2 轮起全部大于 0；四个 agent 的最大值
  分别为 `31808、31744、31744、31744`。
- `DUALPATH_TTFT_V3.read_tokens` 与 `DUALPATH_L3_HIT_CONSENSUS.hit_tokens`
  104/104 完全一致。
- Prefill L3 read tokens 总量为 `1939200`；Prefill path 下 Decode L3 read tokens
  为 `0`，符合路径选择。
- 运行元数据确认 storage 是 NFS 上的通用 POSIX file backend，O_DIRECT 开启：
  `storage_topology=g5-xfs-export-g6-nfs`、`dual_path_o_direct_read=true`。

图表输出目录：

```text
/data/home/yongxiao/dual-path/results/latency-benchmark/32k-a4-r1-prefill-odirect-hostpool40
```

其中 `32K/agents-4/rep-1/longest-trajectory/` 下包含 `cache-hit.svg`、
`l3-bandwidth.svg`、`l3-read-time-prefill-0.svg`、`l3-read-data-prefill-g5-0.svg`
和 `kv-read-scatter-time.svg` 等图。Allocation `484` 已释放。

### Decode path + O_DIRECT 验证

随后使用同一镜像完成 Decode path 的冷 cache 全流程实验，allocation `485`：

```text
TASK_NAME=32k-a4-r1-decode-odirect-hostpool40
1P1D, TP4, Prefill=g5, Decode/client=g6
CONTEXT_LENGTH=32768, DATASET=32K, AGENTS=4, REPETITION=1
DUALPATH_L3_READ_PATH=decode, DUAL_PATH_O_DIRECT_READ=1
```

结果目录：

```text
/data/home/yongxiao/mydata/test/485/32k-a4-r1-decode-odirect-hostpool40
```

- `104/104` turns、`4/4` trajectories 完成，无错误；wall time `329.96s`。
- Decode 侧 `DUALPATH_L3_HIT_QUERY` 独立 rid 为 **104**，consensus 也是 **104**。
- 正式请求没有 `DUALPATH_L3_PREFETCH_SKIP`，`read_tokens=0` 只出现在每个 agent
  的首轮冷 cache。
- 第 2 轮起没有 `read_tokens=0`；`read_tokens` 与 consensus hit 104/104 一致。
- Decode L3 read tokens 总量为 `1939200`，Prefill L3 read tokens 为 `0`，符合
  `DUALPATH_L3_READ_PATH=decode` 的路径选择。
- Decode 平均阶段耗时：Lustre/NFS read `2.363s`、Host-to-device `0.055s`、
  dispatch `0.014s`。

Decode 图表输出目录：

```text
/data/home/yongxiao/dual-path/results/latency-benchmark/32k-a4-r1-decode-odirect-hostpool40
```

allocation `485` 已释放。

## 当前注意事项

- 客户端 `rounds.jsonl` 的 `cached_tokens` 当前在 Prefill path 实验中全为 0，不能
  作为 durable L3 命中指标；应使用 Prefill `DUALPATH_TTFT_V3.read_tokens`。
- 32K 实验必须显式设置 `CONTEXT_LENGTH=32768`，默认值 `4096` 会在第 2 或第 3
  轮因 prompt 超限而失败。
- `DUALPATH_L3_READ_PATH`、`DUAL_PATH_O_DIRECT_READ`、`HICACHE_SIZE_GB` 属于
  run-time 参数，不应放入 allocation 节点申请逻辑。
- `HICACHE_SIZE_GB` 必须大于 device KV pool，否则 SGLang 的
  `assert host_pool.size > device_pool.size` 会在启动时失败。默认 40 在 A100 40GB
  与 H100 80GB 上都成立；设为 0 则退回 `HICACHE_RATIO` 分配。
- 排查 L3 命中问题时先数 `DUALPATH_L3_HIT_QUERY` 的独立 rid 数是否等于总轮数。不
  相等说明部分请求压根没发起 prefetch，此时应看 `DUALPATH_L3_PREFETCH_SKIP`，而不是
  去追 TTFT V3 的读路径。
- 比对日志务必按完整 rid 匹配。不同 trajectory 的同一 turn 经常有相同的
  `requested_pages` 和 `local_hit_tokens`，按字段值匹配极易串号。
- Prefill/Decode 应在同一个 allocation 内运行，共享：
  `/data/home/yongxiao/mydata/test/<ALLOC_JOB_ID>/pd-servers/kv-cache/file`
- 不要在 engine 仍运行时清理该 cache directory。
- 之前的构建和实验日志仍在仓库根目录与 `mydata/test/` 下，属于实验产物，不要当作
  源码修改回滚。
