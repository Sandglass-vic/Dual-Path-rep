# HiCache L3 与 DualPath 近似实现

> 返回 [文档索引](README.md)

## Phase 4：HiCache + Lustre

### 目标语义：与论文 DualPath 对齐——跨 request 不复用 L1/L2

论文 §4.1：PE/DE **staging buffer** 与 **GPU buffer** 在单次 request 的加载/
计算/传输后释放（Decode 在 H2D 完成后释放 CPU buffer）；跨轮 KV 复用依赖
**外部存储（本复现为 Mooncake Store / Lustre DFS）**，而不是把上一轮 KV 留在
本地 L1/L2 供下一轮直接命中。

本项目逻辑层次：

```text
L1：GPU KV，仅服务当前 request 的计算与短期调度（用完即释放）
L2/host staging：容量受控的搬运缓冲，不是跨 turn 命中层
L3：Mooncake Store，负责跨轮次前缀复用
```

期望的第二轮数据流：

```text
1. 请求携带完整累积上下文到 Prefill。
2. Prefill 本地 L1/L2 无上一轮前缀（已 flush/释放）→ 查询 L3。
3. L3 将命中的 Full Blocks 加载到 Prefill staging → GPU。
4. Prefill 只计算 append/cache-miss tokens 的 KV。
5. Prefill 将“L3 命中的 KV + 新计算的 KV”组成完整 prompt KV，P→D 传给 Decode。
6. Decode 生成并写回 L3；本 request 的 staging/GPU 缓冲随后释放。
```

### SGLang 近似实现

SGLang HiCache **无法**关闭 L2 pool（`--hicache-size 0` 仍按 ratio 分配，且
强制 L2>L1）。正式路径用客户端近似论文“用完即放”：

1. 每轮 generate **之后**对 Prefill 调用 `/flush_cache`，清除 L1/L2 与 radix
   metadata；
2. **不**调用 `/clear_hicache_storage_backend`，保留 Mooncake Store；
3. 可选 `STORAGE_SETTLE_SECONDS`（默认 1s），等待 Decode→Store 发布可见；
4. `cache_flush_seconds` / `storage_settle_seconds` 单独记账，**不计入**
   `turn_wall_seconds` / `trajectory_jct`；batch `wall_seconds` 仍含真实墙钟
   （含 flush/settle）。

环境变量（`pd_benchmark_lib.sh` 默认）：

```bash
FLUSH_LOCAL_CACHE_BETWEEN_TURNS=1
STORAGE_SETTLE_SECONDS=1
```

Job `65694` 未 flush，prefix hit 来自 L1/L2，`l3_timing_coverage=0`，**不得**
当作论文 L3 路径结果。

### 当前 SGLang 0.5.7 的限制

当前镜像源码的 host pool 分配逻辑为：

```python
if host_size > 0:
    size = host_size
else:
    size = device_pool.size * host_to_device_ratio

assert host_pool.size > device_pool.size
```

所以：

- `--hicache-size 0` 表示按 `--hicache-ratio` 分配 L2，不是关闭 L2；
- 当前协议强制 L2 大于 L1；
- 仅靠命令行无法得到严格 L3-only；正式路径依赖 **逐轮 `/flush_cache`**。

多 agent 并发时 flush 会清空整机 Prefill 本地层（与论文“每 request 独立从
存储加载”一致），但可能与 in-flight 请求竞态；若出现 flush 失败，改为更小
并发或排查 SGLang idle 要求。

因此必须同时存在两种独立能力：

论文中的两项 Mooncake 能力必须区分：

```text
Mooncake Transfer Engine: PD 阶段把 KV 从 Prefill 发送到 Decode
Mooncake Store:           HiCache 的共享 L3，保存和恢复多轮 KV
```

HiCache 不是独立的存储系统。它是 SGLang 内部的 KV-cache 分层管理器，负责：

- 用 RadixTree/prefix metadata 判断哪些 token 已命中；
- 在 L1、L2、L3 之间组织 KV block；
- 从 L3 prefetch 命中 block；
- 将新 KV 异步 backup/offload 到 L3；
- 向 scheduler 报告可复用 token，避免 Prefill 重算命中部分；
- 在 PD 模式中协调 Prefill 恢复、Decode offload 与 KV transfer。

实际保存 L3 数据的是 Mooncake Store、file、HF3FS 等 storage backend；当前 PD 使用的 Mooncake Transfer Engine 只负责 PE→DE 传输，并不自动启用 HiCache 或 L3。

### 当前 SGLang 0.5.7 的限制

当前镜像源码的 host pool 分配逻辑为：

```python
if host_size > 0:
    size = host_size
else:
    size = device_pool.size * host_to_device_ratio

assert host_pool.size > device_pool.size
```

所以：

- `--hicache-size 0` 表示按 `--hicache-ratio` 分配 L2，不是关闭 L2；
- 当前协议强制 L2 大于 L1；
- 仅靠现有命令行参数无法得到严格的 L3-only HiCache；
- 文档和实验中不得把 `--hicache-size 0` 宣称为“禁用 L2”。

本项目当前接受 SGLang 原生 L1/L2/L3 **池子**存在，但正式性能路径通过逐轮
`/flush_cache` 禁止跨 turn 的 L1/L2 命中，使可复用前缀来自 Mooncake Store。
需要额外证明时仍可用完全重启 Prefill 做隔离对照。

文档必须区分：

- **论文对齐路径**：flush L1/L2，跨 turn 走 L3（正式 benchmark 默认）；
- **层次缓存顺风车**：不 flush（如 Job `65694`），`cached_tokens` 主要来自 L1/L2。

### 测试 KV 数据清理

所有 smoke/replay 使用包含 `$SLURM_JOB_ID` 的独立 `CACHE_DIR`。脚本创建专用
marker，并在 `EXIT`、`INT`、`TERM` trap 中先停止服务，再检查 marker 和删除
整个本 Job KV 目录。成功、验收失败或作业取消都会释放 `.bin` 空间；日志、
response、metrics 和 replay JSON 保留。不得清理模型目录、数据集或没有本 Job
marker 的用户自定义目录。

### L3 backend 启动与配置

安装 `mooncake-transfer-engine-cuda13` 后应同时验证 `mooncake.engine` 和 `mooncake.store`。Mooncake Store 至少需要 `mooncake_master`；官方最小启动方式为：

```bash
mooncake_master --enable_http_metadata_server=true
```

SGLang 侧使用 `MOONCAKE_MASTER=<master-host>:50051` 和
`--hicache-storage-backend mooncake`。Decode 要发布新生成 KV 时增加
`--disaggregation-decode-enable-offload-kvcache`。

正式 backend 已从 SGLang `file` 切换为 Mooncake `0.3.13.post1` 的
descriptor-based DFS：

```text
MOONCAKE_DFS_FS_ADAPTER=posix
MOONCAKE_DFS_ROOT_DIR=<job-specific Lustre path>
MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=distributed_storage_backend
replica_num=1
dfs_replica_num=1
```

`MOONCAKE_OFFLOAD_FILE_STORAGE_PATH` 仍是 FileStorage 初始化必填项，但实际 DFS
数据写入 `MOONCAKE_DFS_ROOT_DIR`。完整配置和 Job `65548` 跨节点验证见
[Mooncake Store 与 Lustre POSIX DFS](mooncake-store.md)。

以下 `file` backend 只保留为历史最小正确性基线，不再用于正式性能实验：

最小 L3 正确性验证可以先使用 `file` backend 并把目录指向 Lustre。注意该配置仍会创建 SGLang 强制要求的 L2；它只用于证明 L3 hit 数据流：

```bash
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$CACHE_DIR/file-backend"
export SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=320GiB

python -m sglang.launch_server \
  ... \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend file \
  --hicache-storage-prefetch-policy wait_complete
```

旧 `file` backend 直接覆盖 `.bin`，共享 Lustre 上曾发生并发 `Short read`。
Mooncake Store 通过 `PutStart/PutEnd` 在完整对象写完后发布 metadata，已替代
Populate/Measured 与 scratch 目录规避方案。

Decode 侧还需启用当前版本对应的异步 offload 参数（官方新版本示例为 `--disaggregation-decode-enable-offload-kvcache`），使每轮新 KV 落入 L3；Prefill 侧启用 HiCache，从 L3 恢复前缀。以 `--help` 的实际参数名为准。

验收条件：

- 早期轮次产生 L3 写流量，并记录实际 offload admission/eviction 行为；
- 在已确认写回完成后的后续轮次提交完整累积上下文时出现可验证的 L3 hit；
- Prefill 日志/metrics 报告 L3 hit tokens，而不是重新计算全部上下文；
- 重启 Prefill、消除 L1/L2 状态后仍可从 L3 命中，证明命中不是本地缓存；
- Prefill compute tokens 等于 append/cache-miss tokens；
- Prefill→Decode 传输字节包含 hit KV 与 miss KV；
- Decode 完成后，新 KV 被写回 L3；
- 保存 L1/L2/L3 hit、KV 读写字节、Lustre 吞吐和 TTFT 分解。

上述观测项的字段、当前自动采集范围和未接入的硬件计数器统一见
[指标规范](metrics.md)。

### 已验证：单节点 Lustre L3 重启命中

2026-09-01 使用 Qwen2.5-32B、TP=8 和 HiCache `file` backend 完成最小
持久化验证：

```text
Slurm Job：64882，COMPLETED，ExitCode=0:0，耗时 3m29s
节点：dgx-gaia-38
后端：SGLang HiCache file
目录：/mnt/lustre/users/sezhou/dual-path/kv-cache/hicache-file-smoke/64882/
page size：64
write policy：write_through
prefetch policy：wait_complete
脚本：scripts/pd_1p2d_benchmark.sbatch（原 hicache/l3 smoke 已并入）
```

验证流程不是在同一进程内重复请求，而是：

1. 启动第一个 SGLang 服务，首次请求插入前缀；
2. 再次请求相同前缀，使 `hit_count` 达到当前实现的 write-through 阈值；
3. 等待 GPU→Host backup 完成并写入 Lustre；
4. 完全终止第一个服务，清除该进程的 L1、L2 和 radix metadata；
5. 启动第二个服务，重新提交完全相同的 prompt；
6. 检查响应 cached tokens 和 storage prefetch Prometheus counter。

结果：

```text
L3 文件：222 个 .bin
L3 字节：465,567,744 bytes
重启后 cached tokens：1,984
prefetched_tokens_total 增量：15,872
HICACHE_LUSTRE_RESTART_HIT_OK
HICACHE_LUSTRE_L3_SMOKE_COMPLETE
```

`prefetched_tokens_total` 带 TP-rank 标签，15,872 是 8 个 rank 的求和，即
`1,984 × 8`，不能解释为 15,872 个逻辑 prompt tokens。服务重启后本地
L1/L2 已不存在，因此本次 1,984-token 前缀命中可归因于 Lustre L3。

本次同时确认当前 SGLang 的 write-through 语义：新插入节点不会立即持久化；
该节点再次命中后才触发 backup。因此测试只发送一次 populate 请求会观察不到
`.bin` 文件，这不是 Lustre 写失败。

`--hicache-ratio 1.2` 使每个 TP rank 分配 44.48 GB Host KV pool，总计约
355.84 GB。重启实验通过消除整个进程内的 L1/L2 状态，证明本次恢复来自 L3；
普通稳态实验仍按 SGLang 原生层次缓存语义解释。

### 已验证：跨节点 Decode 写 L3、Prefill 读 L3

2026-09-01 进一步完成 1P1D 跨节点持久化数据流：

```text
Slurm Job：64889，COMPLETED，ExitCode=0:0，耗时 5m49s
Prefill：dgx-gaia-09，8 × H100，TP=8
Decode：dgx-gaia-10，8 × H100，TP=8
PD transfer：Mooncake Transfer Engine / mlx5_0
L3 backend：HiCache file / Lustre
L3 目录：/mnt/lustre/users/sezhou/dual-path/kv-cache/pd-1p1d-file/64889/
脚本：scripts/pd_1p1d_benchmark.sbatch（原 pd_1p1d_lustre_l3.sbatch 已并入）
```

第一轮由 Prefill 计算 prompt KV 并通过 Mooncake RDMA 发给 Decode。Decode
生成完成后，`--disaggregation-decode-enable-offload-kvcache` 将完整前缀 KV
写入共享 Lustre，得到：

```text
248 个 .bin
520,093,696 bytes
```

随后同时终止 Prefill、Decode 和 Router，重新启动两侧服务。第二轮提交
“第一轮完整 prompt + 第一轮真实生成文本 + 短 append”，Prefill 从共享 L3
恢复前缀并继续计算：

```text
PD_L3_PREFETCH_TOKENS=15872
PD_L3_RECOVER_CACHED_TOKENS=1984
PD_LUSTRE_L3_DECODE_WRITE_PREFILL_READ_OK
PD_1P1D_LUSTRE_L3_SMOKE_COMPLETE
```

同样，15,872 是 8 个 TP rank 的 Prometheus counter 求和，对应 1,984 个逻辑
prefix tokens。Prefill 和 Decode 的恢复阶段日志均确认 Mooncake RDMA 使用
`mlx5_0`，未出现 KV transfer、OOM 或崩溃错误。

当前版本的 Decode 模式会强制使用 chunk cache 并禁用 radix cache，因此
Decode 不能同时显式设置 `--enable-hierarchical-cache`，否则启动时报两参数
互斥。正确配置是在 Decode 侧保留 storage backend 参数，并仅通过
`--disaggregation-decode-enable-offload-kvcache` 启用专用 offload manager；
Prefill 侧仍需显式设置 `--enable-hierarchical-cache`。

该 smoke 已证明“Decode 写 Lustre L3 → 服务重启 → Prefill 跨节点读取 L3 →
Prefill 计算缺失 append → Mooncake 发送完整 KV 给 Decode”的最小闭环。

### 已验证：1P2D 共享 Lustre L3 命中

2026-09-01 完成三个节点的 1P2D + L3 重启恢复：

```text
Slurm Job：64980，COMPLETED，ExitCode=0:0，耗时 4m43s
Prefill：dgx-gaia-51，8 × H100，TP=8
Decode-0：dgx-gaia-52，8 × H100，TP=8
Decode-1：dgx-gaia-53，8 × H100，TP=8
Router policy：Prefill/Decode round_robin
L3：HiCache file backend / 共享 Lustre
脚本：scripts/pd_1p2d_benchmark.sbatch（原 pd_1p2d_lustre_l3.sbatch 已并入）
```

populate 阶段向 1P2D 集群发送两个独立前缀并重复四轮，Decode offload 在共享
目录生成 584 个 `.bin`、1,224,736,768 bytes。停止并重启 Prefill、两个
Decode 和 Router 后，两个前缀均由 Prefill 从 L3 恢复：

```text
prefetched_tokens_total 增量：37,376（8 个 TP rank 求和）
trajectory-0 cached tokens：2,304
trajectory-1 cached tokens：2,368
PD_1P2D_DECODE_OFFLOAD_TO_SHARED_L3
PD_1P2D_BOTH_PREFIXES_RECOVERED
PD_1P2D_LUSTRE_L3_SMOKE_COMPLETE
```

两个 Decode 的恢复日志均记录 `/generate` 请求及 Mooncake RDMA 初始化，未出现
KV transfer、OOM 或崩溃错误。不能把“两个 Decode 每轮都创建新 `.bin`”作为
验收条件：共享 namespace 去重、请求分配和 Decode offload admission 可能使
某个 Decode 处理请求但无需产生新文件。正确条件是共享 L3 有 Decode 写入、
两个独立前缀都能在重启后命中、两个 Decode 都参与恢复数据面。

Job 退出 trap 已删除
`/mnt/lustre/users/sezhou/dual-path/kv-cache/pd-1p2d-file/64980/`，日志和 response
保留。此前历史 L3 smoke/replay 遗留的 7,369 个文件、15,448,588,288 bytes
也已统一清理。

### 已验证：1P2D + L3 并发 fake trajectory

Job `64984` 在 1P2D 上并发执行 4 条 32K 档 trajectory，每条取前 4 轮。
populate 完成 16/16 轮后等待 Decode 异步写回稳定，停止并重启全部三个模型
角色和 Router，再以相同 dataset、offset 和 seed 重放：

```text
Slurm Job：64984，COMPLETED，ExitCode=0:0，耗时 5m11s
节点：dgx-gaia-51 / 52 / 53
并发：4 agents
每条轮数：4
populate：16/16 turns，0 errors，wall 5.795s
recover：16/16 turns，0 errors，wall 5.632s
L3：816 个 .bin，1,711,276,032 bytes
recover cached tokens：min 320，max 1,600
脚本：scripts/pd_1p2d_benchmark.sbatch（原 pd_1p2d_l3_replay.sbatch 已并入）
```

恢复阶段全部 16 个请求的 `cached_tokens > 0`，且 generation token 数全部与
trajectory 目标一致。Decode-0 和 Decode-1 在 populate/recover 阶段各记录
9 个 `/generate`、8 个 RDMA transport 和 64 条 backup 完成日志，证明并发
请求覆盖两个 Decode，并且共享 L3 数据在全服务重启后可由 Prefill 恢复。

该测试用于正确性，不是吞吐 benchmark：populate 与 recover 之间等待文件数
稳定，且只执行每条 trajectory 的前 4 轮。退出时已自动删除
`kv-cache/pd-1p2d-replay/64984/`。

### 历史诊断：1P1D + file L3 完整 32K trajectory

2026-09-01 使用 `dualpath_fake_32k.jsonl` 中 `32k-0250` 完成一条 94 轮
trajectory。每轮都将真实模型输出 token IDs 拼入下一轮，并在下一轮前调用
Prefill `/flush_cache`，主动清除 Prefill L1、L2 和 radix metadata，但不调用
独立的 `/clear_hicache_storage_backend`，因此 Lustre 文件被保留：

```text
Slurm Job：64914
模型执行：94/94 turns，0 errors
最终 context：24,836 tokens
请求延迟：mean 2.605s，p50 2.723s，p90 3.729s，p99 3.929s
总 wall time：1,740.5s
L3 文件：3,088 个 .bin
L3 字节：6,476,005,376 bytes
脚本：scripts/pd_1p1d_benchmark.sbatch（原 pd_1p1d_l3_full_replay.sbatch 已并入）
结果：logs/pd-1p1d-l3-replay/64914/results/
```

94 轮中第 1 轮没有可复用前缀；其余 93 轮有 69 轮报告 L3 cached tokens，
命中轮次比例为 74.19%。第 26 轮首次命中，之后直到第 94 轮连续命中。末轮
`prompt_tokens=24,806`、`cached_tokens=24,640`，恢复率为 99.33%。

该结果揭示旧 file backend 下 Decode 专用 offload manager 的 admission 行为：前 24 个
可复用轮次虽然每轮都有少量 `.bin` 增长，但并未立即持久化完整连续前缀；
context 约 9K 后触发批量 offload，文件数从 216 跃升到 1,160，下一轮开始
稳定命中。因此该历史实现不能把“第 2 轮必定命中”作为正确性条件。

Job 64914 的 94 轮模型执行和结果文件完整，但 Slurm 最终状态是 `FAILED`：
旧版验收仍断言第 2 轮起每轮命中。该断言已按上述实测语义修正；无需把这个
wrapper 断言失败解释为推理或 L3 数据流失败。

这个测试不是性能 benchmark：每轮主动 flush，并等待异步落盘增长，会人为增加
wall time。该方案已经从 `scripts/benchmark/replay_fake_agent_traces.py` 和正式
sbatch 中删除；
Job `64914/65502` 仅保留为历史命中归因证据。

## Phase 6：DualPath 近似实现

只有 Phase 5 已证明 Prefill 侧 Lustre 读取饱和后，才进入代码改造。目标数据路径：

```text
PE read path: Lustre -> PE DRAM -> PE HBM -> DE
DE read path: Lustre -> DE DRAM -> PE HBM
```

最小实现按以下顺序进行：

1. 在 Decode 节点增加只读 L3 fetch，先读入 host buffer；
2. 复用 Mooncake/NIXL transfer engine 将命中的 KV 从 DE 传给 PE；
3. 以 request 为粒度在 PE-read 与 DE-read 二选一，不做单请求拆分；
4. 先用最短读队列策略：`argmin(read_queue_bytes / measured_bandwidth)`；
5. 再加入 unfinished tokens，避免网络平衡但 GPU 失衡；
6. 最后才研究 layer-wise streaming、CNIC QoS/VL 和 copy/compute overlap。

第一版调度输入至少包含：

- 每节点待处理读字节数或读 token 数；
- 每个 engine 的 unfinished token 数；
- Decode 剩余 HBM/host buffer；
- PE/DE pair 的 transfer queue；
- 最近窗口的 Lustre 实测带宽。

不要在第一版实现中声称已复现论文的 CNIC-centric traffic manager。论文把模型通信和 KV 流量映射到不同 IB VL，并在交换机与 NIC 上配置仲裁；这通常需要管理员权限和全网一致配置。
