# 用 NIC 完成 HiCache H2D/D2H（`DUAL_PATH_CUDAMEMCPY`）

> 返回 [文档索引](README.md)

本文对应论文 §5.2 的 CNIC-assisted KV-Cache copy：把 HiCache 在 host pinned pool
与 GPU KV pool 之间的拷贝，从 CUDA 拷贝引擎改为由 GPU 配对的 CX8 以 RDMA
完成，让网卡成为 GPU PCIe 流量唯一的 QoS 仲裁点（见
[VL 流量隔离](vl-traffic-isolation.md)）。mini bench 结论见 §2.1（完整数据见
[io-test/RESULTS.md](../io-test/RESULTS.md)），开关已实现为同步版本并构建了镜像，
用法与验证见 §3。

## 1. 现有 HiCache 拷贝路径

项目在 Prefill 和 Decode 上都使用 `--hicache-io-backend direct`：

| 角色 | host 布局 | H2D（每层） | D2H（全部层） |
|---|---|---|---|
| Prefill | `layer_first` | `transfer_kv_direct` | `transfer_kv_direct` |
| Decode | `page_first_direct` | `transfer_kv_per_layer_direct_pf_lf` | `transfer_kv_all_layer_direct_lf_pf` |

这几个 sgl_kernel 函数按连续段调用 `cudaMemcpyAsync`，走拷贝引擎和 GPU 的
PCIe 链路，不经过网卡。调用点位于镜像中的 `sglang/srt`：

- `managers/cache_controller.py` 的 `start_loading()`：在 `load_stream` 上逐层调用
  `load_to_device_per_layer`，每层之后执行 `producer_event.complete(i)`，模型前向
  通过 `LayerDoneCounter` 等待对应层；
- 同一文件的 `start_writing()`：在 `write_stream` 上等待 `start_event` 后调用
  `backup_from_device_all_layer`，之后记录 `finish_event` 并放入 `ack_write_queue`；
- `mem_cache/memory_pool_host.py` 的 `MHATokenToKVPoolHost`：按 `io_backend` 和
  `layout` 分派到上表中的函数。

每个 rank 的数据粒度（Qwen2.5-32B、TP4、bf16、page 64）：每层每页 K、V 各
32 KiB；一个 32K 上下文每层 32 MiB，64 层共 2 GiB。

## 2. mini bench

| 脚本 | 内容 |
|---|---|
| `io-test/run_nic_loopback_perftest.sh` | 宿主机上的 perftest 同网卡回环：h2d（server 端 `--use_cuda`）、d2h、h2h 基线；消息 4 KiB–4 MiB，QP 1/4/16，SL 0 与 2；GPU0/1 共享 `mlx5_0` 的并发组；`ib_write_lat --write_with_imm` 往返延迟 |
| `io-test/h2d_path_bench.py` | 容器内，按 TP4 形状分配 GPU KV pool 和两种 host 布局；`hicache` 直接调用 sgl_kernel 的 direct 函数，`hicache-kernel` 调用零拷贝 kernel，`nic` 用 Mooncake TE 回环（每个连续段一个请求）；连续与随机两种页分布；1、2、4 个 GPU 并发；首轮做数据校验 |
| `io-test/capture_path_counters.py` | 包裹任意命令：HCA `port_*_data` 与 `hw_counters`（`rx_write_requests` 等）前后差值、`nvidia-smi dmon -s t` 的每 GPU PCIe 字节积分，可选 `--sampler-cmd`（例如以 root 运行 AMDuProfPcm） |
| `io-test/run_h2d_path_bench.sh` | 一键运行 A1–A3，`RUN_INTERFERENCE=1` 时追加 A4（H2D 负载下的 TP4 all-reduce）；无 allocation 时自动 `salloc` g5 |
| `io-test/summarize_h2d_path.py` | 生成 `summary.json`、`REPORT.md`，并按决策门槛逐项判定 |

```bash
cd /data/home/yongxiao/dual-path
RUN_INTERFERENCE=1 io-test/run_h2d_path_bench.sh
# 结果：io-test/results/<jobid>-h2d-path/{REPORT.md,summary.json,perftest/,pybench.jsonl}
```

判定方法：

- **是否经过网卡**：`hw.rx_write_requests`（或 `rx_read_requests`）增量达到请求数量级。
  同网卡回环时 `port_rcv_data` 随数据量增长，而 `port_xmit_data` 不变，说明数据在
  HCA 内部回环，没有经过交换机。
- **是否经过 GPU PCIe**：dmon 积分的 rx+tx 字节与数据量之比接近 1。hicache 与 nic
  都应接近 1（两者都要经过 GPU 的 x16 链路），区别只在于数据是否先经过 HCA。
- **决策门槛**：单 GPU、真实粒度下，nic 的每层耗时 ≤ 1.2 × hicache，或带宽
  ≥ 0.8 × hicache；同时计数器证实经过 HCA，且数据校验通过。

### 2.1 实测结论（2026-09-29，Job 597–599，g5）

- **NIC 路径的硬件上限**：NIC 写 A100 显存（h2d）约 18 GB/s，NIC 读显存（d2h）
  约 10–11 GB/s；拷贝引擎约 26.8 GB/s。两块 GPU 共享一块 CX8 时合计约 28.7 GB/s。
- **Decode（`page_first_direct`）**：nic 的 h2d 为 13.0 GB/s，是 hicache 6.4 GB/s 的
  2.0 倍；d2h 为 8.9 GB/s，是 hicache 的 1.24 倍。原因是 hicache 每层要发 1024 次
  `cudaMemcpyAsync`。这组通过门槛。
- **Prefill（`layer_first`）**：页连续时 hicache 每层只需两次大拷贝，nic 只有它的
  0.60×（h2d）和 0.38×（d2h），不通过门槛；随机页时 nic 为 1.48×（h2d），通过。
- **路径证据**：nic 的所有 case 都有与请求数一致的 `rx_write_requests`，
  `port_rcv_data` 增长而 `port_xmit_data` 不变，即 HCA 内部回环、不经过交换机；
  hicache 的 case 网卡计数为 0。两者的 GPU PCIe 字节都约等于数据量。
- **干扰**：4 块 GPU 同时做 nic H2D 时，TP4 all-reduce 8 KiB 延迟从 45 µs 升到
  118 µs，1 MiB 从 180 µs 升到 313 µs；hicache 负载下基本不变。原因是跨对 NCCL
  与 KV 回环共用同一块 CX8。因此 `nic` 模式必须配合 VL QoS（KV 使用低优先级 SL）
  才有意义，不能单独作为性能优化来开启。
- **实现载体**：第一版用 Mooncake TE。它在连续 h2d 上达到 perftest 上限的 90%，
  32 KiB 散页上达到 73%（约 0.7 µs/请求的额外开销）。自研 verbs 扩展列为后续优化。

## 3. 实现（2026-09-30，第一版）

### 3.1 用法

| 变量 | 取值 | 默认 | 含义 |
|---|---|---|---|
| `DUAL_PATH_CUDAMEMCPY` | `hicache` / `nic` | `hicache` | `hicache` 完全保持原有行为；`nic` 让 HiCache 的 H2D/D2H 改走 CX8 RDMA |
| `DUAL_PATH_NIC_DEVICE` | HCA 名 | 空 | 显式指定网卡；为空时按 GPU 编号查 `IB_DEVICE` / `MOONCAKE_DEVICE` / `DUALPATH_PZA100_IB_DEVICE_MAP` 的 JSON 映射（GPU0/1 为 `mlx5_0`，GPU2/3 为 `mlx5_4`） |
| `DUAL_PATH_NIC_HOST` | IP | `127.0.0.1` | Mooncake 回环握手地址 |
| `DUAL_PATH_NIC_FLUSH` | `0` / `1` | `1` | 每次传输后回读 4 字节，确保网卡的 PCIe 写已落地 |

- 镜像：`sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-l3bps-20260929-nich2d.sqsh`
  （SHA-256 `5a09b0306ba30721ff2e39377b6a3cf7bb0af9e3bd6df4dba2d55ab125a8ca57`），
  由 `scripts/infra/build_nic_h2d_image.sh` 在当前默认的 l3bps 镜像之上叠加补丁得到。
  默认值为 `hicache`，因此不设置开关时，行为与 l3bps 镜像完全相同。
- PD 实验：`IMAGE=<上面的镜像> DUAL_PATH_CUDAMEMCPY=nic scripts/pd_benchmark.sh ...`。
  Prefill、Decode 和 client 三端必须一致；两个变量已加入 `pd_benchmark.sh` 的透传列表，
  并由 `pd_benchmark_lib.sh` 校验和导出，取值记录在结果元数据的
  `dual_path_cudamemcpy` 中。
- 限制：只支持 `--hicache-io-backend direct`，且只对 MHA 模型生效（host 布局为
  `layer_first` 或 `page_first_direct`）。其他布局在 host pool 构造时直接报错；
  其他 io backend 在第一次拷贝时报错。MLA host pool 不受开关影响。

### 3.2 代码

| 文件 | 作用 |
|---|---|
| `scripts/infra/dualpath_nic_copy.py` | 安装为 `sglang/srt/mem_cache/dualpath_nic_copy.py`，内含 `NicCopyEngine` |
| `scripts/infra/patch_sglang_nic_h2d.py` | 可重复执行的补丁：安装上述模块，并在 `MHATokenToKVPoolHost` 中插入三处调用 |
| `scripts/infra/build_nic_h2d_image.sh` | `srun --container-writable --container-save` 派生镜像，并做导入检查 |

`MHATokenToKVPoolHost` 的三处改动：

1. `__init__` 末尾：开关为 `nic` 时注册本 host pool 及其 device pool。第一次调用时
   创建本进程唯一的 `NicCopyEngine`，完成 Mooncake 初始化和 1 MiB 往返自检。
2. `load_to_device_per_layer`：转交 `h2d_layer`。
3. `backup_from_device_all_layer`：转交 `d2h_all_layers`。

三个调用方（`HiCacheController.start_loading` / `start_writing`，以及 Decode PD
staging 的 `stage_received_kv_to_device`）都在“当前 stream”上调用这两个方法，
因此无需改动。

`NicCopyEngine` 的职责：

- **传输实例**：一个独立的 Mooncake `TransferEngine`，会话指向自己，不与 PD 传输
  共用会话。
- **内存注册**：host pool 的 `kv_buffer` 整块注册，device pool 的 `k_buffer` /
  `v_buffer` 逐层注册，每个 pool 只注册一次。A100 的 BAR1 为 64 GiB，可以覆盖整个
  KV pool。实测注册 9.4–12.9 GB 耗时约 0.2 s。
- **请求规划**：把 token 索引划分成 host、device 两侧都连续的段，按层加上偏移，
  得到每个请求的地址和长度。H2D 的规划结果在同一批次的 64 层之间复用。

### 3.3 同步方式：同步阻塞调用线程

每次调用在调用线程中依次完成：

1. `stream.synchronize()` 调用方当前的 stream。这保证 stream 上排在前面的工作（包括
   `load_stream` 对计算流 `start_event` 的等待）都已完成，网卡才开始读写；
2. `batch_transfer_sync_write` 完成全部请求；
3. flush：对目的地址做 4 字节 RDMA READ。

返回之后，调用方在 stream 上记录的逐层事件和 `finish_event` 自然都能看到完整数据，
`LayerDoneCounter` 和 `HiCacheAck` 的语义不变。

设计阶段计划的异步方案（Python 后台线程提交传输，stream 用 `cuStreamWaitValue32`
等待 host-pinned 标志）实测会**死锁**：另一个线程中的 CUDA 调用（例如
caching allocator 的 cudaMalloc）在持有 GIL 的情况下阻塞，等待 GPU；GPU 在等标志；
写标志的 worker 却拿不到 GIL。SGLang 的 scheduler 线程随时可能发生这种阻塞，因此
异步版本必须由原生代码完成“等待传输完成并写标志”这一步（即 verbs 扩展），不能依赖
Python 线程。

同步方式的代价是调用线程被阻塞：

- H2D 按层阻塞，失去 layerwise 拷贝与计算的重叠；
- 开始拷贝前要等调用方 stream 的前序工作完成，在 overlap scheduler 下可能要等上一批
  forward 结束。

另一方面，Decode 使用的 `page_first_direct` 布局下，hicache direct 本来就要花大约
总耗时的 CPU 时间逐页提交 `cudaMemcpyAsync`（§2.1），所以对这个布局而言，同步 nic
占用 scheduler 线程的时间反而更短。

### 3.4 验证

| 验证 | 结果 |
|---|---|
| `scripts/smoke/nic_h2d_selftest.py`：真实 `MHATokenToKVPoolHost` + Qwen2.5-32B TP4 形状 device pool，512 个随机页 × 64 层，两种布局、两个方向、nic 与 hicache 各两轮；consumer stream 只按逐层事件等待后立即比对 | 新镜像中全部 0 个不一致元素，`SELFTEST PASS`；H2D nic 10.7 GB/s，hicache 6.1–8.5 GB/s；D2H nic 6.6–10.5 GB/s，hicache 6.5–8.1 GB/s |
| `scripts/smoke/nic_h2d_hicache_e2e.sh`（Job 607）：Qwen3-0.6B TP1，显存池 16384 token，`write_through`；8843 token 的请求被 4 个请求挤出显存后再次发送 | 两种布局都命中 8832/8843 个 token（来自 host pool）；贪心输出与第一次相同，nic 与 hicache 逐 token 一致；网卡写请求 hicache 为 0，nic 为 148 万；`E2E PASS` |

尚未验证：PD 1P1D 的完整 replay（TTFT/TPOT/JCT）、TP4 多 rank，以及 Decode PD
staging 路径。

## 4. 风险与待确认

- 同步阻塞会增加 scheduler 线程的停顿，需要用 PD replay 量化 TTFT/TPOT 的影响；
  Prefill `layer_first` 页连续时 nic 本身也慢于 hicache（0.60× / 0.38×）。
- 真正的异步需要 verbs C++ 扩展：由原生线程等待 CQ 完成，写 host 标志，stream 用
  `cuStreamWaitValue32` 等待，整个过程不经过 GIL。
- Mooncake 的 SL 由进程级的 `MC_IB_SL` 决定，会同时作用于同一进程里的 PD 传输；
  两者都属于 KV 流量，应该用同一个 SL。
- 同网卡回环不经过交换机（`port_xmit_data` 不变），所以 H2D 只受 HCA 自身发送调度的
  约束。它能否获得 VL 仲裁，要看 HCA 内部回环是否经过端口仲裁器，需要在 QoS 启用后，
  用 [VL 流量隔离](vl-traffic-isolation.md) 的 E3/E5 验证。
- 多平面 CX8 上，回环流量如何在 4 个平面之间分配，尚未观测。
- SGLang 用 `cudaHostRegister` 锁页 host pool 且从不注销；同一进程里释放后重建 host
  pool 会失败。生产代码不会这样做，但测试脚本要注意。
