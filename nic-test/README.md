# D→P RDMA 对完整 forward 的影响（Job 635）

结果在 `results/635-d2p-forward/`，脚本为 `d2p_forward_bench.py`、`run_d2p_forward.sh`、`plot_d2p_forward.py`。

![d2p-forward](results/635-d2p-forward/d2p-forward.svg)

**干扰流量仿照真实 D→P**（`_send_kvcache_generic`，`executor=None`）：
- 每个请求一次 `batch_transfer_sync_write`，包含 64 层 × K/V 共 128 块。每块是该层连续的前缀页：18,626 个命中 token，即 291 页 × 32 KiB = 9.1 MiB。每个 rank 每个请求共 1.22 GB。
- 源是 g6 的 pinned host 内存，目的是 g5 的 GPU 显存，每块 GPU 走各自 PIX 的 CX8。
- 每块 GPU 2 个线程提交请求，按平均速率放行；每个请求内部全速发送，和真实系统一样是突发。
- 最大速率约为每块 GPU 15.5 GB/s（每块网卡约 31 GB/s）。线程数取 1、2、4 时最大速率基本相同（14.4、15.2、15.2 GB/s），已到网卡写 A100 显存的上限。

**探针是完整的 Qwen2.5-32B TP4 forward**：每层依次是 QKV GEMM、对缓存上下文的 attention、O GEMM + all-reduce、gate/up GEMM、SiLU、down GEMM + all-reduce，另有 embedding all-reduce 和 logits GEMM + all-gather。
- Prefill（g5）：1004 个新 token，上下文 19,630，eager。
- Decode（g6）：batch 6，每条上下文 19,782，CUDA graph。

| D→P 每块 GPU | Prefill forward（均值） | 倍数 | Decode 一步（均值 / p99） | 倍数（均值 / p99） |
|---|---|---|---|---|
| 0 | 233.7 ms | 1.00 | 28.5 / 29.1 ms | 1.00 / 1.00 |
| 5 GB/s | 287.7 ms | 1.23 | 39.2 / 99.2 ms | 1.38 / 3.41 |
| 10 GB/s | 375.2 ms | 1.61 | 60.1 / 99.3 ms | 2.11 / 3.41 |
| 15 GB/s（接近跑满） | 547.8 ms | 2.34 | 179.0 / 184.5 ms | 6.28 / 6.34 |
| 20、25 GB/s（跑满，约 15.5） | 547.5 ms | 2.34 | 179.7 / 183.4 ms | 6.30 / 6.30 |

- **跑满时**：Prefill forward 慢 2.34 倍（+314 ms），Decode 一步慢 6.3 倍（+151 ms）。
- **Decode 在 5～10 GB/s 时均值和 p99 差很大**：
  - 每个请求 1.22 GB 以全速发送，约 81 ms 发完。
  - 落在突发期间的 decode 步约慢 3.5 倍（约 99 ms）；不在突发期间的不受影响（5 GB/s 时 p50 仍为 28.5 ms）。
  - 所以平均速率低时，影响体现为间歇性的长尾。
- **图中“compute only”线**是去掉所有集合通信、eager 执行的时间（Prefill 127 ms，Decode 约 25 ms，含 kernel 启动开销），仅作参考。
- **测试脚本的修复**：探针校准时，每个 rank 按自己测到的耗时决定迭代次数，4 个 rank 的次数可能不同，导致集合通信错位而挂死（Job 632）。现在先对耗时取 4 个 rank 的最大值，再统一计算次数。`allreduce_interference_bench.py` 有同样的隐患，已一并修复。

复现：`nic-test/run_d2p_forward.sh`，全程约 5 分钟；可用 `LEVELS`、`D2P_THREADS`、`CASE_SECONDS` 覆盖默认值。

## 真实运行中的 D→P 干扰

用 `analyze_real_d2p.py` 检查了两组真实的 32K、8 agent 运行。每组对比 Decode 路径（有 D→P）和 Prefill 路径（无 D→P），其余配置相同：
- j591 对比 j590：hicache；
- j629 对比 j625：nic-stream。

**结论：干扰真实存在，但比上面的合成测试温和得多；只打到 Decode，Prefill 基本不受影响。**

- **流量条件与测试的跑满档一致**：每个 rank 的 D→P 为 16～20 GB/s，每次持续 66～79 ms，约占总时间的 6%。
- **Decode 侧**（ITL 为 turn ≥ 3 的 token 间隔）：

  | 运行 | 和其他请求 D→P 重叠的 ITL：均值 / p90 / 超过 60 ms 的比例 | 远离 D→P（300 ms 外）的 ITL：均值 / p90 / 超过 60 ms 的比例 | 全部 ITL 均值 | 对应 Prefill 路径的全部 ITL 均值 |
  |---|---|---|---|---|
  | j591（hicache） | 33.4 / 39.9 ms / 1.6% | 29.3 / 32.9 ms / 0.45% | 30.3 ms | 29.6 ms（j590） |
  | j629（nic-stream） | 34.7 / 54.5 ms / 6.5% | 29.9 / 31.1 ms / 1.3% | 30.9 ms | 30.8 ms（j625） |

  - 和 D→P 重叠的 ITL，均值慢约 15%，p90 慢 20%～75%，超过 60 ms 的比例是远离 D→P 时的 3.5～5 倍。
  - 合成测试在同样速率下预测单步慢 3.5～6 倍，真实远没有这么严重。
  - D→P 只占约 6% 的时间，所以整体 ITL 与 Prefill 路径几乎相同。
- **Prefill 侧**：
  - Prefill 只有 22%～23% 的时间在做 forward。
  - 约 190 个 forward 中，只有 2～3 个和其他请求的 D→P 重叠超过 5%。
  - 每个请求的 D→P 紧挨在它自己的 forward 之前，这段时间 Prefill 通常空闲。
  - 所以在当前的全量 D→P 模式下，合成测试中 Prefill 慢 2.34 倍的情况基本不会出现。如果改成按层边传边算，让 D→P 和 forward 重叠，就会真正遇到。
  - Prefill 每 token 耗时的中位数两种路径相同：j591 为 0.275 ms，j590 为 0.274 ms；j629 为 0.276 ms，j625 为 0.299 ms。

**时间线的重建方法**（日志时间戳都是各节点自己的 monotonic 时钟）：

- Decode 侧（g6 时钟）：
  - D→P 窗口：4 个 rank 中最晚完成 L3 读取和 scatter 的时刻（`decode_before_read_end_ns + lustre_read_raw + scatter_raw`），加上 `decode_l3_sync_wait_seconds`，再持续 `d_to_p_transfer_raw_seconds`。
  - Token 时刻：`decode_dispatch_ns` 加上客户端的累计 ITL。客户端也跑在 g6 上。
- 映射到 g5 时钟：用同一请求的 `p_to_d_end_ns`（g5）和 Decode 侧到达时刻（g6）求时钟偏移，各请求间偏差在 ±15 ms 内。
- 窗口位置的校验：同一请求自己的 forward 开始时刻，应在它的 D→P 结束之后。实测中位数晚 58～86 ms，p10 为 −27～−38 ms，说明窗口位置大体可信，但仍有几十毫秒误差，会稀释重叠组的差异。
- **L3 同步等待必须计入窗口起点**：不计入的话，窗口会偏早 100 ms 以上，重叠组和远离组就没有差别了，会误判为“真实运行没有干扰”。

**真实比测试温和的可能原因**（均未验证）：
- 真实传输按 `mooncake_max_transfer_batch_indices=128` 分段提交；
- 4 个 rank 不像测试那样严格同时开始；
- 一次 D→P 期间只有 2～3 个 decode step 落在里面。

要精确测单步幅度，需要在 reverse fetch 中直接记录 D→P 的开始和结束时刻，而不是靠现在的推算。

复现：

```bash
cd /data/home/yongxiao/dual-path
R=results/latency-benchmark; T=/data/home/yongxiao/mydata/test
python3 nic-test/analyze_real_d2p.py $R/32k-a8-r1-decode-l3bps8g-nfsrdma-nic-stream-20260930-j629 \
    $T/629/pd-servers/decode-0.log --prefill-log $T/629/pd-servers/prefill-0.log \
    --baseline $R/32k-a8-r1-prefill-l3bps8g-nfsrdma-nic-stream-20260930-j625
```

---

# VL 隔离对照（Job 649 / 650，2026-10-01）

**结论：把 KV 流量放到另一条 VL，Decode 侧 D→P 的干扰几乎消失（30 倍 → 1.1 倍）；
Prefill 侧和 NIC 回环 H2D 不受影响。**

两组除 SL 以外完全相同：`DUAL_PATH_NCCL_FORCE_NIC=0`，场景 d2p 和 nic_h2d，
档位 0/10/20/100%，每个 case 25 s。SM 未改动，当前 SL0 → VL0、SL1 → VL1，
两条 VL 在 low 表里按 160:112 加权轮转，没有优先级
（见 [docs/vl-traffic-isolation.md](../docs/vl-traffic-isolation.md) §4.5）。

- Job 649（`results/649-allreduce-interference/`）：不设 SL，全部在 VL0。
- Job 650（`results/650-allreduce-interference/`）：`PROBE_SL=0 TRAFFIC_SL=1`，TP4 探针在 VL0，Mooncake 流量在 VL1。

平均通信时间倍数（相对同组 0% 负载）：

| 方案 | 节点 | 10%：无 VL → 有 VL | 20% | 100%（跑满，约 20%～26%） |
|---|---|---|---|---|
| D→P | Decode（g6） | 1.97 → **1.06** | 30.6 → **1.11** | 32.2 → **1.12** |
| D→P | Prefill（g5） | 1.58 → 1.51 | 3.91 → 3.77 | 4.09 → 3.58 |
| NIC 回环 H2D | Decode | 1.38 → 1.39 | 2.17 → 2.21 | 3.72 → 3.90 |
| NIC 回环 H2D | Prefill | 1.22 → 1.32 | 1.42 → 1.74 | 1.43 → 1.82 |

两组的实际流量相同（D→P 约 20%、NIC 回环约 20%～26%），0% 基线也一致
（Prefill 约 123 ms，Decode 约 7.7 ms）。

解读：

- **Decode 侧**：g6 的探针是跨 GPU 对的 NCCL（mlx5_0 → 交换机 → mlx5_4），和发往 g5
  的 D→P 共用 g6 网卡的发送端口。D→P 在 g5 被 GPU PCIe 卡住，credit 反压传回 g6，
  同一条 VL 上的 NCCL 被堵在后面。分到两条 VL 后各用各的 credit，干扰基本消失。
  这证实了 627 中“Decode 慢 18～30 倍”来自 IB 层的 credit 反压。
- **Prefill 侧**：g5 的 NCCL 和 D→P 都要写进 g5 的 GPU 显存，争用的是 GPU 的 Gen4
  PCIe，VL 管不到这一段，倍数基本不变。
- **NIC 回环 H2D**：不经过交换机，SL 对它没有帮助；Prefill 侧有 VL 时反而更慢
  （1.43 → 1.82），原因未查明，可能是 HCA 内部对两条 VL 的调度，只跑了一次。
- 本组只测了 `FORCE_NIC=0`。`=1` 时同一对 GPU 的 NCCL 也在 HCA 内部回环，是否受益
  需要单独测。

复现：

```bash
export DUAL_PATH_NCCL_FORCE_NIC=0 SCHEMES=d2p,nic_h2d LEVELS=0,10,20,100 \
       CASE_SECONDS=25 EXTRA_ARGS="--measure-seconds 8"
nic-test/run_allreduce_interference.sh                          # 无 VL
PROBE_SL=0 TRAFFIC_SL=1 nic-test/run_allreduce_interference.sh  # 有 VL
```

---

# KV 流量对 TP4 all-reduce 的干扰（g5/g6，CX8，无 VL QoS）

测量 P→D RDMA、D→P RDMA、hicache H2D、NIC 回环 H2D 四类 KV 搬运流量，对同时进行的 TP4 通信的影响。有效数据：Job 627，2026-09-30，结果在 `results/627-allreduce-interference/`。

![interference](results/627-allreduce-interference/interference.svg)

## 结论

1. **D→P RDMA 干扰最严重**：10% 负载时 Prefill 通信慢 1.48 倍、Decode 慢 1.75 倍；20% 起跑满，Prefill 慢约 3.6 倍，Decode 慢 18～30 倍（p99 约 24 倍）。
2. **P→D RDMA 几乎无影响**：两种模式都在 1.03～1.06 倍之间，与 D→P 流量相同、方向相反。
3. **NIC 回环 H2D 的干扰明显大于 hicache H2D**：10% 负载时 Prefill 慢 1.21 倍、Decode 慢 1.45 倍；跑满（约 20～26%）后 Prefill 慢 1.41 倍，Decode 慢约 3.9 倍。
4. **hicache H2D 基本无害**：Decode 最多慢 1.22 倍；Prefill 反而快了 15%，原因未查明。
5. **所有方案都到不了单卡 CX8 份额的 100%**：CX8 为 800 Gb/s（100 GB/s），由两块 A100 PCIe Gen4 共用，所以每块 GPU 的份额是 50 GB/s，横坐标以此为 100%。实际上限如下：

   | 方案 | 实际上限（占单卡 50 GB/s） |
   |---|---|
   | P→D、D→P | 约 20～22%（单卡约 10～11 GB/s） |
   | NIC 回环 H2D | 约 20～26% |
   | hicache H2D（不经过网卡，按等量字节折算） | 约 38～48% |

   30% 往后的空心点都表示“跑满时”的影响。

   P→D、D→P 的上限是本测试流量生成器的限制（每个请求 32 KiB、同步提交、10 ms 节拍），不是硬件上限。按真实方式整请求提交时，D→P 每块 GPU 可达约 15.5 GB/s（见 Job 635），真实运行中每个 rank 达到 16～20 GB/s。

对方案选择的含义：

- 当前没有 VL 隔离，把 H2D 换成 NIC 回环，会直接增加 TP4 通信时间，尤其是 Decode。
- 需要优先隔离或限速的，是写入 GPU 显存方向的 RDMA，也就是 D→P。

## 各方案的平均通信时间倍数

倍数相对同一方案 0% 负载。

| 方案 | 10% | 20% | 跑满后 |
|---|---|---|---|
| P→D，Prefill | 1.04 | 1.04 | 1.03～1.06 |
| P→D，Decode | 1.02 | 1.03 | 1.03 |
| D→P，Prefill | 1.48 | 3.74 | 3.5～3.8 |
| D→P，Decode | 1.75 | 27.5 | 18～30 |
| hicache H2D，Prefill | 0.91 | 0.85 | 0.85 |
| hicache H2D，Decode | 1.05 | 1.09 | 1.22 |
| NIC 回环 H2D，Prefill | 1.21 | 1.41 | 1.41 |
| NIC 回环 H2D，Decode | 1.45 | 2.27 | 3.9 |

两种模式的 0% 基线：

- Prefill 每次迭代平均约 125 ms，但 p99 约 205 ms，基线本身尾部就很大，所以 Prefill 的 p99 倍数不可靠，只看平均值。
- Decode 每次迭代平均约 7.9 ms，p99 约 10.7 ms。

完整数据见 `results/627-allreduce-interference/REPORT.md` 和 `summary.json`。

## 原因分析（推测，未专门验证）

- **D→P 与 P→D 的差别**：两者都经过 g5 与 g6 之间的交换机，区别在接收端。
  - P→D 写进 g6 的 host 内存，接收端很快。
  - D→P 写进 g5 的 GPU 显存，接收端被 GPU 的 PCIe 写入卡住。IB 的信用流控会把压力传回交换机和发送端网卡。
  - g6 自己的 TP4 跨 GPU 对通信（mlx5_0 → 交换机 → mlx5_4）和被卡住的 D→P 共用网卡发送队列和交换机缓冲，被堵在后面。小消息的 Decode all-reduce 因此受影响最严重。VL 隔离要解决的正是这类问题。
- **NIC 回环 H2D**：不出网卡，但和 NCCL 的 GDR 流量共用同一个 HCA 和 GPU 上游 PCIe。
- **hicache H2D**：只占用拷贝引擎和 GPU 的 PCIe，不进 HCA。Decode 受到的轻微影响来自 PCIe 竞争。
- **hicache H2D 让 Prefill 变快**：猜测是持续的 H2D 让 PCIe 链路和 GPU 保持在高功耗状态，延迟更低，未验证。

## 测试方法

每个节点起 4 个探针 rank 和 4 个流量生成器，每块 GPU 各一个。

**探针**

模拟 Qwen2.5-32B TP4 单个 engine 的通信：hidden 5120、64 层、bf16、NCCL，TP4 时不用 custom all-reduce。

| 节点 | 模式 | 每次迭代的通信 | 执行方式 |
|---|---|---|---|
| g5 | Prefill（一次 forward） | 129 次 all-reduce，每次 1004 token × 10 KiB = 9.8 MiB，外加 1 次 logits all-gather（每 rank 1 × 38016） | eager，不用 CUDA graph |
| g6 | Decode（一步） | 129 次 all-reduce，每次 batch 6 × 10 KiB = 60 KiB，外加 1 次 all-gather | CUDA graph |

- 1004 是 32K 数据集里每轮平均追加的 token 数，batch 6 来自实测的 decode 批大小。
- NCCL 环境变量：`NCCL_IB_HCA=mlx5_0,mlx5_4`、`NCCL_NET_GDR_LEVEL=PIX`。
- NCCL 走哪条路由开关 `DUAL_PATH_NCCL_FORCE_NIC` 控制，说明见 [docs/nccl-control.md](../docs/nccl-control.md)。
  - **本文的 627 和 630 两组数据都是在 `=0` 下测的**：同一对 GPU 之间（GPU0↔GPU1、GPU2↔GPU3）走 NVLink，只有跨 GPU 对的通信走 NET/IB GDR（mlx5_0 → 交换机 → mlx5_4）。
  - 脚本现在默认 `=1`：所有 TP4 通信都走网卡。

**流量生成器**

每块 GPU 一个，各自持有独立的 Mooncake TE 引擎。

| 方案 | 路径 | 发起方式 |
|---|---|---|
| P→D（`p2d`） | g5 GPU 显存 → g6 host 内存 | RDMA write，`batch_transfer_sync_write`，每个请求 32 KiB |
| D→P（`d2p`） | g6 host 内存 → g5 GPU 显存 | 同上 |
| hicache H2D（`hicache_h2d`） | 两节点各自 pinned host → GPU | 按 1 MiB 分块 `copy_(non_blocking=True)` |
| NIC 回环 H2D（`nic_h2d`） | 两节点各自 host → 配对的 CX8 → GPU | `batch_transfer_write_on_cuda`，异步挂在 CUDA stream 上，每个请求 32 KiB |

**负载档位**

- 目标负载按每块 GPU 分到的 CX8 份额计算：CX8 为 100 GB/s，两块 GPU 共用，每块 GPU 的份额是 50 GB/s，记为 100%。档位为 0%～100%，间隔 10%，每块 GPU 的目标是档位 × 50 GB/s。
- 以 10 ms 为一个节拍控制速率，最多 2 个节拍在途。
- 达不到目标时，生成器全速运行，记录实际达到的流量。图上画成空心点，表示实际流量低于目标的 90%。

**时间线**

- 两节点先对齐时钟，再按同一张时间表执行。
- 每个 case 25 s：
  - 前 4 s 为预热，其中最后 1.5 s 探针先不计时地空跑（Job 630 起才加入；627 没有这一步，所以 627 的 Prefill p99 基线被冷启动的第一次迭代拉高，不可用）；
  - 探针按启动时校准的次数（约 8 s）测量，到 case 结束前 1.5 s 为止；
  - 4 个 rank 通过一次小的 all-reduce 统一决定何时停止。
- 报告中“样本截断”列表示该 case 实际样本数少于计划，即重负载下迭代变慢。
- 每个 case 同时记录 HCA 计数器，确认流量确实经过了网卡。

## 复现

```bash
cd /data/home/yongxiao/dual-path
# 复现本文数据（同一对 GPU 之间走 NVLink）
DUAL_PATH_NCCL_FORCE_NIC=0 CASE_SECONDS=25 EXTRA_ARGS="--measure-seconds 8" nic-test/run_allreduce_interference.sh
# TP4 全部走网卡（脚本默认）
CASE_SECONDS=25 EXTRA_ARGS="--measure-seconds 8" nic-test/run_allreduce_interference.sh
```

- 如果当前不在 Slurm 分配内，脚本会自己用 salloc 申请 g5、g6（`--exclusive`），全程约 20 分钟。
- 可用环境变量 `SCHEMES`、`LEVELS`、`CASE_SECONDS`、`RESULT_DIR`、`DUAL_PATH_NCCL_FORCE_NIC`、`NCCL_DEBUG` 覆盖默认值。
- `PROBE_SL` 设置 TP4 探针的 `NCCL_IB_SL`，`TRAFFIC_SL` 设置流量生成器的 `MC_IB_SL`（p2d、d2p、nic_h2d），取值 0～3，默认 -1 表示不设（SL0）。每个节点的取值记录在 `node<N>-config.json`。
- 镜像使用 nich2d 版本：`sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-l3bps-20260929-nich2d.sqsh`。
- 依赖 `io-test/` 下的 `capture_path_counters.py` 和 `cluster_barrier.py`。

## 文件

| 路径 | 内容 |
|---|---|
| `allreduce_interference_bench.py` | 探针和流量生成器，每个节点一个 task |
| `run_allreduce_interference.sh` | 驱动脚本：srun、容器、NCCL 环境变量，跑完后画图 |
| `plot_allreduce_interference.py` | 生成 `interference.svg`、`summary.json`、`REPORT.md` |
| `d2p_forward_bench.py`、`run_d2p_forward.sh`、`plot_d2p_forward.py` | D→P 对完整 forward 的影响（Job 635） |
| `analyze_real_d2p.py` | 从真实运行的服务端日志和客户端结果重建时间线，检查 D→P 干扰 |
| `summarize_vl_pd.py` | 按 Job 汇总 D→P / NIC H2D 的 TTFT 占比、占空比，以及 D→P 干扰折算到 JCT 的比例（结果见 `docs/vl-traffic-isolation.md` §6.1） |
| `plot_vl_report.py` | 生成 `docs/vl_report.md` 用的三张图（`docs/figures/vl-report-*.{svg,png}`） |
| `results/635-d2p-forward/` | D→P 对完整 forward 的结果；`635-d2p-forward-cal-t2`、`-t4` 和 `632-d2p-forward-cal-t1` 是不同线程数的最大速率校准；`632-d2p-forward-cal-t2`、`-t4` 是挂死作业的残留，作废 |
| `results/627-allreduce-interference/` | 有效的完整矩阵结果 |
| `results/649-allreduce-interference/`、`results/650-allreduce-interference/` | VL 隔离对照：无 VL / 探针 VL0 + 流量 VL1 |
| `results/630-allreduce-interference-warmup-quick/` | 加入探针 warmup 后的 0%/100% 快速测试，基线正常 |
| `results/631-tp4-net-only-baseline/` | `DUAL_PATH_NCCL_FORCE_NIC=1` 的无负载基线，附 `NCCL_DEBUG=INFO` 日志 |
| `results/627-allreduce-interference-smoke/` | 截断逻辑的冒烟测试（D→P、NIC 回环 H2D，0% 和 20%） |
| `results/623-allreduce-interference-invalid-lag/` | **作废**：探针迭代次数固定，重负载下超出 case 窗口，延迟累积，测到的是错位的 case |
| `results/621-*`、`results/623-*-smoke/` | 早期冒烟测试 |
