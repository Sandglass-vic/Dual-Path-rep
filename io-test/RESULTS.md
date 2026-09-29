# Decode KV Read Mini Bench Results

下面的 pza100 结果来自 g5/g6 A100 集群。再往后的 GAIA 结果是 DGX H100 + Lustre 的
历史基线，两者的存储拓扑完全不同，不能互相比较。

---

# pza100（g5 / g6）

## 2026-09-29: H2D/D2H 路径 mini bench（Job 597 / 598 / 599，g5）

`io-test/run_h2d_path_bench.sh`，设计见 [docs/nic-h2d.md](../docs/nic-h2d.md)。原始数据
和完整表格在 `io-test/results/597-h2d-path/`（`REPORT.md`、`summary.json`）。Job 597
跑完了 A1 的 GPU0 部分以及 A2 的 1/2 GPU 并发；Job 598 补跑 4 GPU 并发和 SL2；
Job 599 补跑 GPU2/`mlx5_4` 的 A1 和 A4。

### A1：perftest 同网卡回环（`ib_write_bw`，4 s）

| 路径 | GPU0 / `mlx5_0` | GPU2 / `mlx5_4` | 说明 |
|---|---:|---:|---|
| h2d（host → GPU，NIC 写 GPU） | 17.9 GB/s | 18.4 GB/s | 4 KiB–4 MiB、QP 1/4/16 全部相同 |
| d2h（GPU → host，NIC 读 GPU） | 9.5–14.2 GB/s | 10.9–14.5 GB/s | 4 KiB 且 1 QP 时最高，其余约 10–11 GB/s |
| h2h（host → host） | 54.6 GB/s | 54.6 GB/s | 4 KiB 时只有 22.7 GB/s（包速率受限） |
| 两块 GPU 共享一块 HCA，h2d | 28.7 GB/s（合计） | 28.7 GB/s（合计） | 每块 GPU 约 14.3 GB/s |
| 两块 GPU 共享一块 HCA，d2h | 27.7 GB/s（合计） | — | |
| `ib_write_lat --write_with_imm` | 3.0 / 5.9 / 21.7 µs | — | 4 KiB / 32 KiB / 256 KiB |

- h2h 达到 NIC 的 Gen5 x16 上限，所以 h2d 停在约 18 GB/s 并不是网卡本身的瓶颈，
  而是 NIC 对 A100 显存做 P2P 写的上限（A100 的 Gen4 x16，拷贝引擎能跑到约
  26.8 GB/s）。NIC 读显存（d2h）更慢。
- SL 2 与 SL 0 的结果相同。SM 当前没有启用 QoS，这符合预期。
- **计数器**：所有回环 case 的 `hw.rx_write_requests` 都与请求数一致，
  `port_rcv_data`/`port_rcv_packets` 随数据量增长，而 `port_xmit_*` 增量为 0。
  即数据在 HCA 内部回环，没有经过交换机。经过 GPU 的 case，dmon 积分的 PCIe 字节
  与数据量之比为 0.77–1.08（1 s 采样的边界误差）；h2h 为 0。

### A2：Qwen2.5-32B TP4 形状（每 rank 每次 512 页 × 64 层 = 2 GiB，bf16）

单 GPU（GPU0），每块 GPU 的 GB/s：

| host 布局 | 页分布 | 方向 | hicache（direct） | hicache-kernel | nic（Mooncake） | nic / hicache |
|---|---|---|---:|---:|---:|---:|
| layer_first | 连续 | h2d | **26.8** | 12.7 | 16.0 | 0.60 |
| layer_first | 连续 | d2h | **26.2** | 12.7 | 10.0 | 0.38 |
| layer_first | 随机 | h2d | 8.8 | 12.7 | **13.0** | 1.48 |
| layer_first | 随机 | d2h | 9.5 | **11.8** | 8.9 | 0.94 |
| page_first_direct | 连续/随机 | h2d | 6.4 | — | **13.0** | 2.04 |
| page_first_direct | 连续/随机 | d2h | 7.2 | — | **8.9** | 1.24 |

- hicache（direct）每个连续段调用一次 `cudaMemcpyAsync`。连续的 `layer_first` 每层
  只有两段，因此能打满 GPU 链路；一旦每页一段（随机页，或 Decode 使用的
  `page_first_direct`，每层 1024 段），每层耗时升到 3.8–5.3 ms，提交几乎全部花在
  CPU 上（提交耗时约等于总耗时）。
- nic 的每层耗时与段数关系不大：`page_first_direct` 每层 2.58 ms，
  `layer_first` 连续时 2.09 ms。对比 perftest 的 h2d 上限，32 MiB 每层的理想时间是
  1.87 ms，即 Mooncake 对 1024 个 32 KiB 请求约有 0.7 µs/请求的额外开销。
- nic 的 CPU 时间约 0.4–0.6 s/次（2 GiB），比 hicache 连续时的 0.08 s 高；这部分
  主要花在 Mooncake 的 worker 线程上。
- 并发：GPU0+GPU1 共享 `mlx5_0` 时，nic 每块 GPU 为 11.7–13.3 GB/s；4 块 GPU 同时
  运行时，nic 每块约 11.6 GB/s（h2d，合计约 47 GB/s），hicache `page_first_direct`
  每块约 6.2 GB/s。4 块 GPU 的 worker 都绑定在各自的 NUMA 节点上。
- 所有 case 的首轮数据校验都通过；nic 的 case 全部判定为经过 HCA，hicache 全部判定
  为未经过 HCA。
- `MC_IB_SL=2` 的结果与 SL 0 相同（QoS 未启用）。

### A4：H2D 负载下的 TP4 all-reduce（4 GPU 同时 H2D，`layer_first` 连续，30 s）

| 负载 | 8 KiB 延迟 | 1 MiB 延迟 | 64 MiB busbw |
|---|---:|---:|---:|
| 空闲 | 45 µs | 180 µs | 14.2 GB/s |
| hicache | 51 µs | 158 µs | 19.8 GB/s |
| nic（SL 0） | **118 µs** | **313 µs** | 13.4 GB/s |

nic 负载使小消息 all-reduce 延迟升到空闲时的 1.7–2.6 倍，因为 TP4 的跨对 NCCL
流量（`NET/IB`）与 KV 回环共用同一块 CX8，而当前没有 VL 隔离。hicache 负载下
64 MiB 的 busbw 反而高于空闲，原因未查明，可能是 GPU 频率或电源状态的差异。
每组只测了一次，且 probe 与负载的重叠时间靠 15 s 延迟估计，所以这一节只作为
方向性证据。

### 决策门槛与结论

- 门槛（单 GPU；nic 时间 ≤ 1.2 × hicache 或带宽 ≥ 0.8 × hicache，且经过 HCA、
  校验通过）：`page_first_direct` 的 h2d/d2h 以及 `layer_first` 随机页全部通过；
  `layer_first` 连续页的 h2d（0.60×）和 d2h（0.38×）不通过。
- 实现载体：第一版采用 Mooncake TE。它在连续 h2d 上达到 perftest 上限的 90%，在
  32 KiB 散页上达到 73%。自研 verbs 扩展在 Decode 散页场景最多还能提升约 38%，
  列为后续优化。
- nic 路径本身不能减少对 NCCL 的干扰，反而把干扰集中到了网卡上。只有在 SM 启用
  VL QoS、KV 使用低优先级 SL 之后，它才有意义。所以 `DUAL_PATH_CUDAMEMCPY=nic`
  应该与 [VL 方案](../docs/vl-traffic-isolation.md) 一起评估，不单独作为性能优化。

## 环境与理论上限

2026-09-18 从 sysfs 采集，`io-test/capture_io_state.py` 会把同样的内容写进每次
运行的 `REPORT.md`。

| 环节 | 实际配置 | 理论上限 |
|---|---|---:|
| 存储网卡 `ibs3` = `mlx5_0`（ConnectX-8，`0000:47:00.0`） | IB 4X XDR，ACTIVE | 800 Gb/s = 93.13 GiB/s |
| 该网卡的 PCIe 链路 | Gen5 x16（`32.0 GT/s` × 16，槽位支持 Gen6） | 58.69 GiB/s |
| g5 `/data` 背后的盘 | Solidigm `SSDPF2NV307TZ`（D5-P5316 30.72 TB），PCIe Gen4 x4 | 7.34 GiB/s |

存储路径：

```text
g5 /data: /dev/nvme2n1, XFS, 4 KiB block
g6 /data: 192.168.10.5:/data, NFSv4.1, proto=tcp, rsize=wsize=1 MiB,
          hard, 挂载选项里没有 nconnect（即默认 1 条 TCP 连接）
ibs3:     IPoIB datagram 模式, MTU 2044
```

IB 链路速率**不是**这里的存储上限。NFS 走 TCP、不是 RDMA，IPoIB 又是 datagram
模式 MTU 2044，所以实际可达速率由内核 TCP/IP over IPoIB 决定，见下面 Job 446。

## 2026-09-18: 四象限 Job 431（g5）与 Job 442（g6）

`io-test/run_hot_cache.sh`，TP4、每 rank 一个独立进程、4 MiB/page、每象限 8 GiB、
3 轮，pinned DRAM + torch `page_first -> layer_first` staging。文件 I/O
（open + readinto + close）与 scatter（staging 拷贝）分别计时。两台的 `mincore`
探针都确认 O_DIRECT 绕过了 page cache，**包括 NFS**。

g5（本地 XFS，冷读真的到 NVMe）：

| 象限 | 端到端 GiB/s | 纯文件 I/O | scatter | I/O 占比 | scatter 占比 | 单页读 p50 / p99 | CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| cold-buffered | 6.432 | 7.141 | 63.73 | 90.2% | 10.1% | 2.166 / 2.627 ms | 0.3% |
| cold-direct | 6.432 | 7.510 | 40.53 | 85.6% | 15.9% | 2.048 / 2.383 ms | 0.0% |
| hot-buffered | 21.595 | 42.048 | 44.26 | 51.4% | 49.4% | 0.335 / 0.424 ms | 0.7% |
| hot-direct | 6.432 | 7.556 | 41.98 | 85.1% | 15.3% | 2.034 / 2.383 ms | 0.1% |

g6（NFS client；`DROP_PEER_CACHE=0`，即 g5 的 server cache 是热的，所以 cold
描述的是**存储网络路径**，不是 NVMe）：

| 象限 | 端到端 GiB/s | 纯文件 I/O | scatter | I/O 占比 | scatter 占比 | 单页读 p50 / p99 | CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| cold-buffered | 2.870 | 3.008 | 65.07 | 95.4% | 4.6% | 4.835 / 6.705 ms | 3.8% |
| cold-direct | 2.027 | 2.126 | 45.66 | 95.4% | 4.4% | 7.273 / 9.373 ms | 0.3% |
| hot-buffered | 20.682 | 39.093 | 46.26 | 53.5% | 45.3% | 0.293 / 0.375 ms | 0.8% |
| hot-direct | 1.969 | 2.060 | 45.50 | 95.6% | 4.4% | 5.493 / 6.693 ms | 1.9% |

三条结论。

**hot-buffered 两台几乎相同（21.595 vs 20.682，95.8%），这是对照组成立的证明。**
热缓存读就是节点本地一次 DRAM memcpy，与存储路径无关，本来就该相等。它同时给出了
page cache 的读取上限。

**O_DIRECT 的收益在两台机器上符号相反。** g5 上 `cold-direct / cold-buffered
= 1.000x`（都被盘卡住），纯文件 I/O 甚至略好（7.510 vs 7.141）。g6 上是
`0.706x`——NFS 的 O_DIRECT 会关掉 readahead，而每 rank 队列深度只有 1，没有东西
掩盖 RPC 往返。更极端的是 g6 的 `hot-direct / hot-buffered = 0.095x`：对已在客户端
缓存里的数据用 O_DIRECT，等于扔掉 10 倍性能。因此 `DUAL_PATH_O_DIRECT_READ`
不应全局开关，要按节点存储角色区分。

**scatter 开销两台完全一致，但占比差 10 倍。** 单页 scatter 都是 0.23–0.35 ms
（40–65 GiB/s），因为它是纯 CPU/DRAM 操作。但 g5 的 hot-buffered 里 scatter 占
49.4%，g6 走存储的三个象限里只占 4.4%。也就是说 g5 上的优化目标是 staging 拷贝，
g6 上是网络路径。

已知疑点：g5 纯文件 I/O 量到 7.510–7.556 GiB/s，比 PCIe Gen4 x4 的载荷上限
7.34 GiB/s 高约 3%，物理上不该发生。可能是这块 D5-P5316 为双端口而 sysfs 只报了
一个控制器的链路，也可能是 128b/130b 折算偏保守。结论上链路确已打满，但那 3%
不作为真实余量。

## 2026-09-18: g6 存储路径并发扫描 Job 446

`io-test/run_odirect_sweep.sh`，单文件顺序读，每进程 512 MiB，2 轮。目的是回答
"这条路径最多能做到多少"，而不是"Decode 能拿到多少"。

4 MiB block 下的并发扫描（括号内为单块 p50）：

| jobs | 总量 GiB | direct GiB/s | buffered-cold GiB/s |
|---:|---:|---:|---:|
| 1 | 0.5 | 2.658 (1.437 ms) | 2.803 (1.442 ms) |
| 2 | 1.0 | 2.856 (2.756 ms) | 3.434 (2.232 ms) |
| 4 | 2.0 | 3.005 (5.180 ms) | 3.436 (4.445 ms) |
| 8 | 4.0 | **3.295** (9.613 ms) | **3.469** (8.414 ms) |
| 16 | 8.0 | 3.106 (20.821 ms) | 3.385 (18.105 ms) |
| 32 | 16.0 | 3.020 (42.920 ms) | 3.200 (38.508 ms) |
| 64 | 32.0 | 3.139 (80.386 ms) | 3.272 (73.511 ms) |
| 128 | 64.0 | 3.245 (150.825 ms) | 3.214 (153.314 ms) |

block size 的影响只在低并发、且主要对 O_DIRECT：

| | 256K | 1M | 4M |
|---|---:|---:|---:|
| direct, 1 job | 1.314 | 1.869 | 2.658 |
| direct, 8 jobs | 2.890 | 3.193 | 3.295 |
| buffered, 1 job | 2.608 | 3.319 | 2.803 |
| buffered, 8 jobs | 2.731 | 3.332 | 3.469 |

**天花板是 3.2–3.5 GiB/s = 27–30 Gb/s，即 800 Gb/s 端口的 3.7%、HCA PCIe 的
5.9%，连 g5 那块 NVMe 都只用到 47%。盘完全不是瓶颈。**

并发加不上去：1M block 下 8 进程 3.332、128 进程 3.337（+0.2%），而单块 p50 从
2.263 ms 涨到 34.554 ms——15.3 倍，进程数是 16 倍。多加的并发 100% 变成排队。
4M、128 进程时 p50 已到 151 ms、p95 190 ms，对 Decode 是灾难性尾延迟，所以提高
并发不只是无效，是有害。

决定性证据是**单进程就能拿到 3.319 GiB/s**（buffered、1M），等于 128 进程那个数的
99.5%。若瓶颈是分散在多队列上的包处理能力，加进程必然有收益；只有当瓶颈是一个
**串行共享资源**时才会"一个进程就打满"。那个资源就是 `nconnect=1` 的单条 TCP 连接。

两条线索还会叠加：3.5 GiB/s = 3.76 GB/s，MTU 2044 下约 1.9 M 包/秒，全走一条 TCP
流、由一个 CPU 核的 softirq 处理，单核 1.9 Mpps 差不多就是极限。

### 与四象限的差距

| | g6 GiB/s |
|---|---:|
| 四象限 cold-direct（4 rank，逐页 open/read/close，2048 个 4 MiB 文件） | 2.027 |
| 扫描 direct（4 进程 × 4M，单文件顺序读） | 3.005 |
| 扫描 direct 最佳（8 进程 × 4M） | 3.295 |

同样 4 MiB 的读，四象限里单页 7.310 ms，扫描里单块 5.180 ms，慢 41%。**不是**每页
一次 NFS OPEN/CLOSE 的开销：实测 open 0.030 ms、close 0.005 ms，合计仅占文件 I/O
的 0.5%。更可能是 NFS 客户端对单文件顺序读能持续流水 RPC，而对 2048 个各读一次的
独立小文件不行。此项待单独实验确认，暂不作为结论。

方向上：DualPath 的逐页文件访问形态只能拿到这条路径流式上限的 63%，而流式上限本身
只是网卡能力的 3.7%。

### 待验证的改动

按预期收益排序；第 3 项已于 2026-09-29 实测（见下面 Job 563），其余未测：

1. `nconnect=8` 或 `16`（内核 5.3+，最大 16；g6 是 6.12）。挂载时选项，**按服务器
   生效且第一个挂载点说了算**，所以要把 g6 上挂自 `192.168.10.5` 的 `/data` 和
   `/home/pengzhi` 全部卸掉再重挂。预期低并发点几乎不变、高并发点显著上抬；若连
   高并发点也不动，则瓶颈在别处（如服务器端 nfsd 线程池或 IPoIB 收包路径）。
2. IPoIB 切 connected 模式，MTU 2044 → 65520，包数降约 32 倍。
3. `proto=rdma`，彻底绕开内核 TCP/IP 栈。卡与 OFED 均支持。**已实测**：读带宽提升
   1.8–2.0 倍，大文件读打满 NVMe。

A/B 方式：改动前后用**完全相同**的命令重跑 `run_odirect_sweep.sh`，对比两份
`REPORT.md`。Job 446 就是改动前的基线。

### 尚未测量

g6 在 `DROP_PEER_CACHE=1` 下的四象限，即 g5 也清掉缓存、读请求一路打到 NVMe。按
现有数据预测该档与 2.027 GiB/s 接近——网络已比盘慢三倍以上，盘的 6.4 GiB/s 藏在
网络后面露不出来。（Job 559 已在 `ibs4` 上部分回答：逐页 O_DIRECT 冷热只差 3%。）

## 2026-09-29: `ibs4` 上的 Decode 逐页读复现 Job 559

g6 的 NFS 挂载已从 `ibs3`（CX8）迁到 `ibs4`（CX7，`192.168.11.5:/data`），仍为
NFSv4.1/TCP、`rsize=wsize=1 MiB`、`nconnect=1`、datagram MTU 2044。
`scripts/smoke/g6_decode_read_repro.sbatch`：TP4、每 rank 324 页（真实 Decode 请求
median）× 4 MiB page 文件、每次 5.4 GB，O_DIRECT + pinned + torch staging，与引擎
相同；`hot` 为 g5 server cache 热，`cold` 为读前在 g5 上 `fsync` + `DONTNEED`。3 轮：

| server cache | 端到端 GB/s | 文件 I/O GB/s | 每 rank GB/s | 单页读 p50 | 各轮 GB/s |
|---|---:|---:|---:|---:|---|
| cold | 2.038 | 2.134 | 0.511 | 7.63 ms | 2.038, 2.049, 2.026 |
| hot | 2.095 | 2.196 | 0.525 | 7.61 ms | 2.079, 2.095, 2.097 |

- 与 Job 442（`ibs3`，cold-direct 2.027 GiB/s、7.27 ms/页）同一水平，迁移到 CX7
  没有让这种读法变慢。
- 同日 Allocation 557 的真实 Decode path 为 1.23 GB/s、13.6 ms/页（四个 rank 一致），
  mini bench 复现不了，差异只出现在真实负载下，原因未定。
- 逐页形态只拿到同路径大文件顺序读（Job 555：4 进程 3.575 GB/s）的约 58%。每页
  多出的 2.5–3 ms 不来自 open/close（合计 0.5%）和 g5 盘（冷热差 3%），见上节
  "与四象限的差距"。

结果：`io-test/results/g6-decode-read-repro-559/`。

## 2026-09-29: NFS/TCP 与 NFS/RDMA A/B Job 563

g6 额外挂载同一导出的 NFS/RDMA 视图：`192.168.11.5:/data` →
`/mnt/nfsrdma`，`vers=3,proto=rdma,port=20049`，由 `scripts/pza100-nfsrdma.sh`
管理。必须用 v3：v4.1 会把新挂载合并到已有 `/data` 的 TCP 会话（session
trunking），`/proc/mounts` 显示 `proto=tcp`。`/proc/self/mountstats` 中 `xprt: rdma`
确认传输。

`scripts/smoke/g6_nfs_rdma_ab.sbatch`，宿主机运行、不进容器；g5 准备数据，g6 交替
读 `/data` 与 `/mnt/nfsrdma` 下的同一批文件，每次读前清 g5 server cache，3 轮 median：

| 读法 | TCP | RDMA | 提升 | 单次读 p50 |
|---|---:|---:|---:|---:|
| 逐页：4 rank × 324 × 4 MiB 文件，O_DIRECT | 3.237 GB/s | 5.824 GB/s | 1.8× | 5.20 → 2.79 ms |
| 大文件：8 进程 × 4 MiB block，O_DIRECT | 3.443 GB/s | 6.931 GB/s | 2.0× | 9.91 → 4.66 ms |

- RDMA 大文件读等于 g5 NVMe 本地上限（Job 555：6.95 GB/s），瓶颈从单条 TCP 连接
  转到盘。
- 本组缓冲为 pageable 内存、无 staging 拷贝，TCP 逐页 3.237 GB/s 高于 Job 559 的
  2.04 GB/s（pinned + staging），两组 TCP 数值不能直接比较；pinned 内存在 RDMA
  上的表现未单独测。
- 4 进程大文件一组因脚本循环变量被 `drop_all` 覆盖而未运行，脚本已修复。
- 本组每次读前都清 g5 server cache，只代表"读 NVMe"。真实负载中 Decode 读的是
  刚经 NFS 写回、仍在 g5 page cache 中的 KV（nfsd buffered 读，客户端 O_DIRECT
  无法绕过），见下面 Job 584。在 TCP 下（Job 559）冷热只差 3%，是因为 TCP 本身
  更慢，掩盖了 server cache 的作用；这一结论不能推广到 RDMA。

结果：`io-test/results/g6-nfs-rdma-ab-563/`。PD 运行接入方式（`DUALPATH_NFS_RDMA`）
见 [docs/pza100.md](../docs/pza100.md)。

## 2026-09-29: g5 page cache 热读 Job 584

`scripts/smoke/g5_g6_hot_read.sbatch`：引擎同款读法（TP4 × 324 页 × 4 MiB = 5.4 GB，
pinned + torch staging）。两套互不重叠的数据都在 g5 上准备并以 buffered 读预热进
g5 page cache；g5 本地 buffered 读、不清 cache，g6 经 `/mnt/nfsrdma` O_DIRECT 读，
即 nfsd 从 g5 page cache 返回。同时读两端经 `--sync-host g5` 握手对齐。3 轮 median：

| 场景 | 节点 | 端到端 GB/s | 文件 I/O GB/s | 单页读 p50 | 各轮端到端 GB/s |
|---|---|---:|---:|---:|---|
| 单独 | g5 | 24.149 | 51.910 | 0.29 ms | 23.864, 24.153, 24.149 |
| 单独 | g6 | 13.680 | 21.539 | 0.63 ms | 13.011, 13.815, 13.680 |
| 同时 | g5 | 22.730 | 48.049 | 0.31 ms | 22.197, 22.730, 23.204 |
| 同时 | g6 | 13.550 | 21.693 | 0.63 ms | 13.887, 13.458, 13.550 |

同时读合计 36.1 / 36.2 / 36.8 GB/s。

- 引擎的 L3 读带宽只计文件读（不含 scatter/staging），对应"文件 I/O"列：g6 的
  21.5 GB/s 与 Job 581/582 真实 Decode 的 20.0–20.4 GB/s 一致，证实 Decode 读的是
  g5 page cache。
- g5 buffered 热读的文件 I/O 为 51.9 GB/s，约为 O_DIRECT 读 NVMe（Job 431：
  7.5 GiB/s）的 6.5 倍；端到端差距较小是因为 staging 拷贝约占 g5 一半时间。
- g6 的 21.5 GB/s 约为 CX7 400G 线速（50 GB/s）的 43%，限制在 RDMA/nfsd 路径而
  不是内存；两端同时读互相影响只有 1–6%。
- 每次读仅 0.24–0.40 s：同时读的重叠覆盖了 g5 全程，但只覆盖 g6 约 60%，g6 的
  干扰可能被低估。更长的测量可把 `PAGES_PER_RANK` 调到约 1500（约 25 GB/次）。

结果：`io-test/results/g5-g6-hot-read-584/`。

---

# GAIA 历史基线（DGX H100 + Lustre）

## 2026-09-15: Hot page cache versus O_DIRECT Job 68988

在 `dgx-gaia-17` 上对同一批 8 GiB KV page 重复执行以下顺序：

```text
DONTNEED -> buffered populate -> buffered hot -> O_DIRECT over hot cache
           -> buffered read after O_DIRECT
```

| Repetition | Buffered populate | Buffered hot | O_DIRECT over hot cache | Buffered after O_DIRECT |
|---:|---:|---:|---:|---:|
| 1 | 3.936 GiB/s | 14.551 GiB/s | 5.174 GiB/s | 3.772 GiB/s |
| 2 | 3.931 GiB/s | 14.530 GiB/s | 6.270 GiB/s | 11.418 GiB/s |
| 3 | 3.975 GiB/s | 14.470 GiB/s | 6.281 GiB/s | 11.322 GiB/s |

中位数：

- cold buffered populate：`3.936 GiB/s`；
- hot buffered：`14.530 GiB/s`；
- page cache 已为 hot 时的 `O_DIRECT`：`6.270 GiB/s`。

`O_DIRECT over hot / buffered hot = 43.2%`。因此即使同一文件已经完整驻留在客户端
page cache 中，`O_DIRECT` 也不会命中该 DRAM hot copy，而是继续执行 Lustre Direct
I/O。它仍比本轮 cold buffered 快约 `1.59x`，但明显慢于 hot buffered。

`O_DIRECT` 后再次 buffered read 的结果有较大波动，说明在同一文件上混合 Direct 和
buffered I/O 时，缓存驻留或客户端内存压力可能发生变化；生产设计不应假设 Direct read
之后原有 hot cache 一定保持完整，需要单独采集全文件 residency 才能判断具体机制。

## 2026-09-15: O_DIRECT Job 68987

节点 `dgx-gaia-17`，配置与 Decode file backend 对齐：TP=8、每 rank 独立进程、
2 MiB/page、总计 8 GiB/read、pinned DRAM，以及 Torch
`page_first -> layer_first` staging copy。

客户端 page-cache 驻留探针结果：

| Stage | Resident 4 KiB pages |
|---|---:|
| `DONTNEED` 后 | 0 / 512 |
| `O_DIRECT` read 后 | 0 / 512 |
| 再次 `DONTNEED` 后 | 0 / 512 |
| buffered read 后 | 512 / 512 |

因此本次实验确认 Lustre `O_DIRECT` read 绕过了计算节点 Linux page cache：数据没有
被填充到该文件的客户端缓存页，而相同 buffer 和文件上的 buffered read 会填充全部
512 个页。这里验证的是 Lustre kernel client 上方的 page-cache bypass，不是绕过
Lustre kernel client 的 3FS USRBIO 式完整 kernel bypass。

Cold-read 带宽：

| Repetition | Buffered GiB/s | O_DIRECT GiB/s |
|---:|---:|---:|
| 1 | 3.933 | 5.540 |
| 2 | 3.971 | 6.096 |
| 3 | 3.911 | 6.011 |

中位数为 `3.933 GiB/s` 和 `6.011 GiB/s`，本次 `O_DIRECT / buffered` 中位数比值
为 `1.529x`。这个结果说明 Direct I/O 在该节点和当时 Lustre 负载下有效且可运行，
但它会放弃同节点后续轮次可利用的 page-cache 热读，因此不能仅凭 cold-read 提升就
直接替换生产路径；仍需针对完整 trajectory 做 cold/warm 和 Decode 节点亲和性 A/B。

## 2026-09-15: Job 68967

配置：

- readers: `dgx-gaia-17`, `dgx-gaia-18`
- TP ranks per node: 8
- one independent process and one Torch CPU thread per TP rank
- page size: 2 MiB per rank（64 tokens）
- data per node and phase: 8 GiB（one 32K request）
- target: pinned DRAM
- `HiCacheFile.get()` 等价的 unbuffered `readinto()`
- 与 SGLang 相同的 Torch `page_first -> layer_first` tensor assignment

### Cold Lustre

每轮前执行本地 `posix_fadvise(DONTNEED)`：

| Repetition | 1-node GiB/s | 2-node aggregate GiB/s | Scaling | 2-node per-node retention |
|---:|---:|---:|---:|---:|
| 1 | 3.858 | 6.477 | 1.679x | 93.2% |
| 2 | 3.914 | 7.101 | 1.814x | 96.0% |
| 3 | 3.934 | 6.838 | 1.738x | 91.6% |

中位数 aggregate scaling 为 `1.738x`，中位数 per-node retention 为 `93.2%`。
这是持续冷读 Lustre 的带宽，不应直接与正式实验中受 page cache 影响的单 request
有效带宽比较。

因此同时增加第二个冷读 reader 会产生一定波动和非线性扩展，但没有让原 reader 的
带宽下降到约一半。该竞争不能单独解释正式实验中 1P2D Decode KV Read 平均耗时相对
1P1D 增长约 `1.93x`。

### Cache-preserving

不执行 `DONTNEED`，由 `dgx-gaia-17` 创建文件后立即读取：

- writer 节点第一次单节点读取：`14.52 GiB/s`；
- 双节点第一次读取时 writer 节点：`9.05 GiB/s`；
- 双节点第一次读取时 remote 节点：`0.84 GiB/s`；
- remote 节点完成一次读取后，后续达到 `9.68-12.24 GiB/s`。

`14.52 GiB/s` 与正式 1P1D 的 `13-14 GB/s` 对齐。结果证明 4 GiB/s 是清除本地
缓存后的冷 Lustre 带宽，而不是 TP rank 并行度错误；正式实验的高带宽明显包含
Decode writer 节点的本地 page-cache 局部性。

## 与正式实验关联

正式实验：

- 1P1D Job 68945：Decode KV Read 平均 `357 ms`，平均有效带宽 `13.09 GB/s`；
- 1P2D Job 68940：Decode KV Read 平均 `691 ms`，平均有效带宽 `8.55 GB/s`。

按 trajectory 相邻 turn 的 Decode node 归属拆分 1P2D：

| Routing locality | Requests | Mean read | Median read | Mean effective bandwidth |
|---|---:|---:|---:|---:|
| same Decode node | 116 | 464 ms | 505 ms | 12.43 GB/s |
| switched Decode node | 84 | 1068 ms | 948 ms | 4.01 GB/s |

切换节点的请求虽然平均读取 token 更少，耗时仍更长。按 0-8K、8-16K、16-24K、
24K+ token 分桶后，切换节点的有效带宽仍显著更低。

当前最符合证据的解释是 Decode round-robin 打破了 trajectory 的节点亲和性。Decode
写入 HiCache 文件后，写入节点可能保留相关本地 file page cache；下一 turn 留在同一
节点可获得较高读取带宽，切到另一 Lustre client 节点则需要重新获取数据。共享 Lustre
总带宽竞争是次要因素，不是观测到接近 2 倍读取耗时的主要原因。

后续应使用 trajectory-sticky Decode routing 做直接 A/B 验证，并同时记录同节点与
跨节点 turn 的 file-cache 命中或块设备/网络读取量。
