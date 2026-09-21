# Decode KV Read Mini Bench Results

下面的 pza100 结果来自 g5/g6 A100 集群。再往后的 GAIA 结果是 DGX H100 + Lustre 的
历史基线，两者的存储拓扑完全不同，不能互相比较。

---

# pza100（g5 / g6）

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

按预期收益排序，均未实测：

1. `nconnect=8` 或 `16`（内核 5.3+，最大 16；g6 是 6.12）。挂载时选项，**按服务器
   生效且第一个挂载点说了算**，所以要把 g6 上挂自 `192.168.10.5` 的 `/data` 和
   `/home/pengzhi` 全部卸掉再重挂。预期低并发点几乎不变、高并发点显著上抬；若连
   高并发点也不动，则瓶颈在别处（如服务器端 nfsd 线程池或 IPoIB 收包路径）。
2. IPoIB 切 connected 模式，MTU 2044 → 65520，包数降约 32 倍。
3. `proto=rdma`，彻底绕开内核 TCP/IP 栈。卡与 OFED 均支持。

A/B 方式：改动前后用**完全相同**的命令重跑 `run_odirect_sweep.sh`，对比两份
`REPORT.md`。Job 446 就是改动前的基线。

### 尚未测量

g6 在 `DROP_PEER_CACHE=1` 下的四象限，即 g5 也清掉缓存、读请求一路打到 NVMe。按
现有数据预测该档与 2.027 GiB/s 接近——网络已比盘慢三倍以上，盘的 6.4 GiB/s 藏在
网络后面露不出来。

---

# GAIA 历史基线（DGX H100 + Lustre）

## 2026-09-21: O_DIRECT 双节点聚合扩展 Job 69346（direct）与 Job 69347（buffered）

问题：O_DIRECT 下两个节点的聚合带宽是不是单节点的 2 倍。

节点 `dgx-gaia-09` / `dgx-gaia-10`（exclusive），`io-test/run.sh`，TP8、每 rank 一个
独立进程、2 MiB/page、每节点每相 8 GiB、3 轮、每轮前 `DONTNEED`。与 Job 68967 的
区别是这次**只测文件 I/O**：`USE_CONTAINER=0 BUFFER_TYPE=pageable
COPY_TO_STAGING=0`，不带 pinned DRAM 和 torch staging 拷贝，因此不需要镜像和 GPU。
两个节点读的是同一批 page 文件，与 68967 一致。

| io_mode | 单节点 GiB/s | 双节点聚合 GiB/s | aggregate scaling | per-node retention | 单页 read p50 |
|---|---:|---:|---:|---:|---:|
| `direct` | 6.973 | 13.606 | **1.935x** | 97.5% | 1.835 → 1.875 ms |
| `buffered` | 3.995 | 8.015 | **1.988x** | 102.8% | 3.522 → 3.549 ms |

均为 3 轮中位数；scaling 逐轮为 direct `1.935 / 1.983 / 1.918`、buffered
`1.988 / 1.998 / 1.958`。

**结论：是 2 倍。** O_DIRECT 下双节点聚合 `1.935x`，每节点保持率 97.5%，单页读
p50 从 1.835 ms 只涨到 1.875 ms（+2.2%），第二个 reader 既没有抢走带宽也没有制造
排队。buffered 对照给出同样的形状（`1.988x`），说明这个线性度不是 O_DIRECT 特有的，
而是这段区间里 Lustre 后端对两个 client 还远未饱和。

两节点合计 13.6 GiB/s 仍低于 Job 67340 在单节点用 fio（8 job × iodepth 16、4 MiB）
打到的 22.98 GiB/s 读取峰值，也就是说 TP8 × 队列深度 1 的逐页访问形态本身才是限制，
不是 Lustre 的总吞吐。因此在这个量级上继续加 Decode 节点仍能线性拿到带宽。

O_DIRECT 相对 buffered 快 `1.75x`（6.973 vs 3.995），比 Job 68987 同节点同配置测到
的 `1.529x` 更高——那次带 pinned buffer 和 torch staging 拷贝，端到端里掺了 CPU
开销。纯文件 I/O 口径下 O_DIRECT 的收益更干净。

范围限制：两节点读同一批文件，OSS 侧缓存可能为第二个 reader 提供帮助，本实验不能
区分"两份独立带宽"和"一份被共享但尚未饱和的带宽"；`DONTNEED` 也只清客户端缓存。
要证伪需要让两节点读互不相交的文件集重跑。

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
