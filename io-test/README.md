# DualPath Decode KV Read Mini Bench

## pza100 两节点测试入口

`run_pza100_cluster.sh` 用于当前 g5/g6 A100 集群。它测量的是两个不同视角：

- g5 直接访问本地 XFS `/data`；
- g6 通过 `ibs3` 上的 NFS 访问 g5 的同一 `/data`。

因此报告会分别保留 g5、g6 和双节点结果，不能把它们合并称为两个独立磁盘的
带宽。默认配置使用 TP4、4 MiB/rank page、512 pages/rank，即每节点每阶段
8 GiB，与 Qwen2.5-32B 的 32K KV 数据量保持一致。

所有命令都在 **g5** 的登录 shell 里敲；g6 不需要单独 ssh，是通过 Slurm 的
`--nodelist=g6` 把测试进程投到 g6 上跑的。

### 双节点（自动申请 allocation）

```bash
cd /data/home/yongxiao/dual-path

# 1. smoke，约 2 分钟，确认 pyxis/镜像/GPU 都正常
PZA100_IO_SMOKE=1 io-test/run_pza100_cluster.sh 2>&1 | tee ~/jobs/io-smoke.log

# 2. 只跑四象限主结果（g5 和 g6 各一遍）
RUN_RAW=0 RUN_KV=0 RUN_WRITE=0 io-test/run_pza100_cluster.sh 2>&1 | tee ~/jobs/io-matrix.log

# 3. 全量
io-test/run_pza100_cluster.sh 2>&1 | tee ~/jobs/io-full.log
```

脚本在没有现有 allocation 时自动申请：

```text
partition=build, nodes=g5,g6, exclusive, gpu:1/node, mem=0
```

### 单独测 g5 或单独测 g6

先申请一个单节点 allocation，再在 allocation 里直接调矩阵脚本。`READER_HOST`
决定测哪台：

```bash
# 只测 g5：本地 XFS /data，直连 NVMe
salloc -p build -N1 -w g5 --cpus-per-task=32 --gres=gpu:1 --mem=0 --time=01:00:00
cd /data/home/yongxiao/dual-path
READER_HOST=g5 io-test/run_hot_cache.sh
exit

# 只测 g6：经 ibs3 IPoIB 的 NFSv4.1 client
salloc -p build -N1 -w g6 --cpus-per-task=32 --gres=gpu:1 --mem=0 --time=01:00:00
cd /data/home/yongxiao/dual-path
READER_HOST=g6 io-test/run_hot_cache.sh
exit
```

结束时脚本直接在终端打出四象限表格，JSON 在
`io-test/results/<jobid>-cache-matrix/summary.json`。

g6 的 cold 只清 g6 自己的 page cache，文件仍在 g5 的 NFS server cache 里，所以
测到的是存储网络路径。要连 g5 的缓存一起清、测到 NVMe，需要一个包含两台的
allocation：

```bash
salloc -p build -N2 -w g5,g6 --ntasks=2 --ntasks-per-node=1 \
  --cpus-per-task=32 --gres=gpu:1 --mem=0 --time=02:00:00
cd /data/home/yongxiao/dual-path
READER_HOST=g6 PEER_HOST=g5 DROP_PEER_CACHE=1 io-test/run_hot_cache.sh
exit
```

若 pinned 分配失败（没拿到 GPU），退到 pageable：

```bash
READER_HOST=g6 BUFFER_TYPE=pageable COPY_IMPLEMENTATION=python GRES=none \
  io-test/run_hot_cache.sh
```

常用裁剪：`PAGES_PER_RANK=64`（每象限 1 GiB 而不是 8 GiB）、`REPETITIONS=5`、
`KEEP_DATA=1`（保留 page 文件复用）。

### 并发度：四象限用多少进程

四象限刻意复刻 SGLang Decode 的 file backend 访问形态，不是为了压出硬件峰值：

- 每节点 `TP_RANKS`（默认 4，对应 TP4 / 每节点 4 张 A100）个**独立进程**；
- 每个 rank 串行 `open -> readinto -> close`，一次一个 page，**每 rank 队列深度 1**；
- 每个 rank `torch.set_num_threads(1)`，与镜像里的 `model_runner.py` 一致，这一个
  线程只用于 scatter 拷贝，不影响 I/O 并发。

所以一个节点同时最多 4 个未完成的 4 MiB 读。2026-09-17 在 g5 本地 XFS 上用
`posix_bandwidth_bench.py` 做 O_DIRECT 读并发扫描，膝点正好落在 4：

| jobs | GiB/s | 每块 p50 |
|---:|---:|---:|
| 1 | 4.479 | 0.859 ms |
| 2 | 5.247 | 1.507 ms |
| 4 | 6.462 | 2.422 ms |
| 8 | 6.449 | 4.835 ms |
| 16 | 6.465 | 9.672 ms |
| 32 | 6.467 | 19.233 ms |

4 之后带宽完全平（6.45–6.47 GiB/s，为 PCIe Gen4 x4 上限 7.34 GiB/s 的 88%），延迟
线性增长，纯排队。因此在 g5 上 4 个 rank 已经够，不存在"进程太少压不出带宽"的问题；
反过来降到 1 或 2 会低估 31% / 19%。

g6 不一样：NFS `rsize=1 MiB`，一个 4 MiB page 读会拆成 4 个 RPC，而 `O_DIRECT` 在
NFS 上会关掉 readahead，隐含并发消失。g6 的膝点很可能在 4 以上，也就是说四象限里
g6 的数字应当读作"Decode 这种访问形态实际能拿到多少"，不是"这块网卡能做到多少"。
后者用 `RAW_PHASES` 把 raw 矩阵裁成只读相，再扫并发：

```bash
RUN_MATRIX=0 RUN_KV=0 RUN_WRITE=0 \
  RAW_PHASES="g5-read g6-read" RAW_MODES=direct \
  RAW_JOBS_LIST="1 2 4 8 16 32" REPETITIONS=1 \
  io-test/run_pza100_cluster.sh
```

`TP_RANKS` 本身可以调大，但超过 4 就不再代表 TP4 Decode，只能当硬件能力上界看。

### 双节点同时读：`dual-read` 与 `RAW_DROP_SERVER_CACHE`

`dual-read` 让 g5 和 g6 通过 `cluster_barrier.py` 对齐时钟后同时读，用来回答
"两条路径是两份独立带宽，还是同一份的两个入口"。

```bash
RUN_MATRIX=0 RUN_KV=0 RUN_WRITE=0 \
  RAW_PHASES="g5-read g6-read dual-read" \
  RAW_MODES=direct RAW_JOBS_LIST="4" REPETITIONS=3 \
  io-test/run_pza100_cluster.sh
```

读报告时注意 `dual_aggregate_median_gib_s` 是 `总字节 / 两节点墙钟并集`。g5 比 g6 快
一倍，会先读完然后空转，把这个数拉低。真正反映干扰程度的是 `g5_retention` /
`g6_retention`——它们是各自在 dual 阶段相对单独跑的保持率。

默认情况下 g6 读到的数据仍在 **g5 的 NFS 服务端 page cache** 里：g6 客户端的
`O_DIRECT` 只绕过 g6 自己的缓存。要让 g6 和 dual-read 真正穿到 NVMe，加上：

```bash
RAW_DROP_SERVER_CACHE=1
```

它会在每个 read 相位前到 g5 上对源文件做 `posix_fadvise(DONTNEED)`（`buffered-hot`
除外，热缓存正是那个模式的目的）。两组结果对比即可分离出"g5 内存贡献了多少带宽"。

注意 `DONTNEED` 只回收干净页且无论如何都返回成功，所以 `advise_drop()` 会先
`fsync`。少了这一步，刚写完的源文件会整个留在缓存里，而"冷"相位会静默地在测 DRAM。

### 压存储路径上限：`run_odirect_sweep.sh`

要回答"这条路径最多能做到多少"（而不是"Decode 能拿到多少"），用这个脚本扫进程数
和 block size：

```bash
salloc -p build -N2 -w g5,g6 --ntasks=2 --ntasks-per-node=1 \
  --cpus-per-task=64 --mem=0 --time=02:00:00 \
  bash -c 'cd /data/home/yongxiao/dual-path && HOST=g6 io-test/run_odirect_sweep.sh'
```

allocation 要包含 `PREPARE_HOST`（默认 g5），否则内层 `srun --nodelist=g5` 会报
`Requested node configuration is not available`。只申请单节点也能跑，脚本会检测到
并改为在 `HOST` 上准备源文件，只是 64 GiB 要经 NFS 写回去，慢一些。

默认扫 1–128 进程 × 256K/1M/4M block × direct/buffered-cold，每进程 512 MiB，
所以总 I/O 从 0.5 GiB 涨到 64 GiB。源文件集只准备一次（`MAX_JOBS` 个文件），
每个点读前 N 个，因此单进程工作量固定、总量随并发增长。

这些步骤只用标准库，**不进容器**，省掉每步挂载 13 GB squashfs 的开销。源文件默认
由 `PREPARE_HOST=g5` 本地写入，比从 g6 经 NFS 推 64 GiB 快得多。

结束时打表并写 `results/<jobid>-odirect-sweep-<host>/REPORT.md`，同时给出相对
IB 端口、HCA PCIe 和 g5 NVMe 三条理论上限的占比。

2026-09-17 在 g5 本地 XFS 上的参考结果：`buffered-cold` 与 `direct` 都在 8 进程
处封顶于约 6.3–6.4 GiB/s（NVMe 上限的 85–87%）。注意 O_DIRECT 对 block size 非常
敏感（没有 readahead）：256K 只有 2.99 GiB/s，4M 才到 6.27 GiB/s；buffered 因为有
readahead，256K 就已经满速。

阶段按下面的顺序执行，每组结束后都会刷新 `summary.json` 和 `REPORT.md`，因此
提前中断也能拿到已完成部分的结果：

1. `RUN_MATRIX` — g5 和 g6 各自的 cold/hot × buffered/O_DIRECT 四象限；
2. `RUN_WRITE` — Decode-style buffered 与 batch-`fdatasync` 写入；
3. `RUN_RAW` — fio 或 Python POSIX fallback 的 direct / buffered-cold /
   buffered-hot，覆盖 g5 单独读写、g6 单独读写、双节点同时读写和
   g5-read + g6-write；
4. `RUN_KV` — writer-local、peer-only 和双节点并发的 KV page 亲和性对照。

### 四象限（主结果）

`run_hot_cache.sh` 在同一批 page 文件上测同一个 open/readinto/close 循环和同一个
`page_first -> layer_first` staging 拷贝，只改变缓存状态和 `O_DIRECT`：

| 象限 | 缓存 | 打开方式 |
|---|---|---|
| `cold-buffered` | 先 `POSIX_FADV_DONTNEED` | buffered |
| `hot-buffered` | 不清缓存，沿用上一步填好的 page cache | buffered |
| `hot-direct` | 不清缓存 | `O_DIRECT` |
| `cold-direct` | 先 `POSIX_FADV_DONTNEED` | `O_DIRECT` |

每个象限都把**文件 I/O**（open + readinto + close）和 **scatter**（staging 拷贝）
分别计时，并给出两者的逐 page 延迟分布（p50/p90/p99）。带宽字段相应分成
`bandwidth_gib_s`（端到端）、`file_io_gib_s` 和 `scatter_gib_s`。

`prepare` 对每个 page 文件做 `fdatasync`，冷读阶段前再 `sync(2)` 一次：脏页无法被
`DONTNEED` 驱逐，否则"冷读"会读回刚写入的 page cache。

g6 上的 cold 只清掉 g6 自己的 client cache，文件仍留在 g5 的 NFS server cache，
因此 g6 cold 描述的是**存储网络路径**。要连 g5 的缓存一起清、测到 NVMe，设置
`MATRIX_DROP_PEER_CACHE=1`。

### 跨节点同步

g5 与 g6 没有共同时间源，`CLOCK_REALTIME` 相差数十秒。`cluster_barrier.py` 在放行
前用 NTP 式往返测量 leader-minus-local 偏移（回环实测误差约 1 µs），每条结果都带
`clock_leader_minus_local_ns` 和换算到 leader 时基的 `wall_start_ns_leader` /
`wall_end_ns_leader`。聚合带宽用这些校正后的时间戳计算共同 wall-clock 包络；直接用
本地 epoch 会得到完全错误的 scaling。

默认每组重复 3 次。先跑 smoke；正式运行可用以下变量裁剪或扩展：

```bash
RAW_JOBS_LIST="1 4 8" RAW_BLOCK_SIZES="1M 4M" REPETITIONS=5 \
  io-test/run_pza100_cluster.sh

RUN_RAW=0 RUN_KV=0 RUN_WRITE=0 io-test/run_pza100_cluster.sh
```

### 理论上限

preflight 从 sysfs 读取 IB 端口速率、HCA 的 PCIe 链路、IPoIB MTU/模式，以及本地
文件系统背后 NVMe 的型号和 PCIe 链路，算出各级理论上限写进 `REPORT.md`。注意 IB
链路速率**不是**这里的存储上限：IPoIB 跑 datagram、MTU 2044，NFS 是 `proto=tcp`，
实际可达速率由内核 TCP/IP over IPoIB 决定。

`REPORT.md` 会标记变异系数（CV）超过 10% 的 series；这类结果不作为稳定带宽，
应以相同 `RUN_*`/矩阵设置和 `REPETITIONS=7` 重新运行。

当前 g5 有 fio、g6 没有 fio，因此 `RAW_BACKEND=auto` 会在两节点统一选择
`python-posix`，避免混用不同实现。该 fallback 保留 1/4/8 并发进程、block size、
O_DIRECT、barrier、durable write 和 latency percentile，但不模拟 fio
`libaio iodepth=16`。结果会记录实际 backend；不同 backend 不应直接比较。安装
fio 不是测试前置条件，也不会修改节点软件环境。可显式设置
`RAW_BACKEND=fio|python-posix`，其中强制 fio 时两节点都必须已安装。

结果位于：

```text
/data/home/yongxiao/mydata/dual-test/io-bench/<run-id>/
├── results/
│   ├── summary.json
│   ├── REPORT.md
│   ├── cache-matrix-g5/summary.json
│   ├── cache-matrix-g6/summary.json
│   ├── cache-matrix-g5.log          # 失败时的完整 stdout/stderr
│   ├── telemetry/
│   └── ...
└── data/                    # 成功后默认删除
```

每个阶段采集 `ibs3`、`mlx5_0/mlx5_4`、NFS RPC、mount 和文件系统信息。测试
不会执行全局 `drop_caches`；`posix_fadvise(DONTNEED)` 只能尽量清理 client
page cache，不能清理 g5 的 NFS server cache。Direct/durable 结果用于估计存储
路径带宽，buffered-hot 只代表 page-cache 上限，staging 结果还包含 CPU/DRAM
张量重排开销。`wire_to_io_ratio` 以 g6 client 的 `ibs3` 对应方向字节计算，避免
同时统计 NFS client 与 server 两端而重复计数。

若 NFS 不支持可验证的 O_DIRECT/mincore 语义，对应阶段会记录
`unsupported-or-failed`，不会伪装成通过。失败时保留测试数据用于诊断；成功时
仅在 run marker 匹配的情况下删除 `data/`。

`RUN_ROOT` 必须是 `mydata/dual-test/io-bench` 下一个尚不存在的直接子目录；
脚本拒绝复用已有目录或跟随符号链接。

## GAIA 历史入口

以下内容描述原 GAIA TP8 + Lustre 基准，保留用于历史结果复现。

该测试复现 Decode 纯 Lustre 读取路径的主要 I/O 行为：

- TP=8，每节点启动 8 个独立 reader process；
- 每个 rank 串行执行 `open -> readinto(target) -> close`；
- 默认 page size 为 64 tokens；Qwen2.5-32B TP8 每 rank 的 page 文件为 2 MiB；
- 两节点阶段让两个节点同时读取完全相同的 page 文件；
- target 默认使用 pinned DRAM；
- 读取后使用与 SGLang 相同的 Torch reshape + 张量赋值，将 page 数据拷贝到
  `layer_first` staging；
- 每个 TP worker 默认执行 `torch.set_num_threads(1)`，与镜像中的
  `model_runner.py` 一致；
- 每轮读取前用 `posix_fadvise(DONTNEED)` 尽量清除 reader 节点的本地 page cache。

它隔离测试 file backend 的读取和 page-to-layer-first staging copy，不包含 token
hash、Radix tree、HBM 分配、Host-to-GPU 或 D-to-P reverse transfer。两个节点通过
TCP barrier 同步开始，不使用文件传递同步信号。

## 运行

先申请至少两个 GPU 节点：

```bash
salloc --partition=GAIA --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --cpus-per-task=112 --gres=gpu:8 --exclusive --time=01:00:00
```

在 allocation shell 中运行：

```bash
cd /labhome/sezhou/dual-path
io-test/run.sh
```

默认每节点每轮读取 8 GiB，对应一个完整 32K request：

```text
8 ranks * 512 pages/rank * 2 MiB/page = 8 GiB/node
```

常用覆盖参数：

```bash
PAGES_PER_RANK=4096 REPETITIONS=5 io-test/run.sh
READER_HOSTS="dgx-gaia-21 dgx-gaia-22" io-test/run.sh
BUFFER_TYPE=pageable io-test/run.sh
COPY_TO_STAGING=0 io-test/run.sh
COPY_IMPLEMENTATION=python io-test/run.sh
IO_MODE=direct io-test/run.sh
```

`IO_MODE=direct` 让 single/dual 两相都用 `O_DIRECT` 打开 page 文件，用来回答
"O_DIRECT 下双节点聚合带宽是不是单节点的 2 倍"。只关心文件 I/O 时可以去掉 pinned
DRAM 和 torch staging 拷贝，这样整条链路只用标准库，不进容器也不需要 GPU：

```bash
salloc -p GAIA -N2 --ntasks-per-node=1 --cpus-per-task=32 --mem=0 \
  --exclusive --time=01:00:00

USE_CONTAINER=0 BUFFER_TYPE=pageable COPY_IMPLEMENTATION=python \
  COPY_TO_STAGING=0 GRES=none CPUS_PER_NODE=32 \
  IO_MODE=direct REPETITIONS=3 io-test/run.sh
```

`USE_CONTAINER=0` 时镜像里的 torch 不可用，脚本会拒绝 `BUFFER_TYPE=pinned` 和
`COPY_IMPLEMENTATION=torch`。结果见 [RESULTS.md](RESULTS.md) 的 Job 69346/69347。

## O_DIRECT 验证

申请一个 GPU 节点后运行：

```bash
cd /labhome/sezhou/dual-path
io-test/run_direct.sh
```

该测试使用同一批 2 MiB KV page、8 个 TP reader 和 pinned DRAM，交替执行 cold
buffered read 与 `O_DIRECT` read。两种模式都包含相同的 Torch
`page_first -> layer_first` staging copy。

开始带宽测试前，脚本还会对一个 page 执行 `mincore` 驻留检查：

1. `posix_fadvise(DONTNEED)` 后记录初始 resident page 数；
2. 执行一次 `O_DIRECT` read，再次记录 resident page 数；
3. 再次清理并执行 buffered read，记录最终 resident page 数。

只有 Direct read 不增加 resident page、而 buffered read 增加 resident page 时，
`summary.json` 中的 `direct_bypassed_page_cache` 才会为 `true`。Direct buffer
显式按 4096 bytes 对齐；文件 offset 和默认 2 MiB I/O 长度也满足该对齐要求。

结果默认位于：

```text
~/dual-path/io-test/results/<jobid>/read-results.log
~/dual-path/io-test/results/<jobid>/summary.json
```

测试 page 文件位于 Lustre：

```text
/mnt/lustre/users/sezhou/dual-path/io-test/<jobid>/kv-pages/
```

`run.sh` 不会在 login 节点直接访问 Lustre；数据目录由 compute-node `srun`
创建，并在成功生成 summary 后自动删除。设置 `KEEP_DATA=1` 可保留 page 文件。
`run_direct.sh` 使用相同的清理规则，结果目录后缀为 `<jobid>-direct`。

## 判断标准

`summary.json` 中两个关键值：

- `median_aggregate_scaling`：双节点聚合带宽 / 单节点带宽；理想值接近 2；
- `median_per_node_retention`：双节点时平均每节点带宽 / 单节点带宽；理想值接近 1。

若 aggregate scaling 接近 1、per-node retention 接近 0.5，说明两个 Decode
节点基本在瓜分同一份 Lustre 吞吐，可解释 1P2D 单请求 KV Read 时间高于 1P1D。
若 scaling 接近 2 且 retention 接近 1，则需要继续检查 SGLang 请求并发、文件
分布、运行时外部负载或 Decode 侧 CPU/DRAM 竞争。

注意：`DONTNEED` 只能尽量清理客户端 page cache，不能强制清理 Lustre OSS
缓存。因此脚本交替 single/dual 的执行顺序并重复三轮，避免仅凭一次冷热顺序下结论。
