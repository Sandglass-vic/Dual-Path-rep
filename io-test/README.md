# DualPath Decode KV Read Mini Bench

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
```

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
