# Decode KV Read Mini Bench Results

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
