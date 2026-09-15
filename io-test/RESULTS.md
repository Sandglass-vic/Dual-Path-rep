# Decode KV Read Mini Bench Results

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
