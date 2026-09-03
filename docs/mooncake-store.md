# Mooncake Store 与 Lustre POSIX DFS

> 返回 [文档索引](README.md)。

## 版本选择

正式镜像固定 `mooncake-transfer-engine-cuda13==0.3.13.post1`。

- CUDA 13 wheel 与 NVIDIA SGLang 26.01 的 CUDA 13.1/Python 3.12 匹配；
- `0.3.13` 首次包含 descriptor-based DFS replica、POSIX adapter 和
  `ReplicateConfig.dfs_replica_num`；
- `0.3.13.post1` 是同一功能线的发布修订，修复 wheel 发布流程；
- `0.3.9` 只是早期 PD Transfer Engine 验证时可用的首个 CUDA 13 wheel，
  没有继续固定它的兼容性理由。

当前 SGLang `0.5.7+nv26.1` 的 Mooncake backend 不会把
`dfs_replica_num` 传入 `batch_put_from`。上游 SGLang PR `#37390` 截至
2026-09-03 仍未合入，因此派生镜像包含一个针对当前旧 backend 的最小回移：

```text
replica_num=1
dfs_replica_num=1
enable_ssd_offload=true
MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=distributed_storage_backend
```

只升级 wheel 而不回移该配置，数据只会进入 Mooncake memory replica，不会落到
Lustre DFS。

## 数据与一致性语义

当前配置中每个 HiCache 对象有：

1. 一个 Mooncake memory replica；
2. 一个 descriptor-based DFS replica，位于共享 Lustre shard。

正式 SGL(MC) 对照默认设置 `MOONCAKE_GLOBAL_SEGMENT_SIZE=1536gb`/node，对齐论文
§A.4 的每节点 1.5 TiB Mooncake DRAM 配置；最小 smoke 为缩短启动时间显式覆盖为
32 GiB。SGLang HiCache 自身还会分配 host pool，正式大规模运行前必须检查总
DRAM 余量并记录实际 allocation，不能把 smoke 的 32 GiB 结果当性能配置。

Mooncake master 分配 `(shard, offset, object_size)`，客户端通过 POSIX
`WriteAt/ReadAt` 访问相同的 Lustre shard。对象只在成功 `PutEnd` 后对查询可见，
因此不会像旧 SGLang file backend 那样让读者打开仍在覆盖写入的 `.bin` 文件。
这可以消除该发布模型导致的 `Short read`，但成功写入不承诺额外 `fsync`。

Store key 是 token-prefix 派生的内容键，而不是 agent ID。不同 agent 不需要各自
创建目录：不同内容自然得到不同 key，相同内容可安全去重。实验间则使用独立 DFS
root 和新 master，避免跨 run 污染。

## 正式性能语义

`scripts/pd_1p1d_benchmark.sbatch` / `pd_1p2d_benchmark.sbatch` /
`pd_2p2d_benchmark.sbatch`：

- 单次 rollout 内 Decode 持续 Put，后续 turn 自然 Get；
- 不预 Populate、不重启进入 measured；
- 不逐轮 flush、sleep 或等待 storage quiescence；
- 每个 run 从独立空 Store/DFS namespace 开始；
- run 结束后删除本 Job 的 DFS shard 与 client-init 目录。

`scripts/replay_fake_agent_traces.py` 已移除
`--flush-local-cache-between-turns`、`--inter-turn-delay`、
`--storage-dir` 和 `--storage-sync-timeout` 等正确性测试参数。JCT 只分为模型
请求与不可避免的 client runtime。空流响应也不再随机 backoff/retry；正式
run 遇到服务错误直接失败并报告。

## 已完成 Store 验证

Job `65548` 使用新镜像在两个 GAIA 节点完成：

```text
Mooncake：0.3.13.post1
adapter：distributed_storage_backend + posix
介质：Lustre
写节点：dgx-gaia-57
读节点：dgx-gaia-58
对象：16 个，约 4 MiB/对象
DFS shards：8
实际分配字节：67,600,384
```

writer 为每个对象请求 `replica_num=1, dfs_replica_num=1`，写完后退出以移除其
memory segment；另一节点随后按 key 从 Store 读取，逐对象验证长度和 SHA-256。
结果为 `MOONCAKE_DFS_CROSS_NODE_READ_OK`，没有 short read 或内容损坏。测试
目录已由退出 trap 清理。

验证入口：

- `scripts/mooncake_store_dfs_smoke.py`
- `scripts/mooncake_store_lustre_smoke.sbatch`
- `scripts/pd_1p1d_benchmark.sbatch`（正式 1P1D；含 L3/Mooncake）

## 已完成 SGLang HiCache 集成验证

Job `65557` 使用 Qwen2.5-32B、1P1D、Prefill/Decode 各 TP=8，连续执行一条
trajectory 的前 4 turns，全程没有 Populate、flush、sleep、storage wait 或
请求重试。下列 metric 的统一定义见 [指标规范](metrics.md)：

```text
4/4 turns，0 errors
JCT：3.127 s
mean TTFT：0.253 s
mean TTST：0.257 s
mean TPOT：0.00646 s
有 cache hit：3/4 turns
Lustre DFS 实际分配：269,320,192 bytes
```

Prefill 与 Decode 的 16 个 TP rank 均记录
`replica_num=1, dfs_replica_num=1` 和 `Mooncake store setup successfully`；
日志没有 `Short read`、`KVTransferError` 或崩溃。Job 输出
`SGLANG_HICACHE_MOONCAKE_DFS_SMOKE_OK`，随后自动清理 DFS 数据。

首次集成 Job `65554` 暴露了 NVIDIA 26.01 旧 backend 将 JSON extra config
保留为字符串的问题（`'str' object has no attribute 'get'`）。最终镜像在
backport 中先 `json.loads` 再读取配置；Job `65557` 已验证该兼容修复。

## 当前限制

Mooncake 官方仍将 descriptor-based DFS 标为开发/评估功能：

- master 重启或 HA failover 不恢复 DFS allocator；
- 只支持 default tenant；
- 每个对象最多一个 DFS replica，且必须同时有 memory replica；
- 对象不能跨 shard striping；
- 暂无 DFS 专用 timeout/cancellation/`fsync` 保证；
- 不能与 snapshot/oplog/standby restore 组合。

因此当前目标是论文复现与性能评估，不将其描述为生产级持久化存储。
