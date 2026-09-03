# 当前数据流与消融开关

> 返回 [文档索引](README.md)。

## 1P2D+L3 请求数据流

当前 SGLang 0.5.7+nv26.1 的逻辑可准确描述为：

1. Router 将请求送到 Prefill engine。
2. Prefill 的 HiRadix/HiCache 查找可复用前缀。L3 命中时，KV 经
   `Lustre -> Prefill host DRAM -> Prefill HBM` 恢复。
3. Prefill 只计算未命中的 suffix。Decode 所需的完整前缀由恢复的 KV 和新计算
   KV 共同组成。
4. Prefill 通过 Mooncake 将 Decode 所需 KV 发送到选定 Decode engine。
5. Decode 接收 KV 后执行 decode，逐 token 返回输出。
6. 启用 `DECODE_OFFLOAD_KVCACHE=1` 时，Decode 将新增/更新后的 KV 发布到
   Mooncake Store；Store 同时创建一个 memory replica 和一个 Lustre POSIX
   DFS replica。

正式性能路径不等待额外 `fsync` 或文件静默。Mooncake `PutEnd` 只在完整对象
写入成功后发布该对象，读者不会看到部分对象；这解决的是旧 file backend 的
可见性/`Short read` 问题，不代表 DFS 已执行 `fsync`。

当前 latency benchmark 是同一次 rollout 内的动态反馈环：

```text
turn N:   Prefill 查询 Store -> PD transfer -> Decode -> Store Put
turn N+1: Prefill 使用当时已经发布的最长可复用前缀
```

没有 Populate/Measured、逐轮 cache flush、固定 sleep、storage wait 或 Decode
scratch。每个独立 run 使用新的 Mooncake master 和 DFS root，避免不同实验之间
缓存污染。

## Prefill 与 P→D 传输粒度

当前 Job `65032` 的实际参数为：

```text
chunked_prefill_size=8192
disable_overlap_schedule=False
enable_dynamic_chunking=False
page_size=64
```

因此当前不是 layer-wise streaming，也不是整个 prompt 全算完后才开始传输。
实际边界是：

- 每个最多 8192 tokens 的 prefill chunk 完成全部模型层 forward；
- 将该 chunk 已完成、按 64-token page 对齐的 KV 提交给 Mooncake；
- Mooncake 后台 worker 发送 page，调度器可继续处理后续 chunk/batch；
- chunk 内不会在某一模型 layer 完成时立即发送该 layer 的 KV。

设置 `CHUNKED_PREFILL_SIZE=-1` 后，整个 prompt 的 prefill forward 完成后才触发
发送，但底层仍可能按 page 或 layer 组织多个 transfer block，不能表述成一个
物理网络消息。真正的 layer-wise compute/transfer pipeline 尚未实现。

## 一键消融配置

入口：

```bash
ABLATION_PROFILE=<profile> sbatch scripts/pd_1p2d_benchmark.sbatch
```

profile 定义在 `scripts/ablation_profiles.sh`：

- `full`：Prefill Store 读、Decode Store/DFS 写、8192-token chunk、overlap scheduler；
- `no-prefill-l3`：关闭 Prefill Store 查询，Prefill 计算完整 prompt；
- `no-decode-offload`：保留 Prefill Store backend，关闭 Decode offload；
- `no-l3`：Prefill 不读 Store，Decode 不 offload；
- `bulk-prefill`：在 `full` 基础上设置 `chunked_prefill_size=-1`；
- `no-overlap`：在 `full` 基础上关闭 overlap scheduler。

每个 Job 将解析后的配置写入：

```text
$RESULT_DIR/ablation-profile.env
```

每个 run 的 `run.json` 也记录 profile、Prefill L3、Decode offload、
chunk size 和 overlap 状态；所有 profile 都不再执行预 Populate。

`no-l3` 最小 smoke 已由 Job `65087` 验证：1P2D、4 agents × 2 turns 共
8/8 请求完成，request timing 覆盖率 100%，L3 timing 为预期的 0%，无流式
token 合并、重试或请求错误；Job 专属 KV 目录已由 trap 清理。

注意：

- `bulk-prefill` 在真正 32K/48K/64K prompt 上会显著增加 activation 峰值，
  可能 OOM；先从短 prompt smoke 开始。
- 已观察到关闭 Decode offload 后，16-agent 同步突发可能出现 Mooncake
  `KVTransferError`；这是需要报告的消融结果，不能用重试掩盖。
- `no-overlap` 只关闭 SGLang overlap scheduler；Mooncake 后台 worker 仍是异步
  传输，不等价于强制计算与网络完全串行。
- `PD_TRANSFER_BACKEND` 已从硬编码改为环境变量，但目前 profile 都固定使用
  `mooncake`；`fake` backend 不应当作正确推理结果。
