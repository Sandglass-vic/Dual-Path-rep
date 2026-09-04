# Job 65144：TTFT 全链路历史测例

> 返回 [文档索引](README.md)。

Job `65144` 使用 Qwen2.5-32B、1P2D+L3，准备阶段 16 agents，正式统计仅
1 agent × 4 turns，并选择 repetition 2，以确保 Prefill 已有可命中的 L3 数据。
本文只保留该 Job 的配置与结果。sampling、marker、跨节点时钟、互斥分桶和
当前 timing 窗口统一见 [指标规范](metrics.md#4-服务端-timing-marker)。

## 最小测例结果

4 个请求的平均 TTFT 为 `237.26 ms`：

```text
Prefill Bootstrap & Scheduling       9.56 ms    4.0%
L3 KV Read                           44.31 ms   18.7%
Prefill Forward + Logits + Sampling  31.95 ms   13.5%
Prefill-to-Decode KV Transfer        110.81 ms  46.7%
Decode Prebuilt-token Dispatch       5.57 ms    2.3%
SSE Serialization & Egress           1.58 ms    0.7%
Router, Client & Runtime Gaps        33.48 ms   14.1%
Accounting Error                      0.00 ms    0.0%
```

L3 marker 覆盖 2/4 个请求；`44.31 ms` 是包含两个非 L3 请求零值后的全请求均值，
命中请求的条件均值约为 `88.61 ms`。P→D transfer 是该最小测例中最大的显式
TTFT 成分。

所有堆叠项之和严格等于客户端 TTFT；最大绝对 accounting error 为 0，
cross-node overlap excess 为 0。各桶定义见 [Full-chain additive
breakdown](metrics.md#full-chain-additive-breakdown)。

## 产物与口径

仓库侧 Job `65144` 图表目录已清理；上表数值仍有效。当前正式 full-chain
为**全量 per-request** 明细（`fullchain_rounds.jsonl` 每行一个
`trajectory_id`+`turn_id`）加 `fullchain_summary.json` 聚合，以及
每个 run 的 `all-trajectories/ttft-breakdown.svg`；不再保留单轨迹
fullchain 切片。定义见
[指标规范 §6](metrics.md#6-ttft-breakdown)。Lustre 原始日志仍保留，不复制到
仓库 `results/`。
