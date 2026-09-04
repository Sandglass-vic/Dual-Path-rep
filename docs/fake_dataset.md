# DualPath fake agent 数据集合成方案

> 返回 [文档索引](README.md)

## 1. 目标

根据论文 Table 2 为 32K、48K、64K 各生成 500 条确定性、可重放的
length-only trajectory。数据不包含生产 prompt、代码仓库内容或预生成模型
输出，只描述每轮 append、generation 和累计 prompt context 的 token 数。

生成目录仅在计算节点可见：

```text
/mnt/lustre/users/sezhou/dual-path/test/dataset/
├── dualpath_fake_32k.jsonl
├── dualpath_fake_48k.jsonl
├── dualpath_fake_64k.jsonl
├── manifest.json
└── README.md
```

## 2. 统计量解释

论文只公开五个均值，没有公开原始分布。它们不能通过“每条 trajectory 都使用
固定 Turns、Append、Gen”同时满足。例如 32K 档：

```text
60 × (608 + 148) = 45,360 != 28,639
```

因此合成数据采用以下明确解释：

- `Turns`：每条 trajectory 的 turn 数在数据集上的均值；
- `Append/Gen`：先计算每条 trajectory 的 per-turn 均值，再跨 trajectory 平均；
- `Total`：每条 trajectory 所有 append 和 generation token 数之和的均值；
- `Context`：先计算每条 trajectory 各轮 prompt context 的均值，再跨 trajectory 平均；
- `MaxLen`：每条 trajectory 的 `Total` 均不得超过该档上限。

这是一种满足公开汇总统计的合成解释，不声称恢复了论文私有 trace 的真实分布。

## 3. 合成方法

每档使用等量的两个群体，并令 turn 数与每轮长度负相关：

```text
32K：250 条 × 26 turns，250 条 × 94 turns
48K：250 条 × 45 turns，250 条 × 167 turns
64K：250 条 × 63 turns，250 条 × 251 turns
```

生成器求解两个群体的平均每轮 token 数，使 `Turns`、`Append`、`Gen` 和
`Total` 同时匹配目标；再按论文的 Append:Gen 比例拆分。每条 trajectory 内部
使用指数衰减权重将 token 向前部轮次适度集中，并加入固定种子的轻微
log-normal 扰动。生成器二分搜索衰减系数，使 `Context` 匹配论文均值。

默认 seed 为 `20260226`。整数化采用 largest-remainder 和 balanced rounding，
因此 token 总数保持一致，且同一命令可逐字节复现相同 JSONL。

## 4. 已生成数据的统计

```text
档位  Trajectories  Turns       Append       Gen          Total    Context       Max observed
32K   500           60.0        607.999972   148.000041   28639.0  17182.999815  32444
48K   500           106.0       474.000015   171.999981   42607.0  25120.000106  48155
64K   500           157.0       428.999990   176.000024   55958.0  32720.999994  64274
```

`manifest.json` 保存目标、实测统计、seed、衰减系数、文件大小和 SHA-256。

## 5. JSONL 格式

每行是一条 trajectory：

```json
{
  "schema_version": 1,
  "dataset": "dualpath-fake-32k",
  "trajectory_id": "32k-0000",
  "max_agent_length": 32768,
  "turn_count": 26,
  "total_tokens": 32443,
  "mean_prompt_context_tokens": 19412.3,
  "turns": [
    {
      "turn_id": 1,
      "append_tokens": 1012,
      "generate_tokens": 246,
      "prompt_context_tokens": 1012
    }
  ]
}
```

示例数值仅用于说明字段；以生成文件为准。

## 6. 生成和重放

重新生成：

```bash
srun --partition=GAIA --nodes=1 --ntasks=1 --time=00:05:00 \
  python scripts/generate_fake_agent_traces.py \
  --output-dir /mnt/lustre/users/sezhou/dual-path/test/dataset
```

Replay client 应：

1. 合成恰好 `append_tokens` 个输入 token IDs；
2. 将之前轮次的真实模型输出 IDs 拼入下一轮完整 context；
3. 设置 `max_new_tokens=generate_tokens` 并忽略 EOS，保证精确生成长度；
4. 不保存假的生成内容到数据集，实际 replay 时记录每轮真实响应；
5. 在比较不同系统时固定 JSONL、seed、到达顺序和 cold/warm cache 状态。

离线 replay client 已实现为 `scripts/replay_fake_agent_traces.py`。它使用固定
seed 为每轮构造普通词表范围内的伪随机 token IDs，通过 SGLang `/generate`
发送 `input_ids`，强制精确生成目标长度，并把响应中的真实 `output_ids` 拼入
下一轮 context。不同 trajectory 可并发执行，同一 trajectory 内严格串行。

### 论文的 500→4096 agents 扩增方法

论文 Figure 7 的离线 agent batch size 为 `512/1024/2048/4096`，但每个 MaxLen
档只有 500 条生产 trajectory。附录 A.3 明确说明：需要额外 trajectory 时，从
已有 500 条中采样一条，在其前面添加 synthetic round；该轮使用随机 tokens 作为
`A1`，并固定 `g1=1`。因此 4096 agents 不表示存在 4096 条独立生产轨迹。

Replay client 通过以下参数实现该方法：

```bash
--num-trajectories 4096 \
--paper-expand-agents \
--paper-synthetic-append-tokens 1
```

实现会先保留 500 条基础轨迹，再使用固定 seed 均匀有放回采样，生成剩余 3596
条扩增轨迹，并把来源、seed 和 synthetic round 配置写入结果。论文没有公开
synthetic `A1` 的具体 token 数，因此默认 1 token 是本项目的明确假设，不应称为
该未披露参数的精确复现；可通过 `--paper-synthetic-append-tokens` 固定其他值。
不同系统对比必须使用相同 seed 和该参数。

当目标 agents 小于等于 500 时，`--paper-expand-agents` 使用相同固定 seed 从
500 条基础 trajectory 中均匀无放回抽样，不添加 synthetic round；这用于
32/64/128-agent 小规模实验，避免固定取文件前 N 条造成顺序偏差。

当请求数超过数据集现有轨迹数而没有显式启用 `--paper-expand-agents` 时，client
会直接报错，避免把“请求 4096、实际只运行 500”误当作有效结果。

最小调用示例：

```bash
python scripts/replay_fake_agent_traces.py \
  --dataset "$DATASET_DIR/dualpath_fake_32k.jsonl" \
  --base-url "http://$PREFILL_HOST:8000" \
  --output-dir "$RESULT_DIR/fake-32k" \
  --num-trajectories 32 \
  --max-workers 32 \
  --max-context-tokens 32768
```

输出包括：

- `rounds.jsonl`：每轮 append/prompt/生成/cached token 数、延迟和结束原因；
- `trajectories.jsonl`：每条轨迹的完成轮数、最终 context、JCT 和总命中量；
- `summary.json`：成功/失败数、总 wall time 和请求延迟分位数。

2026-09-01 已完成端到端 smoke：

```text
Slurm Job：64893，COMPLETED，ExitCode=0:0，耗时 1m27s
数据：32K 档，从 trajectory 250 开始取 2 条，每条执行前 2 轮
并发 agents：2
完成：2 trajectories / 4 turns / 0 errors
脚本：scripts/pd_1p2d_benchmark.sbatch（原 replay smoke 已并入正式矩阵）
日志：/mnt/lustre/users/sezhou/dual-path/logs/fake-replay/64893/
```

四轮的目标 generation 长度分别为 81、90、98、88，实际输出完全一致。第二轮
prompt 分别包含第一轮真实输出，报告的 cached tokens 分别为 334 和 403，
证明累计 context 和 trajectory 内前缀复用语义正确。

client 默认保持非流式兼容模式；增加 `--stream` 后使用 SGLang SSE 接口和
`stream_interval=1`，记录客户端延迟。每轮使用 `--request-id-prefix` 生成可与
Prefill timing log 关联的稳定 request ID。当前范围只使用 barrier 同步启动的
离线并发 replay；字段、统计口径与产物见 [指标规范](metrics.md)。

### 完整轨迹性能模式

正式 replay 从第一轮连续运行到最后一轮：

```bash
python scripts/replay_fake_agent_traces.py \
  --dataset "$DATASET_DIR/dualpath_fake_32k.jsonl" \
  --base-url "http://$PREFILL_HOST:8000" \
  --output-dir "$RESULT_DIR/full-32k" \
  --trajectory-offset 250 \
  --num-trajectories 1 \
  --max-workers 1 \
  --max-context-tokens 32768 \
  --stream
```

client 正式默认开启逐轮 Prefill `/flush_cache`（保留 Mooncake Store）与短暂
`STORAGE_SETTLE_SECONDS`，对齐论文 staging/GPU 用完即放、跨 turn 走 L3。
`cache_flush_seconds` / `storage_settle_seconds` 写入 rounds，但不计入
`turn_wall_seconds`。多 agent 仍由 barrier 同时开始。

Job `64914/64984/65032/65502` 的长 storage wait / Populate，以及 Job `65694`
未 flush 的 L1/L2 路径，均不是当前正式口径。
