# Benchmark、workload 与观测规范

> 返回 [文档索引](README.md)

## Phase 5：论文瓶颈趋势

完成 Phase 4 后执行以下基线矩阵：

| 组别 | 架构 | 外部 KV | 用途 |
|---|---|---|---|
| U0 | Unified TP=8 | 关闭 | 模型计算基线 |
| U1 | Unified TP=8 | Lustre L3 | 外存开销 |
| P0 | 1P1D | 关闭 | PD 本身开销 |
| P1 | 1P2D | 关闭 | Decode 扩展效果 |
| B | 1P2D | 仅 PE 从 Lustre 读 | 论文 Basic 近似基线 |
| O | 1P2D | 跳过 L3 读与 PD transfer | Oracle 近似上界 |

主 workload：

- Max Agent Length：32K、48K、64K；
- 每档 500 条基础 trajectory；
- 小规模 smoke：8、32 agents；
- 论文 Figure 7 正式离线 batch size：512、1024、2048、4096 agents；
- Append scale：`1×, 1.5×, 2×, 3×`；
- Generation scale：`1×, 1.5×, 2×, 3×`；
- 每个正式配置至少重复 3 次，报告中位数及 min/max 或置信区间；
- 每组测试前明确采用 warm-cache 还是 cold-cache，二者不可混报。

已根据论文 Table 2 生成三档各 500 条 length-only fake trajectory：

```text
/mnt/lustre/users/sezhou/dual-path/test/dataset/dualpath_fake_32k.jsonl
/mnt/lustre/users/sezhou/dual-path/test/dataset/dualpath_fake_48k.jsonl
/mnt/lustre/users/sezhou/dual-path/test/dataset/dualpath_fake_64k.jsonl
```

生成脚本为 `scripts/benchmark/generate_fake_agent_traces.py`，完整统计解释、合成算法、
JSONL schema 和重放要求见 `docs/fake_dataset.md`。论文公开的五个均值不能由
固定 turn 数和固定 per-turn 长度同时满足，因此该数据使用 turn 数与每轮长度
负相关的两个轨迹群体，并明确记录合成统计口径，不能宣称恢复了原始私有 trace。

离线 replay：所有 agent 同时开始，记录最后一个 trajectory 完成时的 JCT。

当 agent 数超过 500 时，严格采用论文附录 A.3 的扩增语义：采样已有 trajectory，
在前面增加随机 `A1` 和 `g1=1` 的 synthetic round。当前 client 使用固定 seed
均匀有放回采样；由于论文未公开 `A1` 长度，本项目将其作为显式参数记录。详见
[Fake 数据集](fake_dataset.md#论文的-5004096-agents-扩增方法)。

当前范围不包含 Poisson arrival 或 APS sweep；本阶段统一采用 barrier 同步启动的
离线 replay，避免把 offered-load 与固定 agent 数混为同一横轴。

### 1P2D+L3 latency matrix

当前先执行一个成本受控的离线矩阵：

```text
架构：1P2D + Lustre L3
数据档：32K / 48K / 64K
Agents：4 / 8 / 16
Turns：每条 trajectory 的前 4 轮
Repetitions：3
总 measured runs：27
```

该 `4/8/16 agents × 前4轮` 矩阵仅为工程 smoke/趋势测试。正式 Figure 7
实验运行完整 trajectory，使用 `512/1024/2048/4096` agents，设置
`MAX_TURNS=0`、`PAPER_EXPAND_AGENTS=1` 和 `TRAJECTORY_OFFSET=0`。

正式入口为论文对齐的动态 rollout：

- 每个配置从独立、空的 Mooncake Store namespace 开始；
- 不预先 populate，不重启进入 measured 阶段；
- **每轮结束后 Prefill `/flush_cache`**，释放本地 L1/L2（近似论文 staging/GPU
  buffer 用完即放）；**不**清 Mooncake Store；
- 多 agent 时先阻止下一波请求进入并排空当前活跃请求，再只执行一次全局
  `/flush_cache`；避免并发请求期间 flush 返回 HTTP 400；
- `MC_TRANSFER_TIMEOUT=120`（Mooncake 默认 30 秒）：长上下文高并发 PD 同步
  batch 若超过默认阈值，Mooncake 会将 RDMA session 永久标记为 failed；
  实际值写入 `run.json.mooncake_transfer_timeout_seconds`；
- `SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES=1024`：将过大的 all-layer
  同步 PD transfer 按 KV page index 有序分批，避免单次调用耗尽 Mooncake
  瞬态资源；实际值写入 `run.json.mooncake_max_transfer_batch_indices`；
- 默认 `STORAGE_SETTLE_SECONDS=1`，避免 Store 尚未可见就发下一轮；
- Decode 在 rollout 中持续把 KV 发布到 Mooncake Store，下一轮从 L3 加载；
- JCT：`trajectory_jct` / turn_wall **不含** flush/settle；batch `wall_seconds`
  含真实墙钟。

论文 §4.1 明确 Decode 在 H2D 后释放 CPU buffer，跨轮依赖存储而非本地 DRAM
cache。旧“不 flush、吃 L1/L2 顺风车”的路径（Job `65694`）与论文不符，仅作
对照。更早的 Populate/Measured + 长 storage wait（`65032`）是诊断手段，也不是
当前正式口径。

Mooncake 的 `PutStart/PutEnd` 完整对象发布替代旧 SGLang file backend 的
`.bin` 并发覆盖，因此不再使用 Decode scratch 目录规避 `Short read`。每个
run 使用独立 DFS root 是实验间 cold-start 隔离，不是请求 critical path
中的等待。Job 退出后仍删除该 Job 的临时 KV 目录。

Job `65032` 是历史诊断矩阵：它使用 Populate/Measured、服务重启、cache flush
和存储静默等待，只能用于早期正确性与趋势分析，不能作为论文性能结果，也不再
是正式脚本的执行语义。

必须注意：`32K/48K/64K` 是源数据集的 MaxLen 分类。按当前约定直接取原始前
4 轮，因此实际 prompt context 远小于对应 MaxLen。图中必须标注实测
`prompt_tokens_min/max`，不能将结果解释为请求已经达到 32K/48K/64K。

JCT、TTFT、TTST、TPOT、ITL、cache hit、request/L3/P→D timing、两套 TTFT
breakdown、字段和有效性阈值统一见 [指标规范](metrics.md)。本文只定义 workload
和实验协议。

实现入口：

- `scripts/benchmark/replay_fake_agent_traces.py`
- `scripts/infra/patch_sglang_benchmark_timing.py`
- `scripts/infra/build_benchmark_timing_image.sbatch`
- `scripts/pd_1p1d_benchmark.sbatch` / `pd_1p2d_benchmark.sbatch` /
  `pd_2p2d_benchmark.sbatch`
- `scripts/benchmark/collect_benchmark_metrics.py`
- `scripts/benchmark/summarize_latency_benchmark.py` /
  `scripts/benchmark/summarize_ttft_fullchain.py`
- `scripts/plot/`（longest/all-trajectories 与 task 级 JCT 作图）

三个正式 sbatch 都接受任意长度的 dataset、agent 和 repetition 配置。标准
dataset 可写 `32`/`32K`；自定义文件写
`32K=/absolute/path/to/file.jsonl`。`REPETITIONS=N` 表示运行 `1..N`，
`REP_LIST="1 3 5"` 可显式选择 repetition ID：

```bash
TASK_NAME=figure7-main \
DATASETS="32K 64K 128K" \
AGENT_COUNTS="200 400 800" \
REP_LIST="1 2 3" \
sbatch scripts/pd_1p2d_benchmark.sbatch
```

旧 `MAX_LENGTHS` 仍作为 `DATASETS` 未设置时的兼容入口。

历史 Job `65032` 为旧 file-backend / 隔离口径矩阵，不得与正式 Mooncake
动态 rollout 混报。指标与产物口径见 [指标规范](metrics.md)；仓库侧图表在
`results/latency-benchmark/<TASK_NAME>/<DATASET>/agents-<N>/rep-<R>/`
（`REPO_RESULT_DIR`，勿与 Lustre `RESULT_DIR` 混淆）。

数据路径、实际 chunk 传输边界和 `ABLATION_PROFILE` 一键配置见
[当前数据流与消融开关](dataflow-and-ablation.md)。

## Benchmark 与观测

### 服务 smoke benchmark

先使用 SGLang 自带客户端验证端到端服务：

```bash
python -m sglang.bench_serving \
  --backend sglang-oai \
  --base-url "http://$PREFILL_HOST:8000" \
  --model "$MODEL_DIR" \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 128 \
  --random-range-ratio 1 \
  --num-prompts 80 \
  --max-concurrency 16 \
  --request-rate inf \
  --output-file "$RESULT_DIR/smoke.jsonl" \
  --output-details
```

稳态 serving benchmark 的 `num-prompts` 至少应为 `5 × max-concurrency`。该工具适合 smoke、TTFT/TPOT 与固定长度请求，不能替代论文的多轮 trajectory replay 客户端。

多轮离线 replay 使用 `scripts/benchmark/replay_fake_agent_traces.py`，其数据格式、调用方法
和当前非流式指标限制见 [Fake 数据集文档](fake_dataset.md#6-生成和重放)。

### 观测范围

当前已自动采集的客户端、cache 和服务端 timing，以及尚未接入的 IB/Lustre/GPU
硬件计数器，统一列在 [指标规范](metrics.md#7-当前自动采集流水线) 和
[有效性检查](metrics.md#9-有效性检查)。环境与复现元数据见
[复现与安全](reproducibility-and-security.md)。

### 实验有效性检查

每次正式运行前后检查：

1. 所有节点 GPU/NIC 健康且无其他作业干扰；
2. 模型和镜像完全一致；
3. 采样参数固定，强制恰好生成 `gen_len`，忽略 EOS；
4. trace 顺序和随机种子固定；
5. cache 初始状态符合 cold/warm 定义；
6. 服务错误率为 0，或明确报告失败；
7. Lustre 未出现明显外部拥塞；若出现则废弃该次结果；
8. 每次只改变一个主要变量。
