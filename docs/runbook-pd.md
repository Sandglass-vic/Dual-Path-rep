# PD 分离运行手册与已验证结果

> 返回 [文档索引](README.md)

## Phase 2：最小 1P1D

目的：先以两个节点验证 SGLang PD 数据路径和 Mooncake/NIXL transfer engine，再扩到 1P2D。

此阶段必须使用已通过 smoke 的
[Mooncake 派生镜像](containers.md#mooncake-派生镜像)。原始
`26.01-py3.sqsh` 虽然暴露 Mooncake 参数，但未安装 `mooncake` 与
`sglang_router`，不能直接完成 PD。

2026-08-31 已先完成同节点功能验证：

```text
Prefill：GPU 0-3，TP=4
Decode：GPU 4-7，TP=4
Transfer：Mooncake 0.3.13.post1，mlx5_0，RDMA
Router：sgl-model-gateway 0.3.1
模型：Qwen/Qwen2.5-32B
context length：4096
结果：PD_RESPONSE_OK / PD_SMOKE_COMPLETE
生成：def sum_two_integers(a, b): return a + b
```

日志确认 Prefill 和 Decode 都安装了 RDMA transport、发现 1 个 HCA，并使用 `mlx5_0`。可重复脚本位于：

```text
/labhome/sezhou/dual-path/scripts/pd_smoke.sh
```

运行方式：

```bash
srun --nodes=1 --ntasks=1 --gres=gpu:8 \
  --container-image="$IMAGE_DIR/sglang:26.01-py3-mooncake0.3.13.post1-timing.sqsh" \
  --container-mounts="/mnt/lustre:/mnt/lustre,$CODE_HOME:/workspace/dual-path" \
  --container-workdir=/workspace/dual-path \
  --no-container-mount-home \
  bash scripts/pd_smoke.sh
```

该结果证明新镜像支持 SGLang + Mooncake + PD 的完整软件路径。日志中每个 TP worker 都出现一次 `torchvision.io.image` 扩展缺少 JPEG/PNG 动态库的 warning。当前是纯文本 Qwen2.5 推理，不使用 torchvision image I/O，因此不影响本次 PD 验证；不要为了消除该 warning 改动当前已验证镜像。

2026-08-31 进一步完成跨节点 1P1D：

```text
Slurm Job：64778，COMPLETED，ExitCode=0:0，耗时 2m41s
Prefill：dgx-gaia-09，8 × H100，TP=8
Decode：dgx-gaia-10，8 × H100，TP=8
模型：Qwen/Qwen2.5-32B
Mooncake：0.3.9
PD device：两侧均为 mlx5_0
结果：MULTINODE_PD_RESPONSE_OK
RDMA 日志：MULTINODE_RDMA_LOGS_OK
最终状态：MULTINODE_PD_SMOKE_COMPLETE
```

Prefill 的 Mooncake endpoint 为 `10.230.196.72`、LID 165；Decode endpoint 为 `10.230.196.73`、LID 232。两侧各 8 个 TP worker 均发现 1 个 HCA 并安装 RDMA transport，日志中没有 `KVTransferError`、segmentation fault 或 OOM。

可重复脚本：

```text
scripts/pd_prefill_router.sh
scripts/pd_decode.sh
scripts/pd_1p1d_benchmark.sbatch
```

提交命令：

```bash
mkdir -p "$RUNTIME_HOME/logs/latency-benchmark"
sbatch scripts/pd_1p1d_benchmark.sbatch
```

Slurm 总作业成功后，两个长期运行的 server step 会由脚本的 `trap` 终止，因此 `sacct` 中子 step 显示 `CANCELLED by <uid>` 是预期清理；判定结果应看 batch job 的 `COMPLETED/0:0` 和三个 smoke sentinel。

本次使用 base checkpoint，返回的是有效 completion，但不保证遵循自然语言指令。它验证的是模型执行和跨节点 KV transfer；若要验证 instruction-following 或 SWE-bench 质量，应另行下载 Instruct checkpoint。

固定两侧均为 `TP=8`，避免异构 TP 的 KV layout 和 staging buffer 成为额外变量。选择一张属于 compute fabric 的活动 NDR 设备，例如 `mlx5_0`；最终设备必须通过 NCCL/IB 连通测试确认，不能使用 Lustre 路由对应的 `mlx5_1`。

Prefill：

```bash
python -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 30000 \
  --tp 8 \
  --trust-remote-code \
  --context-length 32768 \
  --mem-fraction-static 0.80 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0 \
  --disaggregation-bootstrap-port 8998 \
  --enable-metrics
```

Decode：

```bash
python -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 30000 \
  --tp 8 \
  --trust-remote-code \
  --context-length 32768 \
  --mem-fraction-static 0.80 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0 \
  --enable-metrics
```

Router：

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://<prefill-host>:30000" 8998 \
  --decode "http://<decode-host>:30000" \
  --host 0.0.0.0 \
  --port 8000
```

验收条件：

- Prefill 和 Decode 均被 Router 判定健康；
- 一个请求只在 Prefill 执行 prompt，在 Decode 流式生成；
- Prefill→Decode KV transfer 成功，日志无回退到 TCP 或 layout error；
- 与统一服务相比，短输入下结果一致，且 32K 输入无传输超时；
- 记录 transfer backend、实际 RDMA device、传输耗时和网络计数器。

## Phase 3：目标 1P2D

资源布局固定为三个 DGX 节点：

```text
node-0: Prefill, TP=8
node-1: Decode-0, TP=8
node-2: Decode-1, TP=8
router: 可与 Prefill 同节点，仅占少量 CPU
```

两个 Decode 服务命令相同，只是 hostname 不同。Router 同时注册两个 Decode URL：

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://$PREFILL_HOST:30000" 8998 \
  --decode "http://$DECODE0_HOST:30000" \
  --decode "http://$DECODE1_HOST:30000" \
  --prefill-policy round_robin \
  --decode-policy round_robin \
  --host 0.0.0.0 \
  --port 8000
```

当前 Router 版本必须用重复的 `--decode` 参数注册多个 Decode，并显式设置
`--decode-policy round_robin`。只使用通用的 `--policy round_robin` 时，两个
Decode 都会注册并通过 health check，但本次实测的 6 个短请求全部落到同一个
Decode，不能作为 1P2D 数据面验证。

2026-08-31 已完成跨节点 1P2D 最小正确性验证：

```text
Slurm Job：64791，COMPLETED，ExitCode=0:0，耗时 1m41s
Prefill：dgx-gaia-34，8 × H100，TP=8
Decode-0：dgx-gaia-35，8 × H100，TP=8
Decode-1：dgx-gaia-36，8 × H100，TP=8
Router：Prefill 节点，端口 8000
模型：Qwen2.5-32B，bfloat16，context length 4096
Transfer backend：Mooncake Transfer Engine
RDMA device：mlx5_0
请求：6 个确定性短 prompt，每个最多生成 24 token
日志：/mnt/lustre/users/sezhou/dual-path/logs/pd-1p2d/64791/
脚本：scripts/pd_1p2d_benchmark.sbatch（原 pd_1p2d_multinode.sbatch 已并入）
```

验收结果：

```text
MULTINODE_1P2D_RESPONSES_OK count=6
MULTINODE_1P2D_BOTH_DECODES_USED
MULTINODE_1P2D_RDMA_LOGS_OK
MULTINODE_1P2D_SMOKE_COMPLETE
```

Prefill、Decode-0 和 Decode-1 分别记录了 8 个 RDMA transport；两个 Decode
日志均出现 `Decode batch`，且无 `KVTransferError`、segmentation fault 或
OOM。该 smoke 证明 1P2D 的服务注册、请求分流和跨节点 PE→DE KV 传输可用，
但尚未证明 30 分钟稳定性、32K/48K/64K 长上下文或 L3 KV-cache 复用。

Host DRAM 在此阶段仅由 Mooncake/SGLang 作为 PE→DE 搬运的临时 staging
buffer 使用；本次没有启用 HiCache，因此不存在可跨请求命中的 L2/L3 KV
cache，也没有把 host memory 当作持久缓存或额外分布式存储层。

2026-09-01 已进一步完成 1P2D + 共享 Lustre L3 验证：

```text
Slurm Job：64980，COMPLETED，ExitCode=0:0，耗时 4m43s
节点：dgx-gaia-51 / 52 / 53
populate：584 个 .bin，1,224,736,768 bytes
重启后 cached tokens：2,304 / 2,368
prefetched_tokens_total 增量：37,376（TP8 rank 求和）
脚本：scripts/pd_1p2d_benchmark.sbatch（原 pd_1p2d_lustre_l3.sbatch 已并入）
```

该测试证明两个独立前缀可由 Decode offload 到共享 namespace，并在三个服务
角色全部重启后由 Prefill 从 L3 恢复；两个 Decode 均参与恢复请求。共享文件
去重和 offload admission 可能使处理请求的 Decode 不产生新文件，因此不要求
两个 Decode 每轮都分别新增 `.bin`。Job 结束后 KV 目录已自动删除。

随后 Job `64984` 使用（现已并入）`scripts/pd_1p2d_benchmark.sbatch` 完成 4 agents ×
4 turns 并发 fake trajectory。populate 生成 816 个 `.bin`、约 1.71 GB；
全服务重启后的 16/16 轮全部命中 L3，cached tokens 为 320–1,600。两个
Decode 在 populate/recover 阶段均处理请求并完成 backup。Job
`COMPLETED (0:0)`，耗时 5m11s，KV 目录退出时自动删除。

### 扩展验证：跨节点 2P2D

2P2D 不是论文 Qwen2.5-32B 的默认资源比，本实验仅用于验证 Router 能否同时
管理多个 Prefill 和多个 Decode，以及任意 PE→DE 组合的 Mooncake 传输路径。
资源布局为四个独占 DGX 节点，每个模型实例均使用 8 张 H100、`TP=8`：

```text
node-0: Prefill-0 + Router
node-1: Prefill-1
node-2: Decode-0
node-3: Decode-1
```

Router 需要分别重复 `--prefill` 和 `--decode`；每个 Prefill URL 后必须携带
对应的 bootstrap port。当前各节点使用相同端口号，但监听地址不同：

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://$PREFILL0_HOST:30000" 8998 \
  --prefill "http://$PREFILL1_HOST:30000" 8998 \
  --decode "http://$DECODE0_HOST:30000" \
  --decode "http://$DECODE1_HOST:30000" \
  --prefill-policy round_robin \
  --decode-policy round_robin \
  --host 0.0.0.0 \
  --port 8000
```

2026-09-01 已完成跨节点 2P2D 最小正确性验证：

```text
Slurm Job：64860，COMPLETED，ExitCode=0:0，耗时 3m08s
Prefill-0：dgx-gaia-51，8 × H100，TP=8
Prefill-1：dgx-gaia-52，8 × H100，TP=8
Decode-0：dgx-gaia-53，8 × H100，TP=8
Decode-1：dgx-gaia-54，8 × H100，TP=8
Router：Prefill-0 节点，端口 8000
模型：Qwen2.5-32B，bfloat16，context length 4096
Transfer backend：Mooncake Transfer Engine
RDMA device：mlx5_0
请求：8 个确定性短 prompt，每个最多生成 24 token
日志：/mnt/lustre/users/sezhou/dual-path/logs/pd-2p2d/64860/
脚本：scripts/pd_2p2d_benchmark.sbatch（原 pd_2p2d_multinode.sbatch 已并入）
```

验收结果：

```text
MULTINODE_2P2D_RESPONSES_OK count=8
MULTINODE_2P2D_BOTH_PREFILLS_USED
MULTINODE_2P2D_BOTH_DECODES_USED
MULTINODE_2P2D_RDMA_LOGS_OK
MULTINODE_2P2D_SMOKE_COMPLETE
```

两个 Prefill 日志分别出现 8 和 7 条 `Prefill batch`，两个 Decode 日志均出现
2 条 `Decode batch`；四个模型实例各初始化了 8 个 RDMA transport，且未发现
`KVTransferError`、CUDA error、OOM 或 segmentation fault。因此 2P2D 的注册、
P/D 双侧 round-robin 分流和跨节点 RDMA 数据面已经通过最小验证。日志行数包含
服务 warmup，不能直接解释为精确的 4:4 请求分配；精确负载均衡应使用 Router
指标或带 request ID 的 tracing 验证。

正式成功前的平台故障节点，不应误记为软件配置失败。当前统一排除列表为
`dgx-gaia-30,dgx-gaia-46,dgx-gaia-53`，完整表见
[已知故障节点](platform.md#已知故障节点临时排除)：

- Job `64855`：`dgx-gaia-30`，CUDA `busy or unavailable`；
- Job `64858`：`dgx-gaia-46`，缺少 `nvidia-container-cli`；
- Job `64860`：排除 `30/46` 后 2P2D 成功；
- Job `65566`：`dgx-gaia-53` Decode 失联（Router `HostUnreachable`，
  Prefill bootstrap 超时）；
- Job `65592`：仅排除 `53` 后又分到含 `30` 的节点组，已取消。

提交示例：

```bash
sbatch --exclude=dgx-gaia-30,dgx-gaia-46,dgx-gaia-53 --time=08:00:00 \
  scripts/pd_1p2d_benchmark.sbatch
```

与 1P2D 一样，本次未启用 HiCache，Host DRAM 仅作为临时传输 staging，不构成
可命中的 L2 cache、持久缓存或分布式存储层。

Slurm/Enroot 启动模板：

```bash
salloc --partition=GAIA --nodes=3 --gres=gpu:8 --time=04:00:00
mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export PREFILL_HOST=${HOSTS[0]}
export DECODE0_HOST=${HOSTS[1]}
export DECODE1_HOST=${HOSTS[2]}

# 对三个服务分别使用以下 srun 骨架；ROLE_CMD 替换为上面的 Prefill/Decode 命令。
srun --nodes=1 --ntasks=1 --nodelist="<host>" --gres=gpu:8 \
  --container-image="$IMAGE_DIR/sglang:26.01-py3-mooncake0.3.13.post1-timing.sqsh" \
  --container-mounts="/mnt/lustre:/mnt/lustre,$CODE_HOME:/workspace/dual-path" \
  --container-workdir=/workspace/dual-path \
  --no-container-mount-home \
  bash -lc '<ROLE_CMD>'
```

实际 sbatch 脚本应：

- 给三个 `srun` step 分配互不重叠的节点并放入后台；
- 每个 step 单独写日志；
- 等待三个 `/health` 成功后再启动 Router 和 benchmark；
- 使用 `trap` 在任务结束时清理后台进程；
- 不使用 `--container-writable`，除非确认必须修改容器根文件系统；
- 将模型、结果和缓存目录显式挂载，不依赖自动 home mount。

验收条件：

- 1P2D 连续稳定服务 30 分钟；
- 两个 Decode 都收到请求，分配比例没有长期偏斜；
- 32K、48K、64K 请求均可工作；
- 同一模型与并发下完成 unified、1P1D、1P2D 三组可比测试。

## 平台资源布局补充

推荐最初使用三个完整 DGX 节点：

```text
Prefill node: 8 × H100，TP=8
Decode node 0: 8 × H100，TP=8
Decode node 1: 8 × H100，TP=8
Router: 与 Prefill 同节点，绑定少量 CPU
```

这样与论文的 Qwen SGLang 基线 `TP=8` 对齐，也避免在同一节点混放 P/D 导致 HBM、PCIe、NIC 和 CPU 资源互相干扰。

建议绑定：

- SGLang 模型进程使用整节点 GPU；
- Lustre 走 `mlx5_1`（由系统路由决定）；
- PD transfer 显式选择 compute-fabric device，初测可用 `mlx5_0`；
- Router 不占 GPU；
- 指标采集进程限制 CPU，避免影响 SGLang。

若三个并行 `srun` step 无法同时启动，检查 step 的 `--exclusive`、节点约束和 GPU TRES 分配。不要通过重复启动三个独立 allocation 代替同一 1P2D 作业，否则节点生命周期和失败清理更难控制。
