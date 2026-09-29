# PD 分离运行手册与已验证结果

> 返回 [文档索引](README.md)

> **pza100 状态：尚未完成运行入口迁移。** 本页现有 `P + D + 1 client`、TP8、
> GAIA 节点和 Lustre 命令是历史流程。g5/g6 只有两个四卡节点，不能原样执行。
> 当前路径、镜像和限制见 [pza100 环境](pza100.md)；应先实现 TP4 与 client
> 共置，再进行双节点 1P1D smoke。不要用外层 srun 包住 `pd_benchmark.sh`，
> 原因见 [pza100 启动方式](pza100.md#启动方式不要嵌套-srun)。

当前正式入口为 `scripts/pd_benchmark.sh`。先用 `pd_alloc.sh` 申请 P+D+1 个节点，
再 SSH 到每个节点：P/D 节点分别运行 `prefill`/`decode`，额外节点运行
Router/replay `client`。
不再提供正式 `sbatch` 入口，也不使用文件传递进程启停信号。完整参数和当前退出
语义见 [benchmark.md](benchmark.md)。下文出现的旧 sbatch 名称仅用于标识历史
Job，不能继续提交。

## 当前手工启动流程

以下以 1P2D 为例。`pd_alloc.sh` 只申请节点、镜像和拓扑，不保存 L3 read path：

```bash
scripts/pd_alloc.sh 1 2
```

默认是 Prefill L3 read 和当前默认镜像
`sglang-file-l3-dual-staging-67876.sqsh`。dataset、agent、repetition 和
本次 run 的日志/结果目录属于 client，可在每次 client 启动时设置：

```bash
DATASET=64K AGENTS=8 REPETITION=2 \
  TASK_NAME=1p2d-64k-a8-r2 scripts/pd_benchmark.sh client
```

L3 read path 必须在 Prefill、Decode、client 三个角色启动时分别指定，并且三者相同：

```bash
DUALPATH_L3_READ_PATH=decode scripts/pd_benchmark.sh prefill
DUALPATH_L3_READ_PATH=decode scripts/pd_benchmark.sh decode
DATASET=32K AGENTS=1 REPETITION=1 \
  DUALPATH_L3_READ_PATH=decode scripts/pd_benchmark.sh client
```

不能只修改后续 client 命令来切换路径。缺少参数或三端手工使用不同值时，脚本会
拒绝对应角色启动；脚本不通过 allocation 文件保存该选择。

`pd_alloc.sh` 内部执行等价的 4 节点 `salloc`，分配成功后按 `scontrol show
hostnames` 的顺序输出：前 1 个节点为 `prefill-0`，接下来 2 个为
`decode-0/1`，最后一个为 `client`。allocation shell 必须保持打开。
提交节点不挂载 Lustre，因此 allocation 配置的创建和退出删除由 client 节点上
各一个短 `srun --overlap` step 完成，不会长期占用 engine step。

`pd_alloc.sh` 会为每个角色输出两步交互式命令。例如先登录 Prefill 节点：

```bash
ssh <prefill-host>
cd /labhome/sezhou/dual-path && DUALPATH_L3_READ_PATH=prefill scripts/pd_benchmark.sh prefill
```

Decode 和 client 节点分别执行相同的 `ssh` 步骤，再运行带相同路径参数的
`pd_benchmark.sh decode` 或 `pd_benchmark.sh client`。交互式 SSH 保证 `Ctrl-C`
发送给远端 `srun`。不要写成字面上的 `ssh host && command`，因为后半段会在 SSH
退出后回到提交节点执行。

client 提示命令会显式展示本次默认 workload，例如：

```bash
DATASET=32K AGENTS=1 REPETITION=1 DUALPATH_L3_READ_PATH=prefill \
  scripts/pd_benchmark.sh client
```

其中 L3 read path 来自 allocation 配置且必须与 server 一致；dataset、agents、
repetition、`TASK_NAME`、`LOG_DIR` 和 `RESULT_DIR` 可在每次 client run 时修改。

Prefill/Decode 的 SGLang stdout/stderr 会实时同时输出到当前终端和
`$RUNTIME_HOME/test/<allocation-id>/pd-servers/`。launcher 使用 Python 无缓冲、
`srun --unbuffered` 和 `tee -a`，终端输出与持久日志内容一致。

launcher 根据本机唯一运行中的 allocation 自动读取
`$RUNTIME_HOME/config/allocations/<job-id>.env`，再用 hostname 判断角色。因此
不需要在 SSH 会话中重复 `export`。`prefill`、`decode`、`client` 参数会校验本机
角色；无参数时仍自动识别，`server` 保留为 Prefill/Decode 的兼容别名。
client 命令每次都会打印最终 workload 和输出路径；client replay 完成后仅退出
本机 Router，不影响 Prefill/Decode server，因此可以在同一 allocation 上重复执行。
进度以 turns 完成数/总数和 agents 完成数/总数显示。长期 server 的真实 L3 KV
目录固定为：

```text
$RUNTIME_HOME/test/<allocation-id>/pd-servers/kv-cache/file/
$RUNTIME_HOME/test/<allocation-id>/pd-servers/kv-cache/l3-sync/
```

`file/` 保存 KV，`l3-sync/` 只保存 TP-rank completion marker。多个 client run
共享该 namespace；`TASK_NAME` 隔离日志、结果和 request ID，但不提供 cold start。
client 启动前会检查端口和残留 replay/Router。检测到残留时会打印类似以下命令，
需要人工确认 PID 后执行：

```bash
pgrep -u "$USER" -af 'replay_fake_agent_traces.py|sglang::router'
pkill -TERM -u "$USER" -f 'replay_fake_agent_traces.py|sglang::router'
pkill -KILL -u "$USER" -f 'replay_fake_agent_traces.py|sglang::router'
```

客户端等待所有 engine 健康，启动 Router，运行 replay，并在 replay 结束时清理
本机 Router。client 运行中按 `Ctrl-C` 会显式清理 replay 进程组和 Router，不应
留下后台 Python。全部 client run 完成后，先在 Decode 会话中按一次 Ctrl-C，确认单节点 `srun` 已
退出，再停止 Prefill。launcher 使用 `--quit-on-interrupt`，无需连续发送多次
Ctrl-C。不要直接关闭 SSH 或先释放 allocation。

client 成功完成后会在终端额外打印 allocation 级 cold-cache 清理命令和独立绘图
命令；client 失败或被 Ctrl-C 中断时只打印 cold-cache 清理命令，不提示绘图。

如果下一次实验要求 cold start，停止 Prefill/Decode 后先确认 server 进程已退出，再
释放 allocation，最后删除 allocation 级 HiCache 目录：

```bash
rm -rf /mnt/lustre/users/sezhou/dual-path/test/<allocation-id>/pd-servers/kv-cache
```

`pd_alloc.sh` 会在分配完成时打印带实际 allocation ID 的清理命令。client 不会自动
删除该目录，因为长期 server 可能仍在使用 HiCache metadata 或完成异步 Decode
backup。若不执行清理并在同一 allocation 上再次启动 client，测试会复用已有 L3 KV，
属于 warm-cache，而不是 cold-start。

原始 replay 结果保存在 Lustre 的 allocation 目录：

```text
$RUNTIME_HOME/test/<allocation-id>/<TASK_NAME>/logs/
```

图表由 `pd_plot.sh` 写入仓库主机的 `results/latency-benchmark/<TASK_NAME>/`，不会
把原始 replay JSON 复制到仓库。

最后独立后处理：

```bash
scripts/pd_plot.sh "$TASK_NAME"
```

`TASK_NAME` 不自动追加 Job ID。client 日志和原始结果按该名称隔离；KV 和同步
marker 属于 allocation 并默认保留。新一轮 client 必须换新名称。allocation `.env`
是静态配置文件，不是进程信号；allocation shell
退出时删除。L3 marker 只用于 turn 间持久化完成同步，不控制进程退出。

## Phase 2：最小 1P1D

目的：先以两个节点验证 SGLang PD 数据路径和 Mooncake/NIXL transfer engine，再扩到 1P2D。

此阶段必须使用已通过 smoke 的
[Mooncake 派生镜像](containers.md#mooncake-派生镜像)。原始
`26.01-py3.sqsh` 虽然暴露 Mooncake 参数，但未安装 `mooncake` 与
`sglang_router`，不能直接完成 PD。

开发中的 patch 必须先使用 Job/commit 专属候选镜像完成本节最小测试，再发布为
正式镜像；不得让未验证候选覆盖 benchmark 默认路径。具体流程见
[Patch 验证与正式发布流程](containers.md#patch-验证与正式发布流程)。

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
  --container-image="$IMAGE_DIR/sglang:26.01-py3-mooncake0.3.13.post1-file-l3-dual-staging-v1.sqsh" \
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
scripts/tests/runtime/pd_decode.sh
scripts/pd_benchmark.sh
scripts/tests/runtime/pd_runtime.sh
```

当前手工方案让每个 engine 成为独立的单节点 `srun` step。客户端退出不会触发
Slurm 批量杀其他节点；服务按上面的 Decode-first 顺序由操作者显式停止。

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
Router：当时与 Prefill 同节点，端口 8000
模型：Qwen2.5-32B，bfloat16，context length 4096
Transfer backend：Mooncake Transfer Engine
RDMA device：mlx5_0
请求：6 个确定性短 prompt，每个最多生成 24 token
日志：/mnt/lustre/users/sezhou/dual-path/logs/pd-1p2d/64791/
旧脚本：pd_1p2d_multinode.sbatch（已移除；当前入口为 scripts/pd_benchmark.sh）
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
旧脚本：pd_1p2d_lustre_l3.sbatch（已移除；当前入口为 scripts/pd_benchmark.sh）
```

该测试证明两个独立前缀可由 Decode offload 到共享 namespace，并在三个服务
角色全部重启后由 Prefill 从 L3 恢复；两个 Decode 均参与恢复请求。共享文件
去重和 offload admission 可能使处理请求的 Decode 不产生新文件，因此不要求
两个 Decode 每轮都分别新增 `.bin`。该历史 Job 结束时曾自动删除 KV 目录；
当前手工流程不会自动删除。

随后 Job `64984` 使用已移除的旧 replay 脚本完成 4 agents ×
4 turns 并发 fake trajectory。populate 生成 816 个 `.bin`、约 1.71 GB；
全服务重启后的 16/16 轮全部命中 L3，cached tokens 为 320–1,600。两个
Decode 在 populate/recover 阶段均处理请求并完成 backup。Job
`COMPLETED (0:0)`，耗时 5m11s；这是旧脚本的自动删除行为。

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
Router：当时与 Prefill-0 同节点，端口 8000
模型：Qwen2.5-32B，bfloat16，context length 4096
Transfer backend：Mooncake Transfer Engine
RDMA device：mlx5_0
请求：8 个确定性短 prompt，每个最多生成 24 token
日志：/mnt/lustre/users/sezhou/dual-path/logs/pd-2p2d/64860/
旧脚本：pd_2p2d_multinode.sbatch（已移除；当前入口为 scripts/pd_benchmark.sh）
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
`dgx-gaia-12,dgx-gaia-18,dgx-gaia-30,dgx-gaia-46,dgx-gaia-53`，完整表见
[已知故障节点](platform.md#已知故障节点临时排除)：

- Job `64855`：`dgx-gaia-30`，CUDA `busy or unavailable`；
- Job `64858`：`dgx-gaia-46`，缺少 `nvidia-container-cli`；
- Job `64860`：排除 `30/46` 后 2P2D 成功；
- Job `65566`：`dgx-gaia-53` Decode 失联（Router `HostUnreachable`，
  Prefill bootstrap 超时）；
- Job `65592`：仅排除 `53` 后又分到含 `30` 的节点组，已取消。

历史 Job 使用过 sbatch；当前应在 `salloc` 时应用排除列表：

```bash
salloc --partition=GAIA --nodes=<P+D+1> --ntasks=<P+D+1> --ntasks-per-node=1 \
  --cpus-per-task=112 --gres=gpu:8 --exclusive --time=08:00:00 \
  --exclude=dgx-gaia-12,dgx-gaia-18,dgx-gaia-30,dgx-gaia-35,dgx-gaia-38,dgx-gaia-46,dgx-gaia-53
```

与 1P2D 一样，本次未启用 HiCache，Host DRAM 仅作为临时传输 staging，不构成
可命中的 L2 cache、持久缓存或分布式存储层。

Slurm/Enroot 启动结构：

```text
salloc allocation
├── Prefill node 0..P-1: pd_benchmark.sh prefill -> one-node TP8 srun
├── Decode node 0..D-1:  pd_benchmark.sh decode -> one-node TP8 srun
└── Client node:         pd_benchmark.sh client -> Router + replay
```

`pd_benchmark.sh` 为每个 SSH 会话创建一个绑定本机的 `srun --jobid`：

- engine step 使用整节点 8 GPU；client step 使用 `--gres=none` 和少量 CPU；
- 每个 engine 启动一个官方 TP8 launcher，并通过无缓冲 `tee` 同步输出到终端和日志；
- client 等待全部 `/health` 成功后再启动 Router 和 replay；
- client trap 清理本次 replay 进程组和本机 Router，不跨节点发送 kill；
- 不使用 `--container-writable`，除非确认必须修改容器根文件系统；
- 显式挂载 Lustre 和仓库，不依赖容器 home mount。

验收条件：

- 1P2D 连续稳定服务 30 分钟；
- 两个 Decode 都收到请求，分配比例没有长期偏斜；
- 32K、48K、64K 请求均可工作；
- 同一模型与并发下完成 unified、1P1D、1P2D 三组可比测试。

## 平台资源布局补充

1P2D 推荐使用三个 engine 节点和一个独立客户端节点：

```text
Prefill node: 8 × H100，TP=8
Decode node 0: 8 × H100，TP=8
Decode node 1: 8 × H100，TP=8
Client node: Router + replay，不运行模型
```

这样与论文的 Qwen SGLang 基线 `TP=8` 对齐，也避免在同一节点混放 P/D 导致 HBM、PCIe、NIC 和 CPU 资源互相干扰。

建议绑定：

- SGLang 模型进程使用整节点 GPU；
- Lustre 使用 `mlx5_1/7` storage rails，实际选择由 LNet 决定；
- 正式 PD transfer 按 GPU/TP rank 一一使用
  `mlx5_0,3,4,5,6,9,10,11` 八条 compute-fabric HCA；
- Prefill RDMA 的 Decode 目标是独立的每 rank 40 GB pinned Host staging
  pool；Mooncake completion 后，Decode 按 request page 执行 Host→HBM，
  归还 staging pages，并在 H2D 完成后才返回 transfer success；后续 Decode
  D2H→Lustre 使用另一套 HiCache Host pool；
- Prefill 仅从 L3 读取，PD completion 后立即释放 L1/L2；Decode 根据
  `cached_tokens` 边界持久化 Prefill 新算 KV 与 Decode KV；
- Router/replay 不占 GPU；同构 `salloc` 最简单，但会在客户端节点保留未使用 GPU；
- 指标采集进程限制 CPU，避免影响 SGLang。

若各 SSH 会话的 `srun` 无法启动，检查 `ALLOC_JOB_ID`、本机是否属于 allocation、
step 的 `--exclusive` 和 GPU TRES。不要用多个独立 allocation 拼接同一次 1P2D
实验，否则节点生命周期和失败清理更难控制。
