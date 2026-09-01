# DualPath 复现实验方案

## 1. 目标与边界

本项目在 GAIA（DGX H100）集群上使用 **Qwen2.5-32B + SGLang + Slurm/Pyxis/Enroot + Lustre**，分阶段复现论文 *DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference* 的关键现象。

当前最近目标是：

1. 使用已经下载的 SGLang `26.01-py3` Enroot 镜像，稳定运行 Qwen2.5-32B 的单节点统一服务；
2. 用确定性 prompt 验证 OpenAI API、流式生成、输出 token 数和重复推理正确性；
3. 正确性通过后再构建包含 Mooncake 与 Router 的派生镜像，进入 1P1D/1P2D；
4. 为多轮请求接入 L3 KV-cache；实验不依赖 L2 命中，L2 仅允许退化为有限 staging buffer；
5. SWE-bench 暂定为后续 agent workload，当前阶段不搭建 SWE-bench harness。

本项目的最终目标是复现以下趋势，而不是直接承诺复现论文的绝对数值：

- 多轮、长上下文、短 append 的 agentic workload 具有很高的 KV-cache 命中率；
- 外部 KV-cache 读取逐渐成为主要瓶颈，Prefill 节点的存储链路首先饱和；
- PD 分离能减少 Prefill/Decode 计算干扰，但不能单独解决存储带宽不均衡；
- 让 Decode 节点参与 KV-cache 读取并进行路径调度，可以聚合更多节点的存储带宽。

论文使用作者内部推理框架、3FS、layer-wise prefill、定制流量管理与约 5K 行 DualPath 修改。SGLang 原生 PD 分离只负责 Prefill 到 Decode 的 KV 传输，**不等价于 DualPath**。因此实验必须分层：

- **可直接复现**：模型、1P2D 形态、agent trace 统计、PD 分离、外部 KV-cache 引起的 I/O 瓶颈及指标趋势；
- **近似复现**：使用 Lustre 代替 3FS，使用 SGLang HiCache 代替论文内部 KV 存储栈；
- **需要开发**：Storage→Decode→Prefill 的第二条读路径、基于读队列的路径选择、细粒度流量隔离和 layer-wise streaming。

## 2. 论文配置对齐

论文中与本项目直接相关的设置如下：

- 模型：`Qwen/Qwen2.5-32B`，Dense GQA；
- 默认资源比：1P2D，其中 P/D 均以“节点”为单位；
- SGLang 基线：SGLang commit `19089aa`、HiCache、Mooncake Store、Mooncake Transfer Engine、3FS；
- Qwen 32B 的论文 SGLang 基线采用 `TP=8`；
- DualPath 为 Qwen 32B 分配 320 GB 主机 DRAM KV buffer/pool；
- 关闭 speculative decoding；
- 数据集每档 500 条 trajectory；
- 离线指标为整个任务的 JCT，在线指标为 TTFT、TTST、TPOT；
- 在线 SLO：`TTFT ≤ 4 s`、`TPOT ≤ 50 ms`。

论文 trace 的均值：

| Max Agent Length | 平均轮数 | 每轮 Append | 每轮 Gen | Total | 平均 Context |
|---|---:|---:|---:|---:|---:|
| 32K | 60 | 608 | 148 | 28,639 | 17,183 |
| 48K | 106 | 474 | 172 | 42,607 | 25,120 |
| 64K | 157 | 429 | 176 | 55,958 | 32,721 |

论文的每轮 replay 语义为：

```text
Context_(i+1) = A1 + G1 + A2 + G2 + ... + Ai + Gi
Prompt_(i+1)  = Context_(i+1) + A_(i+1)
```

其中 `Ai` 是该轮新增输入，`Gi` 是该轮模型输出。客户端必须保留真实生成结果并在下一轮重新提交完整上下文，不能把 500 条 trajectory 简化为 500 个互不相关的长 prompt。

### 2.1 SWE-bench 的定位

后续初步计划使用 SWE-bench，建议先选 SWE-bench Verified 做功能和 agent trajectory 采集，再根据成本扩展。它适合产生“读代码—运行命令—修改—测试”的多轮 agent 轨迹，但不等同于论文的生产 agentic RL trace：

- SWE-bench 是任务/正确率数据集，论文实验使用的是轨迹长度统计和 replay；
- 复现系统性能时，应先运行 agent 得到每轮 `append_len`、`gen_len` 和累计 context，再冻结 trace 做 replay；
- 不能在比较 Basic 与 DualPath 时让模型重新做不同的 agent 决策，否则 workload 不一致；
- SWE-bench 需要 Instruct 模型、chat template、tool/terminal sandbox 和 agent harness，这些均不属于当前 Qwen2.5-32B 推理 smoke test。

因此模型分两条轨道：

- 系统性能、论文对齐：`Qwen/Qwen2.5-32B` base checkpoint；
- SWE-bench 功能任务：`Qwen/Qwen2.5-32B-Instruct`。

当前只需选择其中一个 checkpoint 跑通推理。若希望人工检查回答质量，优先下载 Instruct；若希望尽早与论文配置一致，优先 base。

## 3. 模型、精度与容量

### 3.1 模型选择

第一选择为论文使用的基础模型：

```text
Qwen/Qwen2.5-32B
```

它具有 64 层、40 个 attention heads、8 个 KV heads、head dimension 128，原生最大上下文为 131,072。若后续需要真实对话模板，可增加 `Qwen/Qwen2.5-32B-Instruct` 对照组，但其默认 `config.json` 最大长度为 32,768；48K/64K 需要额外配置 YaRN，会引入新的实验变量。

因此，系统性能主实验使用 base checkpoint，更接近论文且避免 64K 长度外推配置差异。

### 3.2 KV-cache 容量核算

Qwen2.5-32B 的逻辑 KV-cache 大小（不计页表、对齐和框架元数据）为：

```text
每 token 字节数
= 64 layers × 8 KV heads × 128 head_dim × 2(K,V) × dtype_bytes
= 256 KiB/token（BF16）
= 128 KiB/token（FP8 KV）
```

由此得到：

- 单条 32K context：BF16 约 8 GiB，FP8 约 4 GiB；
- 单条 48K context：BF16 约 12 GiB，FP8 约 6 GiB；
- 单条 64K context：BF16 约 16 GiB，FP8 约 8 GiB；
- 320 GiB 约可容纳 1.31M BF16 KV tokens 或 2.62M FP8 KV tokens。

`320 GB` 在论文中是主机 DRAM 分配，不应误写为单卡 HBM KV 容量。初期先使用 BF16 权重与默认 KV dtype，待基线稳定后再增加 FP8 KV 对照；不能在主对比中同时改变 PD、KV dtype 和存储后端。

### 3.3 论文中的 KV-cache 存储格式

以下格式来自论文附录 A.5 及 §4.1，作为后续 Lustre backend 与 DualPath 数据路径实现的规范基线。

论文定义两种 block：

```text
Layer Block: [1, tokens, bytes_per_layer_per_token]
Full Block:  [num_layers, tokens, bytes_per_layer_per_token]
```

- `tokens` 是 block 内 token 数，即 `block_size`；
- `bytes_per_layer_per_token` 是单层、单 token 的 K/V 数据字节数；
- Qwen2.5-32B 有 `num_layers=64`；
- 对 BF16 KV，`bytes_per_layer_per_token = 8 × 128 × 2(K,V) × 2 bytes = 4096 bytes`；
- 因此 Qwen2.5-32B 的 BF16 Full Block 为 `64 × block_size × 4096 bytes`；
- 论文举例在 Decode 阶段每积累一个完整 block（例如 64 tokens）就持久化，此时 Qwen BF16 Full Block 的有效载荷约为 16 MiB，FP8 KV 约为 8 MiB。

布局用途：

- 与持久化存储交互时只使用 **Full Block**；
- PE/DE host buffer 与 PE HBM 之间按层流式搬运时使用 **Layer Block**；
- 连续拼接同一 token 区间的 64 个 Layer Block，可直接形成一个 Full Block，避免额外 layout conversion；
- KV-cache 在持久化存储中按 trie 组织，每个 trie node 对应一个 Full Block；
- trie key 必须由模型/版本、token block、KV dtype、layout、TP shard 等共同确定，避免不同实验配置误命中；
- 尾部不足一个完整 block 的 token 需要保留在内存中，达到 `block_size` 后再持久化；trajectory 结束时是否 flush 尾块必须固定并记录。

项目的第一版 Lustre 格式建议保持与论文的逻辑形状一致，但增加自描述 header：

```text
KVBlockHeader
  magic/version
  model_revision
  num_layers
  block_size
  valid_tokens
  kv_heads
  head_dim
  dtype
  tp_size/tp_rank
  layout
  checksum

payload
  Full Block [num_layers, valid_tokens, bytes_per_layer_per_token]
```

物理文件组织需要通过实际 I/O 测试决定。不能未经测试就采用“一 block 一小文件”，否则 Lustre metadata 开销可能掩盖数据带宽瓶颈。建议同时评估：

1. 每个 Full Block 一个对象/文件，作为正确性基线；
2. 多个 Full Block append 到 segment file，并维护独立索引；
3. SGLang HiCache backend 的原生序列化格式。

无论采用哪种物理格式，结果中都必须记录 `block_size`、字节序、dtype、校验方式、TP layout、segment 大小和 Lustre stripe。

## 4. 软件与目录约定

项目分为“随时可见的 Git 代码仓库”和“仅计算节点可见的运行数据目录”：

```text
/labhome/sezhou/dual-path/          Git 根目录，登录节点随时可查
├── docs/                           设计、平台和实验记录
├── src/                            后续 Python/C++ 实现
├── scripts/                        Slurm/Enroot/部署/benchmark 脚本
├── configs/                        不含 secret 的配置模板
├── tests/                          正确性与回归测试
└── manifests/                      镜像、模型、trace 和实验 manifest

/labhome/sezhou/lustre/dual-path/   运行数据；软链接到 /mnt/lustre/users/sezhou/dual-path
├── images/
├── models/
├── kv-cache/
├── datasets/
├── traces/
├── results/
└── logs/
```

所有代码和小型配置必须放在 `/labhome/sezhou/dual-path` 的子目录并由 Git 管理，确保登录节点随时可读。镜像、模型权重、SWE-bench checkout/sandbox、KV-cache、大型 trace 和原始结果放在 Lustre，不提交 Git。

推荐环境变量：

```bash
export CODE_HOME=/labhome/sezhou/dual-path
export RUNTIME_HOME=/labhome/sezhou/lustre/dual-path
export IMAGE_DIR=$RUNTIME_HOME/images
export MODEL_DIR=$RUNTIME_HOME/models/Qwen2.5-32B
export CACHE_DIR=$RUNTIME_HOME/kv-cache
export DATASET_DIR=$RUNTIME_HOME/datasets
export TRACE_DIR=$RUNTIME_HOME/traces
export RESULT_DIR=$RUNTIME_HOME/results
export LOG_DIR=$RUNTIME_HOME/logs
```

当前镜像已经导入：

```text
/labhome/sezhou/lustre/dual-path/images/sglang:26.01-py3.sqsh
来源：nvcr.io/nvidia/sglang:26.01-py3
SHA-256：539419624b34a1973df89e8168ee08f9f41b621556922a5e2a4cc299b05e1545
```

2026-08-31 在 GAIA 计算节点上的实测结果：

```text
SGLang: 0.5.7+nv26.1
Mooncake Python module: 未安装
mooncake-transfer-engine: 未安装
mooncake-transfer-engine-cuda13: 未安装
NIXL Python module: 未安装
sglang_router: 未安装
```

镜像内 `launch_server --help` 已确认存在：

```text
--disaggregation-mode {null,prefill,decode}
--disaggregation-transfer-backend {mooncake,nixl,ascend,fake}
--disaggregation-bootstrap-port
--disaggregation-ib-device
--disaggregation-decode-enable-offload-kvcache
--enable-hierarchical-cache
--hicache-storage-backend {file,mooncake,hf3fs,nixl,aibrix,dynamic,eic}
--context-length
--tp-size
--kv-cache-dtype
```

这说明 SGLang 侧已有 Mooncake adaptor 和命令行接口，但底层 Mooncake wheel 与 Router 缺失；选择相关 backend 时仍会在运行期 import 失败。NVIDIA 26.01 release notes 标注的 SGLang 版本与镜像实测值不一致，后续一律以运行时 `0.5.7+nv26.1` 和镜像 SHA-256 为准。

### 4.1 Mooncake 派生镜像

结论与当前状态：

- **当前“跑通 Qwen2.5-32B 统一推理”不需要 Mooncake，也不需要重建镜像。**
- 为后续启用 Mooncake PD transfer 和 HiCache Mooncake Store，已从基础镜像成功派生新 sqsh，没有修改原始镜像。
- Mooncake Transfer Engine 和 Mooncake Store 是两种用途，但现代 Mooncake wheel 同时提供 `mooncake.engine` 与 `mooncake.store`。
- 论文的 SGL(MC) 同时用了 Transfer Engine（PE→DE）和 Store/HiCache（L3）；仅安装 wheel 不会自动得到 3FS，Lustre 也不等同于 3FS。

26.01 镜像使用 CUDA 13.1，官方 Mooncake 当前要求安装 CUDA 13 变体：

```text
mooncake-transfer-engine-cuda13
```

不能直接安装不带后缀的 CUDA 12 wheel。已知 SGLang 0.5.7 曾与 Mooncake 0.3.8 配套部署，但 PyPI 的 CUDA 13 变体从 0.3.9 开始提供。本项目已验证 0.3.9，不直接采用与旧 SGLang 未验证的最新 0.3.13。

已验证组件：

```text
基础镜像：SGLang 0.5.7+nv26.1 / CUDA 13.1
Mooncake：mooncake-transfer-engine-cuda13 0.3.9
Router：sglang-router 0.3.1（sgl-model-gateway 0.3.1）
GPU：NVIDIA H100 80GB HBM3
```

派生镜像：

```text
/labhome/sezhou/lustre/dual-path/images/sglang:26.01-py3-mooncake0.3.9.sqsh
SHA-256：46ea6cd0c763443e47d14f8370316f3ac283620ad22197fa00d732f7a77abf4f
```

实际使用的 Enroot 流程（在计算节点执行）：

```bash
export BASE_IMAGE=$IMAGE_DIR/sglang:26.01-py3.sqsh
export DERIVED_IMAGE=$IMAGE_DIR/sglang:26.01-py3-mooncake0.3.9.sqsh
export ENROOT_DATA_PATH=$RUNTIME_HOME/enroot-data
export ENROOT_CACHE_PATH=$RUNTIME_HOME/enroot-cache

enroot create --name sglang-26.01-mooncake "$BASE_IMAGE"
enroot start --root --rw sglang-26.01-mooncake bash -lc '
  python -m pip install --no-cache-dir \
    "mooncake-transfer-engine-cuda13==0.3.9" \
    "sglang-router==0.3.1"
  python -c "from mooncake.engine import TransferEngine; from mooncake.store import MooncakeDistributedStore"
  python -m sglang_router --version
'
enroot export --output "$DERIVED_IMAGE" sglang-26.01-mooncake
enroot remove sglang-26.01-mooncake
sha256sum "$BASE_IMAGE" "$DERIVED_IMAGE"
```

基础镜像执行 `pip check` 会报告 `triton-kernels` 缺少 `pytest`。这是基础镜像已有的开发/测试依赖缺口，不是 Mooncake 安装造成的运行依赖冲突，因此未为了消除该提示修改推理环境。

派生镜像已经通过：

- `mooncake.store.MooncakeDistributedStore` import；
- `mooncake.engine.TransferEngine` import；
- Store 与 Engine 同进程联合 import；
- `sglang_router` import 和 `sgl-model-gateway --version`；
- H100/CUDA 13.1 可见性；
- SGLang Mooncake PD 参数检查；
- Qwen2.5-32B 实际 Prefill→Decode→Router 请求。

若后续多节点 RDMA 或 HiCache Store smoke 失败：

1. 优先保留 SGLang `0.5.7+nv26.1`，改用 Mooncake 0.3.8 源码按 CUDA 13.1 构建；
2. 若旧 SGLang adaptor 与新 Mooncake API 不兼容，再升级到更新的 NVIDIA SGLang 镜像，并重新跑统一推理基线；
3. 不以论文 commit `19089aa` 作为当前安装目标，它用于说明论文基线，而不是与 2026 年 Mooncake wheel 的兼容承诺。

## 5. 分阶段执行方案

### Phase 0：当前镜像与模型准备

目的：不引入 PD、HiCache 和 Mooncake，只验证镜像与 Qwen2.5-32B。

1. 申请 1 个节点；
2. 验证容器内可见 8 张 H100 和 Lustre；
3. 将一个 Qwen2.5-32B checkpoint 完整下载到 `$MODEL_DIR`；
4. 检查 tokenizer、config 和所有 weight shards；
5. 记录模型 revision、镜像 SHA-256 和 SGLang 版本。

模型约 65 GB，建议直接下载到 Lustre。以下 Python 方式不依赖 `hf` CLI 是否存在；正式运行时将 `<commit-sha>` 替换为固定的 Hugging Face revision：

```bash
srun --partition=GAIA --nodes=1 --ntasks=1 --gres=gpu:1 --time=01:00:00 \
  --container-image="$IMAGE_DIR/sglang:26.01-py3.sqsh" \
  --container-mounts=/mnt/lustre:/mnt/lustre \
  --no-container-mount-home \
  bash -lc '
    python - <<"PY"
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-32B",
    revision="<commit-sha>",
    local_dir="'"$MODEL_DIR"'",
)
PY
  '
```

若先验证未来 SWE-bench 所需的对话模型，将 repo id 和目录改为 `Qwen/Qwen2.5-32B-Instruct`。两种 checkpoint 不得写入同一目录。

当前 base 模型已下载完成：

```text
repo：Qwen/Qwen2.5-32B
revision：1818d35814b8319459f4bd55ed1ac8709630f003
目录：/labhome/sezhou/lustre/dual-path/models/Qwen2.5-32B
占用：约 62 GiB
```

验收条件：

- 容器可稳定启动；
- 8 张 GPU 可见；
- `python -c 'import sglang, torch'` 成功；
- 模型目录不依赖运行时联网补文件；
- 模型 revision 和文件校验信息已保存到 `manifests/`。

### Phase 1：单节点统一服务

目的：排除模型、长度、显存和镜像问题，建立 non-PD 基线。

推荐先使用 8 卡 TP，以减少权重显存压力并与论文 SGL(MC) 的 Qwen 配置一致。完整 Slurm/Enroot 启动命令：

```bash
srun --partition=GAIA \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:8 \
  --time=01:00:00 \
  --container-image="$IMAGE_DIR/sglang:26.01-py3.sqsh" \
  --container-mounts="/mnt/lustre:/mnt/lustre,$CODE_HOME:/workspace/dual-path" \
  --container-workdir=/workspace/dual-path \
  --no-container-mount-home \
  bash -lc '
    python -m sglang.launch_server \
      --model-path "'"$MODEL_DIR"'" \
      --served-model-name qwen2.5-32b \
      --host 0.0.0.0 \
      --port 8000 \
      --tp-size 8 \
      --dtype bfloat16 \
      --kv-cache-dtype auto \
      --context-length 4096 \
      --mem-fraction-static 0.80 \
      --trust-remote-code \
      --enable-metrics
  '
```

第一次不要启用 HiCache、Mooncake、PD、FP8 KV、speculative decoding 或自定义 attention backend。按以下顺序验证：

1. `context-length=4096`，单请求、确定性生成；
2. 同一 prompt 连续请求 10 次，确认无失败且输出稳定；
3. `context-length=32768`，单请求；
4. 当前目标完成后才扩到 64K 和并发请求。

基础检查：

```bash
curl -f http://<compute-host>:8000/health
curl -s http://<compute-host>:8000/v1/models

curl -s http://<compute-host>:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-32b",
    "prompt": "Write a Python function that returns the sum of two integers.",
    "temperature": 0,
    "max_tokens": 64
  }'
```

若使用 Instruct checkpoint，再增加 `/v1/chat/completions` 测试；base checkpoint 的主要正确性入口使用 completions，避免 chat template 差异干扰。

验收条件：

- `/health` 和 `/v1/models` 成功；
- completion 返回非空文本，finish reason 正常；
- `temperature=0` 的重复请求 token 序列一致；
- 4K 与 32K 输入均能完成，无 OOM、hang、NaN 或请求错误；
- 8 张 GPU 均加载模型且有合理显存占用；
- 保存完整启动参数、服务日志、模型 revision 和 GPU 峰值显存。

### Phase 2：最小 1P1D

目的：先以两个节点验证 SGLang PD 数据路径和 Mooncake/NIXL transfer engine，再扩到 1P2D。

此阶段必须使用通过 §4.1 smoke 的派生镜像。当前原始 `26.01-py3.sqsh` 虽然暴露 Mooncake 参数，但未安装 `mooncake` 与 `sglang_router`，不能直接完成 PD。

2026-08-31 已先完成同节点功能验证：

```text
Prefill：GPU 0-3，TP=4
Decode：GPU 4-7，TP=4
Transfer：Mooncake 0.3.9，mlx5_0，RDMA
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
  --container-image="$IMAGE_DIR/sglang:26.01-py3-mooncake0.3.9.sqsh" \
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
scripts/pd_1p1d_multinode.sbatch
```

提交命令：

```bash
mkdir -p "$RUNTIME_HOME/logs/pd-1p1d"
sbatch scripts/pd_1p1d_multinode.sbatch
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

### Phase 3：目标 1P2D

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
脚本：scripts/pd_1p2d_multinode.sbatch
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

#### 扩展验证：跨节点 2P2D

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
脚本：scripts/pd_2p2d_multinode.sbatch
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

正式成功前有两次平台故障，不应误记为软件配置失败：

- Job `64855` 的 `dgx-gaia-30` 在 CUDA 初始化时返回
  `CUDA-capable device(s) is/are busy or unavailable`；
- Job `64858` 的 `dgx-gaia-46` 缺少 `nvidia-container-cli`，Pyxis 无法启动容器；
- Job `64860` 排除上述节点后无需修改 2P2D 参数即成功。

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
  --container-image="$IMAGE_DIR/sglang:26.01-py3-mooncake0.3.9.sqsh" \
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

### Phase 4：HiCache + Lustre

#### 目标语义：只复用 L3，不使用 L2 cache

本项目需要的逻辑层次为：

```text
L1：GPU KV-cache，仅服务当前计算和短期调度
L2：不作为 KV-cache 命中层，不允许实验收益来自 host DRAM cache
L3：共享外部 KV-cache，负责跨轮次、跨实例和重启后的前缀复用
```

“不使用 L2”指不在 host DRAM 中保留可再次命中的 KV 副本。实现上允许存在容量受控、生命周期短的 host staging/ring buffer，用于 L3→GPU、GPU→L3 和 PE→DE 搬运；staging buffer 不是缓存，数据传输完成后即可覆盖。

期望的第二轮数据流：

```text
1. 请求携带完整累积上下文到 Prefill。
2. Prefill 根据 token prefix 查询 L3。
3. L3 将命中的 Full Blocks 加载到 Prefill。
4. Prefill 只计算 append/cache-miss tokens 的 KV。
5. Prefill 将“L3 命中的 KV + 新计算的 KV”组成完整 prompt KV。
6. Mooncake Transfer Engine 通过 RDMA 将完整 prompt KV 发送给 Decode。
7. Decode 生成新 token，并把新产生的 KV 写回 L3，供下一轮使用。
```

因此必须同时存在两种独立能力：

论文中的两项 Mooncake 能力必须区分：

```text
Mooncake Transfer Engine: PD 阶段把 KV 从 Prefill 发送到 Decode
Mooncake Store:           HiCache 的共享 L3，保存和恢复多轮 KV
```

HiCache 不是独立的存储系统。它是 SGLang 内部的 KV-cache 分层管理器，负责：

- 用 RadixTree/prefix metadata 判断哪些 token 已命中；
- 在 L1、L2、L3 之间组织 KV block；
- 从 L3 prefetch 命中 block；
- 将新 KV 异步 backup/offload 到 L3；
- 向 scheduler 报告可复用 token，避免 Prefill 重算命中部分；
- 在 PD 模式中协调 Prefill 恢复、Decode offload 与 KV transfer。

实际保存 L3 数据的是 Mooncake Store、file、HF3FS 等 storage backend；当前 PD 使用的 Mooncake Transfer Engine 只负责 PE→DE 传输，并不自动启用 HiCache 或 L3。

#### 当前 SGLang 0.5.7 的限制

当前镜像源码的 host pool 分配逻辑为：

```python
if host_size > 0:
    size = host_size
else:
    size = device_pool.size * host_to_device_ratio

assert host_pool.size > device_pool.size
```

所以：

- `--hicache-size 0` 表示按 `--hicache-ratio` 分配 L2，不是关闭 L2；
- 当前协议强制 L2 大于 L1；
- 仅靠现有命令行参数无法得到严格的 L3-only HiCache；
- 文档和实验中不得把 `--hicache-size 0` 宣称为“禁用 L2”。

第一版可以启用 HiCache + L3，但必须通过重启 Prefill、清空本地 radix cache，或使用超过 L2 容量的 working set，确保观测到的命中来自 L3。严格 L3-only 需要后续修改 SGLang：

1. 将 host pool 改为固定的小型 staging ring，而不是可命中的 L2 pool；
2. 禁止 L2 admission 和 L2 lookup；
3. 保留 L3 metadata、prefetch、backup 与 block layout；
4. L3 block 到达 staging 后立即送入 L1/PE HBM；
5. 传输完成后释放/复用 staging slot；
6. 增加 `l2_hit_tokens == 0`、`l3_hit_tokens > 0` 的强制断言与指标。

在该修改完成前，实验配置应称为“L3 命中验证”，而不是“严格 L3-only”。

#### L3 backend 启动与配置

安装 `mooncake-transfer-engine-cuda13` 后应同时验证 `mooncake.engine` 和 `mooncake.store`。Mooncake Store 至少需要 `mooncake_master`；官方最小启动方式为：

```bash
mooncake_master --enable_http_metadata_server=true
```

SGLang 侧使用 `MOONCAKE_MASTER=<master-host>:50051` 和 `--hicache-storage-backend mooncake`。Decode 要持久化新生成 KV 时增加 `--disaggregation-decode-enable-offload-kvcache`。

Mooncake 的 SSD/file offload 默认面向节点本地 SSD。把 `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH` 指向 Lustre 是否正确、是否支持多 client 共享同一路径，需要单独验证，不能直接当作论文 3FS backend。现阶段保留以下 file backend 作为 Lustre 最小正确性基线。

最小 L3 正确性验证可以先使用 `file` backend 并把目录指向 Lustre。注意该配置仍会创建 SGLang 强制要求的 L2；它只用于证明 L3 hit 数据流：

```bash
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$CACHE_DIR/file-backend"
export SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=320GiB

python -m sglang.launch_server \
  ... \
  --page-size 64 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-size 0 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend file \
  --hicache-storage-prefetch-policy wait_complete
```

注意：官方文档说明 `file` backend 默认写 `/tmp/hicache`，默认是节点本地；只有显式将环境变量指向共享 Lustre 目录才可能跨节点看到同一命名空间。仍需通过跨节点写后读测试确认当前版本是否满足一致性与并发访问要求。若不满足，按以下优先级处理：

1. SGLang dynamic HiCache backend：实现以 Lustre 文件为介质的 `get/exists/set` 及批量接口；
2. Mooncake Store + SSD/file offload，把 offload path 指向 Lustre；
3. 仅为严格对齐论文时部署 HF3FS/3FS，不作为当前第一目标。

Decode 侧还需启用当前版本对应的异步 offload 参数（官方新版本示例为 `--disaggregation-decode-enable-offload-kvcache`），使每轮新 KV 落入 L3；Prefill 侧启用 HiCache，从 L3 恢复前缀。以 `--help` 的实际参数名为准。

验收条件：

- 第 1 轮产生 L3 写流量；
- 第 2 轮提交完整累积上下文时出现可验证的 L3 hit；
- Prefill 日志/metrics 报告 L3 hit tokens，而不是重新计算全部上下文；
- 重启 Prefill、消除 L1/L2 状态后仍可从 L3 命中，证明命中不是本地缓存；
- Prefill compute tokens 等于 append/cache-miss tokens；
- Prefill→Decode 传输字节包含 hit KV 与 miss KV；
- Decode 完成后，新 KV 被写回 L3；
- 严格 L3-only 版本必须满足 `l2_hit_tokens == 0`；
- 保存 L1/L2/L3 hit、KV 读写字节、Lustre 吞吐和 TTFT 分解。

### Phase 5：论文瓶颈趋势

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
- 正式规模：32、64、128 agents，稳定后再扩；
- Append scale：`1×, 1.5×, 2×, 3×`；
- Generation scale：`1×, 1.5×, 2×, 3×`；
- 每个正式配置至少重复 3 次，报告中位数及 min/max 或置信区间；
- 每组测试前明确采用 warm-cache 还是 cold-cache，二者不可混报。

离线 replay：所有 agent 同时开始，记录最后一个 trajectory 完成时的 JCT。

在线 replay：agent 按 Poisson 过程到达，逐步提升 APS；每档至少运行至稳定窗口。若实验成本过高，先使用 10 分钟稳定窗口探索，最终结果再按论文的 150 秒滑窗和 30 分钟历史窗口执行。

### Phase 6：DualPath 近似实现

只有 Phase 5 已证明 Prefill 侧 Lustre 读取饱和后，才进入代码改造。目标数据路径：

```text
PE read path: Lustre -> PE DRAM -> PE HBM -> DE
DE read path: Lustre -> DE DRAM -> PE HBM
```

最小实现按以下顺序进行：

1. 在 Decode 节点增加只读 L3 fetch，先读入 host buffer；
2. 复用 Mooncake/NIXL transfer engine 将命中的 KV 从 DE 传给 PE；
3. 以 request 为粒度在 PE-read 与 DE-read 二选一，不做单请求拆分；
4. 先用最短读队列策略：`argmin(read_queue_bytes / measured_bandwidth)`；
5. 再加入 unfinished tokens，避免网络平衡但 GPU 失衡；
6. 最后才研究 layer-wise streaming、CNIC QoS/VL 和 copy/compute overlap。

第一版调度输入至少包含：

- 每节点待处理读字节数或读 token 数；
- 每个 engine 的 unfinished token 数；
- Decode 剩余 HBM/host buffer；
- PE/DE pair 的 transfer queue；
- 最近窗口的 Lustre 实测带宽。

不要在第一版实现中声称已复现论文的 CNIC-centric traffic manager。论文把模型通信和 KV 流量映射到不同 IB VL，并在交换机与 NIC 上配置仲裁；这通常需要管理员权限和全网一致配置。

## 6. Benchmark 与观测

### 6.1 服务 smoke benchmark

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

### 6.2 必须记录的指标

端到端：

- JCT、agent throughput、request throughput、prompt/output token throughput；
- TTFT、TTST、TPOT、ITL 的 mean/P50/P90/P99；
- 成功/失败/超时/OOM 请求数。

缓存与 I/O：

- 每轮 context、append、generation、hit token 和 hit ratio；
- L1/L2/L3 hit/miss；严格 L3-only 实验要求 L2 hit 恒为 0；
- 每节点 Lustre read/write bytes、GB/s、IOPS、平均请求大小；
- PE/DE 读队列长度；
- cold-cache 与 warm-cache 标记。

计算与网络：

- 每卡 GPU utilization、HBM 使用量、功耗；
- Prefill compute、KV load、PD transfer、queue wait 分解；
- 每个 IB 端口发送/接收字节及错误计数；
- Decode-0/Decode-1 请求数与 unfinished tokens；
- CPU、DRAM、NUMA 和本地 `/raid` 使用量。

环境元数据：

- Slurm Job ID、节点列表、时间；
- 模型 repo revision 和 tokenizer revision；
- 镜像 SHA-256、SGLang/Git commit、CUDA/driver/NCCL/Mooncake；
- 完整服务参数、环境变量（排除 secret）、trace seed；
- Lustre 路径、stripe count/size、容量与文件数。

### 6.3 实验有效性检查

每次正式运行前后检查：

1. 所有节点 GPU/NIC 健康且无其他作业干扰；
2. 模型和镜像完全一致；
3. 采样参数固定，强制恰好生成 `gen_len`，忽略 EOS；
4. trace 顺序和随机种子固定；
5. cache 初始状态符合 cold/warm 定义；
6. 服务错误率为 0，或明确报告失败；
7. Lustre 未出现明显外部拥塞；若出现则废弃该次结果；
8. 每次只改变一个主要变量。

## 7. 结果判定

当前里程碑 M1 的完成标准：

- Qwen2.5-32B 在单节点 `TP=8` 稳定运行；
- 4K 和 32K completion 正确完成；
- 确定性请求可重复；
- 记录模型 revision、镜像 SHA-256、Enroot/Slurm/SGLang 命令；
- 本里程碑不要求 Mooncake、PD 或 SWE-bench。

里程碑 M2 的完成标准：

- 派生镜像可 import `mooncake.engine`、`mooncake.store` 和 `sglang_router`；
- 1P1D 和 1P2D 均通过 32K 请求；
- 1P2D 两个 Decode 实例都有流量；
- Decode 写出的 KV 可持久化到 Lustre；
- 后续 Prefill 能从 Lustre 命中同一 trajectory 的前缀；
- 32K/48K/64K 下能观察到 KV 读取随 context 增长；
- Prefill 节点存储带宽高于 Decode 节点，形成论文所述不平衡。

里程碑 M3 的完成标准：

- DE-read path 可用；
- 单路径与双路径总 GPU 数一致；
- 双路径提高聚合 Lustre 读取带宽并降低 JCT 或 TTFT；
- TPOT 无显著回归；
- 所有结论以实测数据为准，不以理论峰值代替。

## 8. 风险与决策点

- **镜像兼容性**：当前基础镜像为 NGC `26.01-py3`，实测 SGLang `0.5.7+nv26.1`，有 PD/HiCache adaptor 但没有 Mooncake、NIXL 和 Router。当前统一推理直接使用基础镜像；PD 使用独立派生镜像。
- **Mooncake 版本**：CUDA 13 wheel 0.3.9 已通过同节点 Qwen2.5-32B PD 验证；不要未经测试把 0.3.13 与旧 SGLang 固化在一起。
- **PD layout 一致性**：Prefill/Decode 必须固定相同 `tp-size`、`kv-cache-dtype`、page size、模型 revision 和 attention/KV layout；已知不一致会导致连接冲突或越界写。
- **Router 缺失**：基础镜像中没有 `sglang_router`，PD 前必须安装并固定 Router 版本。
- **Lustre backend 语义**：共享目录不自动等于正确的分布式 KV store。需要验证 key namespace、一致性、并发写、eviction 和重启恢复。
- **HiCache L2 强制分配**：当前 SGLang 0.5.7 强制 host pool 大于 device pool，`--hicache-size 0` 不能关闭 L2。严格 L3-only 需要代码修改；修改前必须通过重启/清缓存证明收益来自 L3。
- **64K workload**：base 模型原生支持；Instruct checkpoint 默认配置不直接支持，避免不经说明地修改 RoPE。
- **PD 与 DualPath 混淆**：PD transfer 是 PE→DE；DualPath 新路径的关键方向是 storage→DE→PE，方向相反。
- **流量隔离权限**：IB VL/QoS 很可能需要管理员协助，先观测干扰，再决定是否申请配置。
- **共享存储噪声**：Lustre 是共享基础设施，必须重复实验并记录并发负载，否则小幅差异不能归因于系统改动。
- **密钥泄露**：历史参考脚本中的明文 NGC key 必须吊销，后续结果包不得包含 `.credentials`。

## 9. 推荐执行顺序

```text
平台探测
  -> 核验现有 26.01 基础镜像
  -> 下载并固定 Qwen2.5-32B revision
  -> Unified TP=8 / 4K 确定性推理
  -> Unified TP=8 / 32K
  -> 当前里程碑完成
  -> 构建 Mooncake + Router 派生镜像
  -> 1P1D / 4K、32K
  -> 1P2D / 32K、48K、64K
  -> HiCache + L3 最小正确性
  -> Lustre L3 单节点重启命中
  -> Lustre L3 Decode 写 / Prefill 读
  -> 禁止 L2 admission/lookup，仅保留 staging buffer
  -> 断言 L2 hit=0、L3 hit>0
  -> 论文 trace replay
  -> 单路径瓶颈确认
  -> DualPath 代码改造
```

在当前阶段，只完成 `Unified TP=8 -> 4K -> 32K`。不安装 Mooncake、不启用 HiCache、不运行 SWE-bench，也不同时引入 64K 高并发和 PD。统一推理稳定后再创建派生镜像，避免把模型问题、容器依赖问题和 RDMA 问题混在一起。

## 10. 参考资料

- 项目内论文：`docs/Wu et al. - 2026 - DualPath Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference.pdf`
- SGLang PD Disaggregation：https://docs.sglang.io/docs/advanced_features/pd_disaggregation
- SGLang HiCache Best Practices：https://docs.sglang.io/docs/advanced_features/hicache_best_practices
- SGLang Benchmark Serving：https://docs.sglang.io/docs/developer_guide/bench_serving
- NVIDIA SGLang 26.01 Release Notes：https://docs.nvidia.com/deeplearning/frameworks/sglang-release-notes/rel-26-01.html
- Mooncake Quick Start：https://kvcache-ai.github.io/Mooncake/getting_started/quick-start.html
- Mooncake + SGLang PD：https://kvcache-ai.github.io/Mooncake/deployment/integrations/sglang/pd-disaggregation.html
- Mooncake + SGLang HiCache：https://kvcache-ai.github.io/Mooncake/deployment/integrations/sglang/hicache-quick-start.html
- Qwen2.5-32B：https://huggingface.co/Qwen/Qwen2.5-32B
- NVIDIA SGLang NGC：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/sglang
- NVIDIA DGX SuperPOD Containers：https://docs.nvidia.com/dgx-superpod/user-guide-dgx-superpod/latest/containers.html
