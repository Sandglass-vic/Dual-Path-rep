# SGLang、Enroot 与 Mooncake 镜像

> 返回 [文档索引](README.md)

计算节点已实测 Enroot 3.2.0，Pyxis 已集成到 `srun`。正式作业使用 Lustre 上 `.sqsh` 的绝对路径，显式挂载代码和运行目录，默认不启用 writable root，也不自动挂载 home。

当前镜像已经导入：

```text
/mnt/lustre/users/sezhou/dual-path/images/sglang:26.01-py3.sqsh
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

## Mooncake 派生镜像

结论与当前状态：

- **当前“跑通 Qwen2.5-32B 统一推理”不需要 Mooncake，也不需要重建镜像。**
- 为后续启用 Mooncake PD transfer 和 HiCache Mooncake Store，已从基础镜像成功派生新 sqsh，没有修改原始镜像。
- Mooncake Transfer Engine 和 Mooncake Store 是两种用途，但现代 Mooncake wheel 同时提供 `mooncake.engine` 与 `mooncake.store`。
- 论文的 SGL(MC) 同时用了 Transfer Engine（PE→DE）和 Store/HiCache（L3）；仅安装 wheel 不会自动得到 3FS，Lustre 也不等同于 3FS。

26.01 镜像使用 CUDA 13.1，官方 Mooncake 当前要求安装 CUDA 13 变体：

```text
mooncake-transfer-engine-cuda13
```

不能直接安装不带后缀的 CUDA 12 wheel。`0.3.9` 是早期验证时首个可用的 CUDA
13 wheel，所以旧镜像曾固定它；这不是继续使用旧版的兼容性依据。当前已升级到
`0.3.13.post1`，因为 `0.3.13` 首次正式包含 descriptor-based DFS、POSIX
adapter 和 `ReplicateConfig.dfs_replica_num`，`post1` 则修正该 release 的
wheel 发布流程。

当前组件：

```text
基础镜像：SGLang 0.5.7+nv26.1 / CUDA 13.1
Mooncake：mooncake-transfer-engine-cuda13 0.3.13.post1
Router：sglang-router 0.3.1（sgl-model-gateway 0.3.1）
GPU：NVIDIA H100 80GB HBM3
```

当前正式派生镜像：

```text
/mnt/lustre/users/sezhou/dual-path/images/sglang:26.01-py3-mooncake0.3.13.post1-file-l3-dual-staging-v1.sqsh
候选构建 Job：67193，COMPLETED，ExitCode=0:0
TP8 1P1D smoke Job：67234，COMPLETED，ExitCode=0:0
正式默认配置 smoke Job：67246，COMPLETED，ExitCode=0:0
正式发布 Job：67243，COMPLETED，ExitCode=0:0
SHA-256：5274263decb923c869b594efef573a2d1f87313538c22caf01cc2ca8470dff1d
```

该镜像使用两个独立 Host pool：40 GB/rank 的 `layer_first` Decode PD pinned
staging，以及由 `DECODE_HICACHE_RATIO` 控制的 `page_first_direct` Decode
HiCache Host pool。HiCache L3 使用 Lustre file backend；Mooncake 仅负责 P→D
Transfer Engine，不使用 Mooncake Store/global segment。

当前 Timing V3 测试镜像（2026-09-17）：

```text
/mnt/lustre/users/sezhou/dual-path/images/sglang-file-l3-dual-staging-69302.sqsh
构建 Job：69302，COMPLETED，ExitCode=0:0
SHA-256：99f8b0ba9cd0c1c58da526fc5df5bfc78325f6a4b09cd0cbce1216602d6f88af
```

该镜像加入低开销 `DUALPATH_TTFT_V3` request/rank 聚合日志、Prefill/Decode
Lustre read 与 scatter 分离、GPU H2D/computation overlap 归属、双向 Mooncake
transfer timing，以及 V3 offline 汇总所需的 raw/critical/overlap 字段。构建脚本
自带的 patch assertions 和 SGLang 源码 compile 已通过。Job `69290` 修正
Decode preallocation queue 的 TP CPU group 传递，使用 queue 自身的
`self.gloo_group`，避免启动阶段访问不存在的 `self.attn_tp_cpu_group`；Job
`69293` 补齐 `schedule_batch.py` 的 `DUALPATH_TTFT_DETAIL` import，避免首个
Decode output 调用 `Req.log_time_stats()` 时出现 `NameError`。Job `69297` 将
Decode Lustre prefetch 生命周期移入 scheduler-owned `DecodePreallocQueue`，复用
Prefill HiCacheController 的 query/I/O worker 流程，并限制每个 TP rank 仅有一个
有序在途 prefetch，避免同步 TP collective 在并发请求间交错；Decode 仍保留独立
Host staging buffer。Job `69302` 修正 receiver 能力判断：只有显式声明支持
scheduler prefetch 的 `MooncakeKVReceiver` 才接收 request、Host staging 和 timing
扩展参数，`FakeKVReceiver` 等基线 receiver 保持原有三参数 `init()` 调用。
`scripts/pd_alloc.sh` 默认使用该镜像。**Timing V3，尚未 smoke，由用户验证；
当前不得视为已验证性能基线。**

上一版 scheduler-owned Decode prefetch 镜像：

```text
/mnt/lustre/users/sezhou/dual-path/images/sglang-file-l3-dual-staging-69297.sqsh
构建 Job：69297，COMPLETED，ExitCode=0:0
SHA-256：02b3444bbfa3bc31eee650bde689ecd5dce5cadf8677a880f93066c0efc78669
```

该镜像会向 `FakeKVReceiver.init()` 传入 Mooncake 专用扩展参数，运行时触发
`unexpected keyword argument 'request_id'`，不得继续用于实验。

更早的 Timing V3 启动修复镜像：

```text
/mnt/lustre/users/sezhou/dual-path/images/sglang-file-l3-dual-staging-69293.sqsh
构建 Job：69293，COMPLETED，ExitCode=0:0
SHA-256：5214d7e17a963144b674430635574fde5f7cdb9da7a2dfbb1bf2fec9c852dd3e
```

该镜像不包含 scheduler-owned 异步 Decode prefetch。Allocation `69298` 误用该
旧默认镜像，不能用于验证 Job `69297` 的实现。

上一版 KV read/scatter timing 测试镜像：

```text
/mnt/lustre/users/sezhou/dual-path/images/sglang-file-l3-dual-staging-69108.sqsh
构建 Job：69108，COMPLETED，ExitCode=0:0
SHA-256：6b3cb67ed3737184c8ebffc856a7f73c303bbae457a94215e2f146f530e73805
```

该镜像继承 `69054` 的 O_DIRECT Prefill buffer reuse：在
`DUAL_PATH_O_DIRECT_READ=1` 时让 Prefill 每个 TP rank 复用一个
4096-byte-aligned pinned page，替代上游 generic path 每批创建 128 个临时 page
的行为。它新增 Prefill/Decode `storage_backend.get()` 与 Host-pool scatter 的独立
计时。构建期源码、配置实例化与 patch 顺序检查已通过；该镜像已由 Timing V3
镜像 `69302` 取代，仍保留用于旧结果复现。

候选 `shared-host-v5`（构建 Job `67084`，SHA-256
`bdeae5df...6237f2`）尝试让 P→D staging 与 Decode D2H→L3 offload 共用
HiCache Host pool，但动态 Job `67086` 发现 Mooncake Store 会把请求的
`layer_first` 自动改回 `page_first_direct`，因此候选未通过、不得用于正式
benchmark。Mooncake `0.3.9`、旧 timing、共享池候选与未修正 L3 timing 的
更早中间镜像均不得用于正式实验。

实际构建入口：

```bash
sbatch scripts/infra/build_benchmark_timing_image.sbatch
```

### Patch 验证与正式发布流程

新的 SGLang/Mooncake patch 不应直接覆盖正式镜像。推荐使用三阶段流程：

1. 从最后一个稳定镜像或基础镜像创建 writable Enroot container，应用 patch；
2. 直接在 `images/` 导出 Job/commit 专属的候选镜像，并用该镜像执行多节点 smoke；
3. 仅在 smoke 成功后更新 SHA-256、Job ID 和 benchmark 默认路径；失败时删除该
   Job 镜像，保留原默认版本。

`build_benchmark_timing_image.sbatch` 支持通过 `OUTPUT_IMAGE` 指向候选路径：

```bash
candidate="$RUNTIME_HOME/images/sglang-shared-host-${USER}-$(date +%s).sqsh"
build_job=$(
  sbatch --parsable \
    --export=ALL,OUTPUT_IMAGE="$candidate" \
    scripts/infra/build_benchmark_timing_image.sbatch
)

# build 成功后，smoke 启动时显式覆盖 IMAGE="$candidate"。
# smoke 验收通过后记录校验值并更新 benchmark 默认 IMAGE：
sha256sum "$candidate"
```

候选镜像必须位于所有测试节点可见的 Lustre `images/` 路径。完全不导出 sqsh、直接修改
每个 Pyxis runtime root 虽然可行，但多节点需要分别应用 patch，容易出现节点间
源码不一致，因此不作为正式验证方式。候选镜像不是正式发布物，不得写入 benchmark
默认值；测试失败或取消后应自动清理。

该脚本从未修改的 `sglang:26.01-py3.sqsh` 创建 writable Enroot container，
安装固定 Mooncake Transfer Engine wheel 与 Router，运行
`scripts/infra/patch_sglang_benchmark_timing.py`，执行 import/API/source compile
断言后直接导出到 `images/` 下的 Job 专属候选 sqsh。脚本不再注入 Mooncake
Store DFS backport；正式实验必须经过 smoke，并记录 SHA-256 后再更新默认路径。

基础镜像执行 `pip check` 会报告 `triton-kernels` 缺少 `pytest`。这是基础镜像已有的开发/测试依赖缺口，不是 Mooncake 安装造成的运行依赖冲突，因此未为了消除该提示修改推理环境。

构建脚本还回移了当前旧 SGLang Mooncake backend 缺少的 request-scoped DFS
replica 配置、all-layer 同步 transfer 的 KV-index 分批，并加入 benchmark
timing 埋点。前两项上游对应功能分别为 SGLang PR `#37390` 与 `#32758`；
不能把这些回移描述成 NVIDIA 26.01 原生能力。

派生镜像已经通过：

- `mooncake.store.MooncakeDistributedStore` import；
- `mooncake.engine.TransferEngine` import；
- Store 与 Engine 同进程联合 import；
- `sglang_router` import 和 `sgl-model-gateway --version`；
- H100/CUDA 13.1 可见性；
- SGLang Mooncake PD 参数检查；
- `ReplicateConfig.dfs_replica_num` API 与 SGLang backport 静态检查；
- Mooncake Store 跨节点 Lustre DFS 写后读（Job `65548`）；
- Qwen2.5-32B 1P1D HiCache/PD/DFS 动态 4-turn smoke（Job `65557`）；
- Qwen2.5-32B 1P1D TP8、8-HCA Decode Host staging 2-turn smoke
  （Job `66924`）：Prefill RDMA → Decode pinned Host → HBM，16/16
  request/rank timing markers 完整；
- Qwen2.5-32B 1P1D、32K、1-agent、2-turn Decode-owned offload
  （Job `66987`）：Prefill read-only/立即释放，Decode 从 `cached_tokens`
  边界发布新 KV，第二轮 L3 恢复成功。
- Qwen2.5-32B 1P1D、32K、1-agent、2-turn file L3 与双 Host pool smoke
  （Job `67234`）：8 张 HCA 均初始化，PD Host staging 的 TP8 H2D marker
  完整；两轮 Decode file backup 均在 ACK 后生成 8/8 rank 原子 marker，
  `source=file_backend` 且路径位于 run-specific `l3-sync`。
- 使用正式默认镜像并移除 Store-only DFS/global-segment 环境后的同配置 smoke
  （Job `67246`）再次通过，确认 Mooncake master 仅承担 P→D Transfer Engine
  metadata 服务时数据流正常。

若后续多节点 RDMA 或 HiCache Store smoke 失败：

1. 先检查回移补丁与 `0.3.13.post1` API，而不是退回会失去 POSIX DFS 的
   `0.3.9`；
2. 若旧 SGLang adaptor 无法稳定工作，再升级 NVIDIA SGLang 镜像并重新跑统一
   推理和 PD 基线；
3. 不以论文 commit `19089aa` 作为当前安装目标，它用于说明论文基线，而不是与
   2026 年 Mooncake wheel 的兼容承诺。

Store/Lustre 的配置、对象发布语义、实测和限制见
[Mooncake Store 与 Lustre POSIX DFS](mooncake-store.md)。
