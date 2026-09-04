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

派生镜像：

```text
/mnt/lustre/users/sezhou/dual-path/images/sglang:26.01-py3-mooncake0.3.13.post1-timing-v2.sqsh
最终构建 Job：66119，COMPLETED，ExitCode=0:0
SHA-256：792bffc289fe519803d4c97453d4fe56afeed92866aa4ceafc8de0b7691e8ef2
```

镜像目录只保留上述最终派生版和最初基础版；Mooncake `0.3.9`、旧 timing 与
未修正 L3 timing 的中间镜像均已删除。

实际构建入口：

```bash
sbatch scripts/infra/build_benchmark_timing_image.sbatch
```

该脚本从未修改的 `sglang:26.01-py3.sqsh` 创建 writable Enroot container，
安装固定 wheel 与 Router，依次运行
`scripts/infra/patch_sglang_mooncake_dfs.py`、
`scripts/infra/patch_sglang_benchmark_timing.py`，执行
import/API/source compile 断言后导出新 sqsh。重新运行会覆盖同名输出镜像，
因此正式实验必须记录 SHA-256。

基础镜像执行 `pip check` 会报告 `triton-kernels` 缺少 `pytest`。这是基础镜像已有的开发/测试依赖缺口，不是 Mooncake 安装造成的运行依赖冲突，因此未为了消除该提示修改推理环境。

构建脚本还回移了当前旧 SGLang Mooncake backend 缺少的 request-scoped DFS
replica 配置，并加入 benchmark timing 埋点。上游对应功能为尚未合入的
SGLang PR `#37390`；不能把该回移描述成 NVIDIA 26.01 原生能力。

派生镜像已经通过：

- `mooncake.store.MooncakeDistributedStore` import；
- `mooncake.engine.TransferEngine` import；
- Store 与 Engine 同进程联合 import；
- `sglang_router` import 和 `sgl-model-gateway --version`；
- H100/CUDA 13.1 可见性；
- SGLang Mooncake PD 参数检查；
- `ReplicateConfig.dfs_replica_num` API 与 SGLang backport 静态检查；
- Mooncake Store 跨节点 Lustre DFS 写后读（Job `65548`）；
- Qwen2.5-32B 1P1D HiCache/PD/DFS 动态 4-turn smoke（Job `65557`）。

若后续多节点 RDMA 或 HiCache Store smoke 失败：

1. 先检查回移补丁与 `0.3.13.post1` API，而不是退回会失去 POSIX DFS 的
   `0.3.9`；
2. 若旧 SGLang adaptor 无法稳定工作，再升级 NVIDIA SGLang 镜像并重新跑统一
   推理和 PD 基线；
3. 不以论文 commit `19089aa` 作为当前安装目标，它用于说明论文基线，而不是与
   2026 年 Mooncake wheel 的兼容承诺。

Store/Lustre 的配置、对象发布语义、实测和限制见
[Mooncake Store 与 Lustre POSIX DFS](mooncake-store.md)。
