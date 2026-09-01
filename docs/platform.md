# GAIA 实验平台记录

本文只记录与 DualPath 复现实验直接相关的平台事实、验证命令和待确认项。标记为“已实测”的信息来自 2026-08-31 的登录节点或一次 GAIA 计算节点探测；集群配置可能变化，正式实验应把同样的信息写入每个结果目录。

## 1. Slurm

登录节点已实测：

```text
ClusterName: mtle2edeepops
默认分区: GAIA
调度器: sched/backfill
资源选择: select/cons_tres
默认 MPI: pmix
Slurm controller: tlv01-e2e-slurm12:6817
Slurmd port: 6818
```

探测时 `GAIA` 分区为 `up`，`sinfo` 共报告 55 个节点：

```text
1 × gpu:1
3 × gpu:8
51 × gpu:8（GRES 带 socket 标注）
```

这只是探测时刻的配置快照，不等于作业可立即使用的空闲节点数。正式申请前执行：

```bash
sinfo -p GAIA -N -o '%N %T %G %c %m'
squeue -p GAIA
```

常用资源申请：

```bash
# 单节点交互调试
srun --partition=GAIA \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:8 \
  --time=01:00:00 \
  --pty bash

# 3 节点 1P2D
salloc --partition=GAIA \
  --nodes=3 \
  --gres=gpu:8 \
  --time=04:00:00
```

在 allocation 中取得节点名：

```bash
scontrol show hostnames "$SLURM_JOB_NODELIST"
```

正式作业必须记录：

```bash
scontrol show job "$SLURM_JOB_ID"
scontrol show hostnames "$SLURM_JOB_NODELIST"
```

2026-09-01 的 2P2D 调试中发现两个节点级异常：

- `dgx-gaia-30` 已获得整节点 GPU allocation，但 CUDA 初始化返回
  `CUDA-capable device(s) is/are busy or unavailable`；
- `dgx-gaia-46` 的 Enroot NVIDIA hook 报告找不到
  `nvidia-container-cli`，Pyxis 容器无法启动。

这两项是当时的平台状态，不代表节点永久不可用。自动化 smoke 若在模型加载前
遇到相同错误，应保存日志、取消剩余 server step，并临时通过
`sbatch --exclude=<nodes>` 重试，同时向平台管理员报告；不要通过修改 SGLang
模型或 PD 参数掩盖节点故障。排除上述节点后，Job `64860` 在
`dgx-gaia-[51-54]` 上完成 2P2D 验证。

## 2. 计算节点硬件

实测节点：`dgx-gaia-12`。

GPU：

```text
8 × NVIDIA H100 80GB HBM3
每卡 nvidia-smi 报告显存: 81,559 MiB
探测时 NVIDIA driver: 580.95.05
```

CPU 与内存：

```text
2 × Intel Xeon Platinum 8480C
每 socket 56 cores
224 logical CPUs
约 2.0 TiB DRAM
无 swap
```

本地存储：

```text
/raid
文件系统: ext4
容量: 约 28 TiB
```

`/raid/users` 目录存在，但节点本地数据不会自动在其他节点共享。适合临时文件、解压和本地 I/O 对照，不适合作为跨节点 KV-cache 后端。作业结束后是否保留、是否会自动清理仍需管理员确认。

快速复核：

```bash
hostname
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader
lscpu
free -h
df -hT /raid
```

## 3. InfiniBand 与 NIC

计算节点实测存在 `mlx5_0` 到 `mlx5_11` 共 12 个 IB device。探测时：

```text
400 Gb/s ACTIVE: mlx5_0, mlx5_3, mlx5_4, mlx5_5,
                 mlx5_6, mlx5_9, mlx5_10, mlx5_11
200 Gb/s ACTIVE: mlx5_1, mlx5_7
DOWN:            mlx5_2, mlx5_8
```

接口映射：

```text
mlx5_0  -> ibp24s0
mlx5_1  -> ibp41s0f0
mlx5_2  -> ibp41s0f1  (DOWN)
mlx5_3  -> ibp64s0
mlx5_4  -> ibp79s0
mlx5_5  -> ibp94s0
mlx5_6  -> ibp154s0
mlx5_7  -> ibp170s0f0
mlx5_8  -> ibp170s0f1 (DOWN)
mlx5_9  -> ibp192s0
mlx5_10 -> ibp206s0
mlx5_11 -> ibp220s0
```

到一个 Lustre server 地址 `192.1.3.230` 的路由实测为：

```text
dev ibp41s0f0
src 192.1.0.12
```

因此当前节点的 Lustre 流量经过 `ibp41s0f0 / mlx5_1`，链路速率为 200 Gb/s。该结果支持“存储网络与 8 张 GPU 的 400G compute fabric 分离”的判断，但仍应在每个作业节点复核，不能假设所有节点映射永久相同。

对 SGLang PD transfer，候选 compute-fabric device 可先从 `mlx5_0` 开始验证，**不要使用 `mlx5_1`**，以免把 PD KV 流量放到 Lustre 存储链路。最终应根据 GPU/NIC PCIe affinity 选择每卡配对设备，而不是仅凭编号。

复核命令：

```bash
ibdev2netdev
ibstat
rdma link
ip route get 192.1.3.230

for dev in /sys/class/infiniband/mlx5_*; do
  name=${dev##*/}
  printf '%s | %s | %s\n' \
    "$name" \
    "$(<"$dev/ports/1/rate")" \
    "$(<"$dev/ports/1/state")"
done

nvidia-smi topo -m
```

待确认：

- 各 H100 与 `mlx5_0,3,4,5,6,9,10,11` 的一一 PCIe/NVLink affinity；
- SGLang/Mooncake 使用单 device 还是 device list 性能更好；
- compute fabric 与 storage fabric 是否具有不同 IB partition/P_Key；
- 是否允许用户读取端口计数器；
- IB VL/QoS 配置是否需要管理员操作。

## 4. Lustre

Lustre 仅在计算节点可见，登录节点上 `/mnt/lustre` 不存在。

计算节点实测：

```text
挂载点: /mnt/lustre
文件系统类型: lustre
容量: 约 518 TiB
已用: 约 21 TiB
可用: 约 493 TiB
探测时使用率: 4%
后端名称: gaiafs
```

公共上层目录 `/mnt/lustre/gaia` 存在；当前用户的默认子目录尚不存在。项目不应直接假定 `/mnt/lustre/gaia/$USER` 已创建，应先确认平台规定的个人路径和 quota，再创建：

```bash
srun -p GAIA -N1 -n1 --gres=gpu:1 \
  bash -lc 'ls -ld /mnt/lustre /mnt/lustre/gaia; lfs quota -u "$USER" /mnt/lustre'
```

建议的项目布局：

```text
/mnt/lustre/gaia/<user>/dual-path/
├── images/      Enroot sqsh
├── models/      Qwen2.5-32B checkpoint
├── kv-cache/    HiCache L3 / Mooncake offload
├── traces/      replay trace
├── results/     结构化 benchmark 结果
└── logs/        服务与采集日志
```

创建 KV-cache 目录后记录 Lustre layout：

```bash
lfs getstripe "$CACHE_DIR"
lfs df -h "$CACHE_DIR"
lfs quota -u "$USER" "$CACHE_DIR"
```

stripe 参数不能凭经验直接固定。先使用 1、4、8 个 OST 和 1 MiB、4 MiB、16 MiB stripe size 做大文件顺序读写及实际 KV block workload 测试，再选主实验配置。所有对比组必须使用同一 layout。

需要分别测量：

- 单进程顺序读写；
- 同节点 8 并发读写；
- 1P2D 三节点同时读取；
- 4 MiB、16 MiB、64 MiB、256 MiB 等不同请求大小；
- cold-cache 与 warm-cache；
- `/raid` 和 `/mnt/lustre` 对照。

不应通过删除系统 cache 或影响其他用户的方式制造 cold-cache。若无管理员支持，使用全新文件/key namespace，并在结果中说明其近似性。

## 5. Enroot 与 Pyxis

登录节点未发现 `enroot`，计算节点已实测：

```text
Enroot: /usr/bin/enroot
版本: 3.2.0
Pyxis: 已集成到 srun
```

`srun --help` 已确认支持：

```text
--container-image=[USER@][REGISTRY#]IMAGE[:TAG]|PATH
--container-mounts=SRC:DST[:FLAGS][,SRC:DST...]
```

推荐预先把大镜像导入 Lustre 上的 `.sqsh`，避免每次作业重新拉取。项目当前指定的镜像源为 NVIDIA 内部 GitLab registry：

```text
gitlab-master.nvidia.com/dl/dgx/sglang:25.08-py3-devel
```

Enroot URI 使用 `#` 分隔 registry host 与 repository。私有 registry 的凭据应写入 `$HOME/.config/enroot/.credentials`，示意格式如下，其中占位符不能原样使用：

```text
machine gitlab-master.nvidia.com login <gitlab-user> password <access-token>
```

执行：

```bash
export ENROOT_CONFIG_PATH=$HOME/.config/enroot
chmod 600 "$ENROOT_CONFIG_PATH/.credentials"

enroot import \
  --output /mnt/lustre/gaia/<user>/dual-path/images/sglang-25.08-py3-devel.sqsh \
  'docker://gitlab-master.nvidia.com#dl/dgx/sglang:25.08-py3-devel'
```

容器 smoke test：

```bash
srun --partition=GAIA \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:8 \
  --time=00:10:00 \
  --container-image=/mnt/lustre/gaia/<user>/dual-path/images/sglang-25.08-py3-devel.sqsh \
  --container-mounts=/mnt/lustre:/mnt/lustre \
  --no-container-mount-home \
  bash -lc '
    nvidia-smi
    python -c "import sglang, torch; print(sglang.__version__, torch.__version__)"
  '
```

原则：

- 正式作业使用 sqsh 的绝对路径；
- 默认不使用 `--container-writable`；
- 显式 mount 项目、模型、缓存和结果目录；
- 不自动 mount home，避免把 credential 和无关文件带入容器；
- 如确需 Hugging Face、NGC 或内部 GitLab token，只挂载最小只读 credential 文件；
- 导入后记录 `sha256sum`，不要仅记录可变 tag。

`25.08-py3-devel` 是当前用户指定的内部开发镜像。tag 中的年月不能证明其内部 SGLang 源码比公开镜像更新，因此导入后必须记录：

```bash
sha256sum "$IMAGE_DIR/sglang-25.08-py3-devel.sqsh"
python -c 'import sglang; print(sglang.__version__)'
git -C /opt/sglang rev-parse HEAD 2>/dev/null || true
```

若镜像没有保留 `.git`，至少记录 `/opt/sglang` 的版本文件、Python package version 及 PD/HiCache 参数探测结果。参考脚本提到的 `sglang-25.11.sqsh` 路径已实测不存在，不能直接作为当前实验依赖。

## 6. 软件状态

裸计算节点环境中未安装 Python `sglang` 和 `mooncake` 包，实验依赖必须来自容器或项目自建环境。

每个候选容器都要记录：

```bash
python - <<'PY'
import torch
import sglang
print("sglang", sglang.__version__)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
PY

python -m sglang.launch_server --help
python -m sglang_router.launch_router --help
nvidia-smi
```

还需确认：

- `mooncake-transfer-engine` 是否可 import；
- Router 是否已包含在镜像；
- PD backend 支持 `mooncake` 或 `nixl`；
- HiCache 是否支持 `file`、`mooncake` 和 dynamic backend；
- Qwen2.5-32B、BF16、TP=8、64K context 的兼容性；
- NCCL 是否正确选择 8 个 compute NIC。

## 7. 1P2D 的平台资源布局

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

## 8. 平台基线与实验记录模板

每次正式实验在结果目录保存：

```text
metadata/
├── slurm_job.txt
├── nodes.txt
├── nvidia_smi.txt
├── nvidia_topo.txt
├── ib_devices.txt
├── routes.txt
├── lustre_mount.txt
├── lustre_stripe.txt
├── container_versions.txt
├── container_sha256.txt
└── command_and_env.txt
```

`command_and_env.txt` 必须过滤以下内容：

- NGC API Key；
- Hugging Face token；
- `.credentials` 内容；
- 其他 bearer token、密码和私钥。

推荐的结果元数据字段：

```text
timestamp
slurm_job_id
node_names
git_commit
model_id
model_revision
image_path
image_sha256
sglang_version
cuda_version
driver_version
transfer_backend
pd_ib_device
lustre_path
lustre_stripe_count
lustre_stripe_size
precision
kv_dtype
tp_size
pd_ratio
context_length
trace_id
seed
cache_state
```

## 9. 已确认事实与待确认事项

已确认：

- GAIA 使用 Slurm，默认分区为 `GAIA`；
- 一个标准节点有 8 × H100 80GB、约 2 TiB DRAM；
- 计算节点有 Enroot 3.2.0 和 Pyxis；
- 计算节点有 `/raid` 本地盘与 `/mnt/lustre` 共享 Lustre；
- Lustre 在登录节点不可见；
- 实测节点有 8 个 active 400G compute-fabric 候选 NIC；
- Lustre 路由使用一个 active 200G device `mlx5_1`；
- 裸节点不提供 SGLang/Mooncake Python 包；
- 当前用户尚无已确认的 Lustre 项目目录；
- 可从内部 registry 获取 `dl/dgx/sglang:25.08-py3-devel`；
- 参考的 `sglang-25.11.sqsh` 文件当前不存在。

待确认，确认后应更新本文：

1. 当前用户正式的 Lustre 路径、quota 和 inode 限额；
2. 模型 checkpoint 的实际路径与 revision；
3. `sglang-25.08-py3-devel.sqsh` 导入结果和 SHA-256；
4. 容器内 SGLang、Mooncake、NIXL、CUDA、NCCL 版本；
5. 三节点间 PD transfer 的最佳 IB device/list；
6. Lustre 的推荐 stripe 与实测单节点/三节点带宽；
7. 用户是否有权限获取 Lustre OST 与 IB port telemetry；
8. 是否允许配置 IB QoS/VL，及需要联系的管理员；
9. `/raid` 的清理和持久化策略；
10. GAIA 作业的推荐 CPU/NUMA 绑定参数。

## 10. Git 管理

项目根目录 `/labhome/sezhou/dual-path` 使用 Git 管理。Git 只保存代码、配置模板、文档和小型结果摘要，不保存大文件与运行时数据。

应纳入版本控制：

- `docs/` 中的 Markdown 文档；
- Slurm/Enroot 启动脚本；
- trace 生成器、replay client、指标采集和分析代码；
- 不含凭据的配置样例；
- 实验 manifest 和小型汇总结果。

默认不纳入版本控制：

- `*.sqsh` 容器镜像；
- 模型权重及 Hugging Face cache；
- KV-cache block、segment 和索引；
- 原始 benchmark 日志、profile、core dump；
- `.credentials`、`.env`、token 和私钥；
- 大型 trace 和原始结果。

每次正式实验记录当前代码版本：

```bash
git status --short
git rev-parse HEAD
git diff --stat
```

若工作区非 clean，必须将 `git diff` 随实验元数据归档；推荐正式结果只从已提交 commit 运行。镜像、模型、trace 和数据不依赖 Git 文件内容寻址，应在 manifest 中额外记录 SHA-256 或仓库 revision。

论文 PDF 属于静态研究输入，可以直接纳入 Git；若后续加入大量论文或大于仓库政策限制的二进制文件，再启用 Git LFS。不要在没有配置远端 LFS 的情况下只提交 LFS pointer。

## 11. 安全说明

参考脚本中包含明文 NGC API Key。该凭据已经暴露在普通文本中，应视为泄露：

1. 在 NGC API Keys 页面立即吊销；
2. 生成新 key；
3. 仅写入 `$HOME/.config/enroot/.credentials`；
4. 执行 `chmod 600`；
5. 检查 shell history、作业日志和版本库中是否还有副本；
6. 不在本文或后续脚本中填写真实 token。

内部 GitLab access token 采用相同规则，并确保 `.credentials`、`.env` 和本地 secret 文件被 `.gitignore` 排除。

本文不会记录登录节点 IP、个人凭据或未经当前环境验证的网络拓扑宣传资料。对于会影响结论的平台信息，以作业节点实时采集结果为准。
