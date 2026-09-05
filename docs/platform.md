# GAIA 平台、网络与 Lustre

> 返回 [文档索引](README.md)

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

### 已知故障节点（临时排除）

以下节点在本项目作业中曾出现平台级故障。在管理员确认恢复前，正式多节点
作业一律排除，不要用改 SGLang / Mooncake 参数去掩盖：

| 节点 | 首次相关 Job | 现象 |
|---|---|---|
| `dgx-gaia-30` | `64855` | 已拿到整节点 GPU allocation，CUDA 初始化仍报 `CUDA-capable device(s) is/are busy or unavailable` |
| `dgx-gaia-46` | `64858` | Enroot NVIDIA hook 找不到 `nvidia-container-cli`，Pyxis 无法启动容器 |
| `dgx-gaia-53` | `65566` | 32-agent 完整 rollout 中 Decode 突然停写日志；Router `HostUnreachable` / `No route to host`；Prefill `KVTransferError` 与 bootstrap 300 s 超时 |
| `dgx-gaia-18` | `65658` | 1P2D 1-agent 启动时 Decode-1 上 Mooncake Store `warmup()` `put` 断言失败；日志含 `Failed to get segment descriptor for segment dgx-gaia-16:...`；Prefill/Decode-0 已健康但 Decode-1 未就绪 |
| `dgx-gaia-12` | `66412` | Mooncake warmup 前 `mlx5_0` 连续报告 RDMA port error/inactive，随后 `AddressNotRegistered`，Store `put` 断言失败，Prefill 无法启动 |

统一排除列表：

```text
dgx-gaia-12,dgx-gaia-18,dgx-gaia-30,dgx-gaia-46,dgx-gaia-53
```

正式 benchmark 脚本（`pd_1p1d_benchmark.sbatch` /
`pd_1p2d_benchmark.sbatch` / `pd_2p2d_benchmark.sbatch`）已写入对应
`#SBATCH --exclude=...`。临时覆盖提交时也应显式带上同一列表，例如：

```bash
sbatch --exclude=dgx-gaia-12,dgx-gaia-18,dgx-gaia-30,dgx-gaia-46,dgx-gaia-53 --time=08:00:00 \
  scripts/pd_1p2d_benchmark.sbatch
```

自动化 smoke 若在模型加载前遇到同类节点错误：保存日志、取消剩余 server
step，按上表扩大 exclude，并向平台管理员报告。排除 `dgx-gaia-30/46` 后，
Job `64860` 曾在 `dgx-gaia-[51-54]` 完成 2P2D 验证。

Job `65566` 细节：节点 `dgx-gaia-[51-53]`，Decode-1=`dgx-gaia-53` 约在
`22:17:29` 失联（当时约 11 running / 8 transfer，token usage ≈ 0.34，无
CUDA OOM/segfault/traceback）；客户端仅 2/32 trajectory 完成，其余为空流或
短生成，正式汇总图未生成。Job `65592` 曾只排除 `dgx-gaia-53`，但又被分到
含 `dgx-gaia-30` 的 `dgx-gaia-[29-31]`，因此取消后改为排除全部三节点重提。

Job `65658` 细节：节点 `dgx-gaia-[16-18]`，Decode-1=`dgx-gaia-18` 在 Mooncake
Store warmup 阶段失败（`AssertionError` on `store.put`），约 4.5 分钟后
FAILED；未进入 replay，无正式 metrics。随后将 `dgx-gaia-18` 并入排除列表重提。

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

该命令只说明到特定 server IP 的普通 IP 路由，不能代表 Lustre 的全部 LNet
数据路径。后续 HCA counter 实测确认 Lustre 会使用 `mlx5_1` 和 `mlx5_7`
两条 200 Gb/s storage rail，详见下方 Direct I/O 测试。

DGX H100 官方 GPU/compute-fabric HCA 拓扑为一一对应。正式 SGLang/Mooncake
PD 和 HiCache client 均使用逐 TP rank JSON 映射：

```text
GPU0 mlx5_0   GPU1 mlx5_3   GPU2 mlx5_4   GPU3 mlx5_5
GPU4 mlx5_6   GPU5 mlx5_9   GPU6 mlx5_10  GPU7 mlx5_11
```

对应 `--disaggregation-ib-device`：

```json
{"0":"mlx5_0","1":"mlx5_3","2":"mlx5_4","3":"mlx5_5","4":"mlx5_6","5":"mlx5_9","6":"mlx5_10","7":"mlx5_11"}
```

SGLang 0.5.7 会按当前 `gpu_id` 选择映射值；Mooncake Store 也会按
`storage_config.tp_rank` 选择 `device_name`。不要把 `mlx5_1/7` 加入 PD 映射，
它们属于 Lustre storage fabric。

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

- compute fabric 与 storage fabric 是否具有不同 IB partition/P_Key；
- 是否允许用户读取端口计数器；
- IB VL/QoS 配置是否需要管理员操作。

### Lustre 单节点 Direct I/O 实测

2026-09-05 在 `dgx-gaia-14` 使用
`scripts/smoke/lustre_io_bandwidth.sbatch` 测试。每轮总数据量 128 GiB，
4 MiB block、`O_DIRECT`，并同步采集 HCA 端口计数器；临时文件在结束时删除。

- 8 jobs / 8 stripes：写 10.94 GiB/s，读 10.77 GiB/s；
- 16 jobs / 16 stripes：写 18.24 GiB/s，读 19.35 GiB/s；
- 32 jobs / 32 stripes：写 22.93 GiB/s，读 30.49 GiB/s；
- 64 jobs / 32 stripes：写 **23.31 GiB/s**，读 **32.37 GiB/s**。

单条 200 Gb/s 链路的十进制理论峰值是 25 GB/s，即 23.28 GiB/s。因此写入
已经达到单 rail 线速（计时边界存在约 1% 误差），但不是双 rail 的理论峰值。
64-job 写入流量约 98.4% 经过 `mlx5_7`；读取约 70.3% 经过 `mlx5_1`、29.7%
经过 `mlx5_7`，说明 Lustre/LNet 的读取路径已启用两个 storage rail，但当前
负载分配不均。读取 32.37 GiB/s 是双 200 Gb/s 理论值 46.57 GiB/s 的约 69.5%。

此前 `ip route get 192.1.3.230` 只描述到一个 server IP 的普通 IP 路由，不能
代表 Lustre 全部 LNet 数据路径；HCA 计数器是本次流量归属判断依据。

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

## 尚待确认的平台事项

1. Lustre quota、inode 限额、stripe 和单/多节点带宽；
2. 64K context 的容器兼容性和资源余量；
3. 多节点 PD transfer 的最佳 IB device/list；
4. OST 与 IB port telemetry 权限；
5. IB QoS/VL 配置权限与管理员流程；
6. `/raid` 清理策略以及推荐 CPU/NUMA 绑定。
