# NCCL 路由控制（`DUAL_PATH_NCCL_FORCE_NIC`）

控制 TP 内 NCCL 通信走 NVLink 还是全部走 CX8 网卡。PD 测试脚本和 `nic-test/` 干扰测试都默认 `1`，也就是全部走网卡。

## 开关

| 变量 | 取值 | 默认 | 作用 |
|---|---|---|---|
| `DUAL_PATH_NCCL_FORCE_NIC` | `0` / `1` | `1` | `1`：导出 `NCCL_P2P_DISABLE=1` 和 `NCCL_SHM_DISABLE=1`，TP 内所有连接都走 NET/IB。`0`：不改动这两个变量，NCCL 自己选路 |

生效位置：

- **PD 测试**：`scripts/tests/runtime/pd_benchmark_lib.sh` 的 `dualpath_export_common_env`。`scripts/pd_benchmark.sh` 把这个变量列入运行时覆盖列表，启动命令里设置的值会覆盖 allocation 配置中的值。运行元数据会记录 `dual_path_nccl_force_nic`、`nccl_p2p_disable` 和 `nccl_shm_disable`。
- **干扰测试**：`nic-test/run_allreduce_interference.sh`，只作用于 TP4 探针进程。

```bash
# PD 实验：恢复 NVLink
DUAL_PATH_NCCL_FORCE_NIC=0 scripts/pd_benchmark.sh prefill ...
# 干扰测试：TP4 全部走网卡（默认），并打印 NCCL 选路日志
NCCL_DEBUG=INFO nic-test/run_allreduce_interference.sh
```

## 为什么两个变量都要关

g5/g6 上 NCCL 的机内传输有三种，优先级从高到低：

| 传输 | 用于 | 关闭方式 |
|---|---|---|
| P2P（`P2P/CUMEM`、`P2P/IPC`） | 同一对 GPU（GPU0↔GPU1、GPU2↔GPU3）之间的 NVLink | `NCCL_P2P_DISABLE=1`，与 `NCCL_P2P_LEVEL=LOC` 等价 |
| SHM | 经 host 共享内存中转 | `NCCL_SHM_DISABLE=1` |
| NET（`NET/IB…/GDRDMA`） | 网卡 RDMA，GPUDirect 直接读写显存 | 由 `NCCL_IB_HCA`、`NCCL_NET_GDR_LEVEL=PIX` 控制 |

只关 P2P 时，同节点的 GPU 会退到 SHM，仍然不走网卡。两个都关掉，才只剩 NET。

默认配置下，跨 GPU 对（GPU0/1↔GPU2/3）本来就走 NET/IB：NCCL 认为经 PIX 网卡的路径优于经 CPU 的 SYS 路径（见 [vl-traffic-isolation.md](vl-traffic-isolation.md)）。所以这个开关实际只改变同一对 GPU 之间的路径。

SGLang 在 TP4 下不启用 custom all-reduce，所有 TP 通信都经过 NCCL。如果换成其他 TP 规模或其他版本，需要加 `--disable-custom-all-reduce`，否则 custom all-reduce 会绕过 NCCL 直接走 NVLink。

## 开启后的路径

以 TP4 的环 0 → 1 → 2 → 3 → 0 为例：

| 连接 | `=0` | `=1` |
|---|---|---|
| GPU0 → GPU1 | NVLink | 在 mlx5_0 内部回环：进出网卡的 PCIe，不经过交换机 |
| GPU1 → GPU2 | mlx5_0 → 交换机 → mlx5_4 | 不变 |
| GPU2 → GPU3 | NVLink | 在 mlx5_4 内部回环 |
| GPU3 → GPU0 | mlx5_4 → 交换机 → mlx5_0 | 不变 |

需要注意：

- 回环流量不上交换机，所以交换机侧的 VL 仲裁管不到它。它是否经过 HCA 端口的 VL 仲裁，还没有验证。
- 回环流量会和 NIC 回环 H2D、P→D 和 D→P RDMA 争用同一块 HCA，以及 GPU 的 PCIe 带宽（网卡写 GPU 约 18 GB/s，读 GPU 约 10～11 GB/s）。

## 验证（Job 631，2026-09-30）

用 `DUAL_PATH_NCCL_FORCE_NIC=1 NCCL_DEBUG=INFO` 跑 `nic-test` 的无负载探针，结果在 `nic-test/results/631-tp4-net-only-baseline/`。

- **NCCL 日志**：两个节点共 64 条选路日志（每个 channel 的收、发各一条），全部是 `via NET/IBext_v11/<0|1>/GDRDMA`，没有出现 `P2P` 或 `SHM`。日志中也能看到 `NCCL_P2P_DISABLE set by environment to 1` 和 `NCCL_SHM_DISABLE set by environment to 1`。
- **网卡计数器**：每块 HCA 的接收量正好是发送量的 2 倍，g5 为发送 4.3 GB/s、接收 8.6 GB/s。回环流量只计入 `port_rcv_data`，不计入 `port_xmit_data`，所以多出来的一份就是原来走 NVLink 的流量。`=0` 时收发相等，g5 约 6.7 GB/s。
- **TP4 通信耗时**（每次迭代，无负载）：

  | 模式 | `=0`（Job 630） | `=1`（Job 631） | 变化 |
  |---|---|---|---|
  | Prefill：129 次 9.8 MiB all-reduce + 1 次 all-gather，eager | 124 ms | 193 ms | +55% |
  | Decode：129 次 60 KiB all-reduce + 1 次 all-gather，CUDA graph | 7.7 ms | 10.0 ms | +30% |

检查任何一次运行时，都可以设置 `NCCL_DEBUG=INFO`，在日志里搜 `via`。开启后应该只出现 `NET/IB…/GDRDMA`。

## 相关文档

- [nic-test/README.md](../nic-test/README.md)：KV 流量对 TP4 all-reduce 的干扰测试。当前结果是在 `=0` 下测的。
- [vl-traffic-isolation.md](vl-traffic-isolation.md)：SL/VL 隔离方案。
- [nic-h2d.md](nic-h2d.md)：`DUAL_PATH_CUDAMEMCPY=nic` 的 HiCache H2D/D2H 走网卡方案。
