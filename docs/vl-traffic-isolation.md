# InfiniBand VL 流量隔离（pza100 方案）

> 返回 [文档索引](README.md)

本文把 DualPath 论文 §5.1 / 附录 A.1 的 VL 隔离落到 g5/g6。平台数据为
2026-09-29 在 g5 上以普通用户身份采集（verbs、sysfs、Mooncake 二进制）；需要
root 的 SMP/SA 查询尚未执行，对应结论标为“待探查”。

## 1. 论文做法

- **CNIC-centric**：进出 GPU 的所有数据（含本机 H2D/D2H）都经过 GPU 配对的
  compute NIC，以 GPUDirect RDMA 完成，使 NIC 成为 GPU PCIe 流量唯一的 QoS
  仲裁点。原因是 GPU 不支持 PCIe QoS，copy engine / GDS 不受网络 QoS 控制。
- **VL 分级**：模型执行通信（EP AllToAll、TP/CP 集合通信）走高优先级 VL，
  KV-Cache 等其他流量走低优先级 VL。所有交换机和 NIC 的 VL 仲裁器用 WRR，高
  优先级约占 99%，低优先级只防饿死。
- **附录 A.1 参数**：

```text
qos_max_vls     4
qos_high_limit  240
qos_vlarb_high  0:192,1:192,2:0,3:192
qos_vlarb_low   0:192,1:192,2:64,3:192
```

VL2 不在 high 表中，只能在 low 表获得机会。`VLHighLimit=240` 以 4 KiB 为单位，
即连续发送约 960 KiB 高优先级数据后才给 low 表一次机会；VL2 权重 64 × 64 B =
4 KiB，因此链路饱和时 VL2 至多约 4 / 964 ≈ 0.4%。

## 2. 平台实测

| 设备 | 型号 / FW | 链路 | PCIe | NUMA | 用途 |
|---|---|---|---|---|---|
| `mlx5_0` → `ibs3` | CX8 MT4131 / 40.48.1000 | IB 4X XDR 800G，LID 14 | Gen5 x16 | 0 | GPU0/1 的 CNIC（PIX）；IPoIB `192.168.10.5` |
| `mlx5_4` → `ibp197s0` | CX8 MT4131 | IB 4X XDR 800G，LID 15 | Gen5 x16 | 2 | GPU2/3 的 CNIC（PIX） |
| `mlx5_8` / `mlx5_9` | CX8 VF MT4126 | 共享 `mlx5_0` / `mlx5_4` 物理口 | — | — | SR-IOV VF（Twin VM 用），LID 22 等 |
| `mlx5_1` → `ibs4` | CX7 MT4129 / 28.48.1000 | IB 4X NDR 400G，LID 20 | Gen5 x16 | 1 | IPoIB `192.168.11.5`；g6 的 `/data` NFS 走这里 |
| `mlx5_5` → `ibp134s0` | CX7 MT4129 | IB 4X NDR 400G，LID 23 | Gen5 x16 | 3 | 未使用 |
| `mlx5_2/3/6/7` | BlueField-3 MT41692 | 100GbE | — | — | 以太网，与本方案无关 |
| GPU0–3 | A100-PCIe-40GB | GPU0↔1、GPU2↔3 NV12 | **Gen4 x16** | 0 / 2 | 跨对无 NVLink，拓扑为 SYS |

要点：

1. **硬件 VL 能力与论文一致**。所有 IB 端口 `max_vl_num = 4`（VLCap = VL0–3），
   `active_mtu = 4096`。论文的 `qos_max_vls 4` 可以原样使用，不存在 VL 数不足的问题。
2. **两个独立 IB 子网**。CX8 子网的 SM LID = 1，CX7 子网的 SM LID = 7，两边端口的
   `sm_sl` 都是 0。同一子网内所有端口只能指向同一个 master SM，SM LID 不同即说明是
   两个子网；IPoIB 也分属 `192.168.10.0/24` 和 `192.168.11.0/24`。
   2026-09-29 root 探查（`results/ib-topo-g{5,6}-20260929-*`）：
   - **CX7 子网**是共享存储网。master SM 是主机 `h3 mlx5_0` 上的 SM（LID 7，
     priority 15），不是 y2。交换机为 `ndr-switch-01`（MQM9700，Quantum-2 NDR）、
     `spine01/02` 和 `switch-5ead94`（MQM8700 HDR），以及 `storage-leaf01`。g5 的
     `mlx5_1` 接在 `ndr-switch-01` 的 12 口，g6 的 `mlx5_1` 接在 9 口。同一子网上还有
     h1–h4、h6、y2 `mlx5_1`、gpuserver1/2、k8s-worker1 等，g5/g6 的 `mlx5_5` 也在其中。
   - **CX8 子网**（`results/ib-cx8-diag-g5-20260929-014636`，SA NodeRecord）只有一台
     Quantum-X800 **Q3400-RA**（`nvos-011200`，4 颗 ASIC U1–U4，各 149 口，另有 4 个
     SHARP Aggregation Node）。端点只有 g5、g6 各两块 CX8（LID 14–17），它们的 4 个
     VF（LID 20–23），以及 **`y2 smi0`（LID 1，即 master SM）**。与 CX7 子网没有
     任何共同的节点 GUID，两个子网在物理上使用不同的交换机。
3. **CX8 运行在多平面模式**：`mlxconfig` 中 `NUM_OF_PLANES_P1 = 4`，SA 记录中每块
   CX8 的 `num_ports = 4`，分别对应 Q3400 的 4 颗 ASIC。聚合设备 `mlx5_0` / `mlx5_4`
   只提供 GSI（SA、PMA 可用），不提供 SMI。因此 `sminfo`、`smpquery`、
   `ibnetdiscover` 会返回 ENODEV，也没有 `issm0/issm4`。要发 SMP，需要先建立
   SMI 子设备（`rdma dev add smi0 type SMI parent mlx5_0`，y2 上的 `smi0` 就是这种
   设备）。VL 相关的 SL2VL / VLArb 表按平面端口分别存在，SM 必须对 4 个平面分别
   下发。
4. **CX8 子网的 SM 在 y2 上，子网只有我们在用**。g5 本机没有运行 OpenSM
   （`/etc/opensm` 为空）。y2（`192.168.10.32`）的 `https://…/ufmRest/` 返回 302
   （登录跳转），说明 y2 跑着 UFM，而 CX8 子网的 master SM 正是 `y2 smi0`。
   CX8 子网上除 g5、g6、y2 外只有 4 个 VF。`ibs3` ARP 表中的 `192.168.10.21–24`
   最可能是这些 VF（Twin VM）的 IPoIB 地址，而不是其他物理主机。改 SM 的 SL2VL /
   VLArb 只会影响 g5、g6、y2 和这些 VF。
5. **存储已经在 CX7 上，计算在 CX8 上**。2026-09-29 g5 的 `/proc/net/tcp` 中唯一的
   NFS 连接是 `192.168.11.6 → 192.168.11.5:2049`，即 g6 经 `ibs4`（CX7）挂载 `/data`；
   `/etc/exports` 同时导出给两个网段。`mlx5_1` 自开机以来发出约 542 GB，与 `ibs4`
   计数一致。9 月 17–18 日文档中“g6 经 `ibs3` 挂载”的记录已过时。CX8 上承载 NCCL
   （`NCCL_IB_HCA=mlx5_0,mlx5_4`）、Mooncake PD 传输（GPU0/1→`mlx5_0`、
   GPU2/3→`mlx5_4`）以及 `ibs3` 上的 IPoIB（Gloo、Harbor 等）。论文要求的计算网
   与存储网分离已经满足。
6. **应用侧都能打 SL，不需要改代码**：
   - NCCL：`NCCL_IB_SL`
   - Mooncake Transfer Engine 0.3.13.post1：`MC_IB_SL`（`engine.so` 中确认有该变量和
     `ib_service_level` 字段，还有 `MC_IB_TC`、`MC_PKEY_INDEX`）
   - perftest：`--sl`
   - 自研 verbs 代码：在 `ibv_modify_qp(RTR)` 时设置 `ah_attr.sl`
7. 当前 OperVLs、SL2VL、VLArb 表仍**待探查**：CX8 需要先建 SMI 子设备（第 3 条），
   然后以 root 运行 `CA_PORTS="smi0 smi1 mlx5_1 mlx5_5" scripts/infra/ib_qos_probe.sh`。

## 3. 与论文的关键差异：瓶颈位置倒置

| 环节 | 论文 DGX H100 | pza100 |
|---|---:|---:|
| 每个 CNIC 的线路速率 | 400 Gb/s | 800 Gb/s |
| CNIC 主机侧 PCIe | Gen5 x16 ≈ 500 Gb/s | Gen5 x16 ≈ 500 Gb/s |
| 每个 CNIC 服务的 GPU | 1 | 2 |
| GPU PCIe | Gen5 x16 ≈ 500 Gb/s | Gen4 x16 ≈ 252 Gb/s |
| 最窄环节 | IB 线路 | GPU PCIe，其次是 NIC PCIe |

VL 仲裁发生在 IB **输出端口**：只有当报文在某个端口排队、需要决定下一个发哪条 VL
时，权重才起作用。论文平台上线路是最窄环节，拥塞就发生在 VL 仲裁器所在的位置，
所以隔离效果直接。pza100 上 800G 线路比 NIC 的 PCIe（约 500G）和 GPU 的 PCIe
（约 252G）都宽，典型情况是：

- **发送端**：HCA 受 PCIe DMA 读速率限制，出口端口不排队，VL 仲裁可能根本没有
  发生的机会。高低优先级之间的比例取决于 HCA 内部的 SQ 调度是否识别 VL，IB 规范
  对此没有要求。
- **接收端**：数据写入 GPU 受 Gen4 x16 限制，排队发生在 HCA 接收缓冲。HCA 按 VL
  返还 credit，交换机出口会因此出现拥塞，但各 VL 能否按权重获得 credit，取决于
  HCA 按什么顺序把各 VL 的缓冲排到 PCIe。这一点同样没有规范保证。
- **节点内**：SGLang 在 TP4 下会禁用 custom all-reduce（“not supported on more than
  two PCIe-only GPUs”），TP 通信全部交给 NCCL。实测（§4.4）NVLink 对内走
  `P2P/CUMEM`，跨对（GPU0↔GPU2 等）走 `NET/IB` GDRDMA，也就是经 `mlx5_0` →
  交换机 → `mlx5_4` 回到本机，**已经在网卡上，可以用 `NCCL_IB_SL` 纳入 VL**。
  真正在 VL 体系之外的是 SGLang HiCache 的 H2D/D2H：它用 CUDA copy engine /
  kernel 直接读 pinned host memory，不经过网卡。把它改走网卡的开关与实测见
  [nic-h2d.md](nic-h2d.md)。

结论：**VL 方案在本平台可以完整配置出来，但只配 VL 并不能保证获得论文里那种隔离
效果**。方案必须同时处理下面两件事，并用实验量化效果：

1. 把 GPU 的进出流量尽量收拢到 NIC 上（即论文的 CNIC-centric 做法）；
2. 必要时让线路重新成为瓶颈（例如把链路降速到 400G），或者对 KV 流量做速率整形。

## 4. 方案

### 4.1 流量分类与 SL/VL 映射

| 类别 | 具体流量 | 端口 | 标记方式 | SL | VL | 仲裁 |
|---|---|---|---|---:|---:|---|
| 默认 / 控制 | IPoIB（Gloo、NCCL bootstrap、Harbor、SSH）、y2、Twin VM 的 SR-IOV VF（`.21–.24`）、SA/CM | `ibs3` 等 | SM 组播组默认 SL | 0 | 0 | high + low |
| 模型执行 | 跨节点 NCCL；节点内跨对 NCCL（默认即走 NET/IB，§4.4） | `mlx5_0/4` | `NCCL_IB_SL=1` | 1 | 1 | high + low |
| KV-Cache | Mooncake PD 传输、DE→PE 层级传输、CNIC 辅助 H2D/D2H | `mlx5_0/4` | `MC_IB_SL=2`；自研 QP 设 `ah_attr.sl=2` | 2 | 2 | **low only** |
| 预留 | — | — | — | 3 | 3 | high + low |
| 存储 | NFS / KV L3 读写 | CX7 `ibs4`（独立子网，已是现状） | — | — | — | — |
| SM | SMP MAD | 全部 | — | — | 15 | — |

保持 SL0 → VL0 并留在 high 表中，是为了让 y2、VF 和 IPoIB 的默认流量不受影响。
SL4–15 全部映射到 VL0，避免出现指向非运行 VL 的条目（这类报文会被直接丢弃）。

SM 侧配置见 `scripts/infra/opensm-qos-dualpath.conf`，仲裁参数与论文相同，只多了
一行明确的 `qos_sl2vl`。

### 4.2 SM 侧

一个子网只能有一个 master SM，**不要在 g5/g6 上另起 OpenSM**。步骤如下：

1. CX8 子网的 master SM 是 `y2 smi0`（LID 1），由 y2 上的 UFM 管理。配置文件通常
   在 `/opt/ufm/files/conf/opensm/opensm.conf`（以 UFM 版本为准），合并片段后按 UFM
   的方式重启 SM。先在 y2 上确认该 OpenSM / UFM 版本支持多平面，并且 QoS 表会对
   每个平面端口分别下发。
2. 修改前备份原文件，并记录探查得到的当前 OperVLs / SL2VL / VLArb，作为回滚依据。
3. 如果当前 OperVLs 已经是 VL0–3，启用 QoS 只会改表内容。如果不是，修改 OperVLs
   可能触发链路重新协商（短暂断链），需要安排维护窗口。
4. 生效后在 g5/g6 两块 CX8 的全部 4 个平面端口**以及**对端 Q3400 各 ASIC 的端口上
   复查：`smpquery sl2vl` / `vlarb` 应与片段一致（经 SMI 子设备重新运行探查脚本即可）。
5. 该子网只包含 g5、g6、y2 和 Twin VM 的 VF，但 SM 和 UFM 由朋志维护，**改动前仍需
   他确认**。

CX7 子网（SM 在 `h3`，LID 7）是多台主机共用的存储网，不在本方案的修改范围内。

### 4.3 存储流量

g6 的 `/data` 已经经 CX7（`192.168.11.5`）挂载，相当于论文里的 SNIC，VL 方案
不需要为存储做任何改动。CX7 子网里只有存储流量，也不需要 QoS。

仍需注意：

- g5 上的 NFS 连接只有一条 TCP，说明仍是 `nconnect=1`，
  [pza100 环境](pza100.md) 中记录的单连接瓶颈依然存在；改 `nconnect` 是独立变量，
  对比实验中要单独控制。
- CX7 位于 NUMA 1/3，而 GPU 在 NUMA 0/2，Host staging buffer 的 NUMA 绑定需要
  一起评估。
- g5 本机读 L3 直接走本地 NVMe，不经过任何网卡；只有 g6 的读写经过 CX7。

### 4.4 节点内流量（CNIC-centric）

| 流量 | 现状 | 让它进入 VL 体系的办法 |
|---|---|---|
| NVLink 对内 NCCL（GPU0↔1、GPU2↔3） | NVLink | 不占 PCIe，本来就不冲突 |
| 跨对 NCCL（GPU0↔2 等） | 默认即走 `NET/IB` GDRDMA（`mlx5_0`→交换机→`mlx5_4`） | 设置 `NCCL_IB_SL=1` 即可纳入 VL；也可以改用 TP2（NVLink 对）× DP2，让 NCCL 完全不占网卡，但 32B 模型在 2 × 40 GB 上显存很紧 |
| HiCache H2D/D2H | `--hicache-io-backend direct`：`cudaMemcpyAsync`，走拷贝引擎 | 开关 `DUAL_PATH_CUDAMEMCPY=nic`：经配对 NIC 的 RDMA 回环，SL 取 `DUAL_PATH_NIC_SL=2`；设计与 mini bench 见 [nic-h2d.md](nic-h2d.md) |

2026-09-29 在 g5 上用 `scripts/smoke/nccl_tp4_path_probe.py` 实测（镜像
`sglang:26.01-py3`，NCCL 2.29.2，环境变量与 PD 脚本相同）。日志位于
`mydata/test/nccl-path-probe/default.log`：

```text
0[0] -> 1[1] via P2P/CUMEM/read           # NVLink 对内
0[0] -> 2[2] [send] via NET/IBext_v11/0/GDRDMA
2[2] -> 0[0] [send] via NET/IBext_v11/1/GDRDMA
allreduce 8 KiB 45.5 us | 1 MiB 172.4 us, busbw 9.1 GB/s | 64 MiB busbw 14.5 GB/s
```

跨节点 TP8 作业 401 的日志（NCCL 2.29.7）中，同节点跨对的连接同样是 `NET/IB`。
`nvidia-smi topo -p2p` 显示跨对 PCIe P2P 读写均为 OK，是 NCCL 在拓扑搜索中认为经
PIX 网卡的路径优于 SYS。由于两块 CX8 的 LID 不同，这些报文会经过交换机，因此
交换机端口和 HCA 端口的 VL 仲裁都会作用到它们。

2026-09-29 的 mini bench（[nic-h2d.md](nic-h2d.md)）观测到：同网卡回环时
`port_rcv_data` 随数据量增长，而 `port_xmit_data` 不变，即数据在 HCA 内部回环，
没有经过交换机，交换机端口的 VL 仲裁对它不起作用。

需要实验确认：HCA 内部 loopback（QP 发给同一 LID）是否经过端口 VL 仲裁器。如果
不经过，就需要让 H2D 走另一块 NIC 绕交换机一圈，或者接受这部分不受 VL 保护。

### 4.5 应用侧环境变量

在 `dualpath_export_common_env`（`scripts/tests/runtime/pd_benchmark_lib.sh`）中
按开关注入：

```bash
if [[ ${DUALPATH_IB_QOS:-0} == 1 ]]; then
  export NCCL_IB_SL=${NCCL_IB_SL:-1}
  export MC_IB_SL=${MC_IB_SL:-2}
fi
```

默认关闭，以免在 SM 配置确认前改变现有实验。开启 QoS 的 run 需要在结果元数据中
记录 `DUALPATH_IB_QOS`、`NCCL_IB_SL`、`MC_IB_SL` 和 SM 配置版本。

## 5. 分阶段实施

| 阶段 | 内容 | 需要 | 风险 |
|---|---|---|---|
| 0 探查 | 两节点分别 `sudo bash scripts/infra/ib_qos_probe.sh`；结果写入 `results/ib-qos-probe/` | root | 只读 |
| 1 打标 | 确认当前 SL2VL 中 SL1/2 映射到运行中的 VL 后，开启 `DUALPATH_IB_QOS=1`，用 `perfquery -X/-S`（按 SL 的收发计数）确认 NCCL 和 Mooncake 分别计入 SL1 和 SL2 | 无 SM 改动 | 若 SL1/2 指向非运行 VL，报文会被丢弃，必须先看阶段 0 的结果 |
| 2 存储分离 | 已是现状：g6 经 `ibs4`（CX7）挂载（§4.3） | — | — |
| 3 SM QoS | 合并 `opensm-qos-dualpath.conf`，复查全部相关端口的表 | SM 管理权限、与朋志协调 | 共享子网；OperVLs 变化可能导致断链 |
| 4 节点内 | 在 §4.4 的 A/B 中选一个；实现或评估 CNIC 辅助的 H2D | 代码 | 性能回退需要量化 |
| 5 验证 | 按 §6 的实验矩阵 | g6 恢复可调度 | — |

回滚顺序：先取消 `DUALPATH_IB_QOS`（立即生效，所有流量回到 SL0/VL0，而 VL0 在
high 表里），再按需恢复 SM 的原配置文件。

## 6. 验证实验

所有 perftest 在 g5 → g6 之间运行，OOB 走 `ibs3`；每个实验分别在 QoS 关闭和开启
两种状态下各跑一次。

| 编号 | 场景 | 高优先级探针（SL1） | 背景负载（SL2） | 回答的问题 |
|---|---|---|---|---|
| E0 | 基线 | `ib_write_lat -s 4096`、单独运行的 `ib_write_bw` | 无 | 单 VL 带宽、4 VL 缓冲切分后是否掉速 |
| E1 | 同 NIC 发送端，host 内存 | `ib_write_lat --sl 1` + `ib_write_bw --sl 1` | `ib_write_bw --sl 2 -q 16 -s 1M` | NIC PCIe 受限时，HCA 发送调度是否识别 VL |
| E1b | 同 E1，链路降到 NDR 400G | 同上 | 同上 | 线路重新成为瓶颈后，隔离是否达到约 99% |
| E2 | incast：g5 的 `mlx5_0` 和 `mlx5_4` → g6 的 `mlx5_0` | 探针走 `mlx5_0` | 背景走 `mlx5_4` | 交换机出口仲裁和接收端 credit 行为 |
| E3 | 写入 GPU（`--use_cuda`，需 perftest 带 CUDA 支持） | 写入 GPU1 或 GPU0 | 写入 GPU0 | 同 NIC 不同 GPU 与同一 GPU 两种情况；后者是 Gen4 瓶颈 |
| E4 | nccl-tests `all_reduce_perf` / `alltoall_perf` 跨节点，8 KB–64 MB | `NCCL_IB_SL=1` | E1 的背景负载 | 小消息延迟和 busbw 退化 |
| E5 | 节点内 | §4.4 的 A/B 方案 | copy engine H2D vs RDMA loopback H2D | 节点内流量是否真正得到保护 |
| E6 | 端到端 1P1D | 正式 replay | 真实 KV | TPOT / TTST 是否不受 L3 和 PD 流量影响，JCT 是否回退 |

示例（E1，两端各开两组端口号）：

```bash
# g6
ib_write_bw  -d mlx5_0 --sl 2 -q 16 -s 1048576 -D 30 --report_gbits -p 18515 &
ib_write_lat -d mlx5_0 --sl 1 -s 4096 -n 200000 -p 18516
# g5
ib_write_bw  -d mlx5_0 --sl 2 -q 16 -s 1048576 -D 30 --report_gbits -p 18515 192.168.10.6 &
sleep 3
ib_write_lat -d mlx5_0 --sl 1 -s 4096 -n 200000 -p 18516 192.168.10.6
```

判定标准（适用于线路受限的 E1b / E2）：QoS 开启后，SL1 的带宽应接近 E0 的单独
运行值，p99 延迟增幅不超过 10%；SL2 在 SL1 饱和时应降到约 0.4%，在 SL1 空闲时应
回到满速。E1 和 E3 不预设通过标准，只用来测量 PCIe 受限时隔离还能保留多少。
每次实验前后都要采集 `perfquery -X/-S` 和 `--vlxmitcounters`，确认流量确实落在
预期的 SL/VL 上。

## 7. 风险与待确认

- y2 上 UFM / OpenSM 的版本、修改权限，以及它对多平面 QoS 的支持。
- 当前各平面的 OperVLs、SL2VL、VLArb 实际值；SHARP（Q3400 上有 4 个 Aggregation
  Node）或自适应路由是否已经占用 SL≥1。
- NCCL / Mooncake 的 QP 在 4 个平面上如何分布，以及每个平面各自的拥塞情况。
- §3 中两个没有规范保证的行为：HCA 在 PCIe 受限时的发送调度，以及接收端各 VL
  的排空顺序。E1、E2、E3 就是为了测这两点。
- HCA 内部 loopback 是否经过 VL 仲裁。
- 如果 E1/E3 显示隔离无效：考虑把链路降到 400G（`mlxlink --speeds`，可逆，
  影响交换机上的对应端口），或对 KV QP 做速率整形（需确认 IB RC QP 上的
  `ibv_modify_qp_rate_limit` 支持情况）。
- g6 当前在 Slurm 中是 `idle*`（slurmd 无响应），阶段 5 需要等它恢复。
