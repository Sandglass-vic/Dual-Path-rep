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
| 模型执行 | TP 内 NCCL（`DUAL_PATH_NCCL_FORCE_NIC=1` 时全部走 NET/IB，§4.4） | `mlx5_0/4` | `DUAL_PATH_VL_P_NCCL=1`、`DUAL_PATH_VL_D_NCCL=1` | 1 | 1 | high + low |
| KV-Cache | P→D、D→P、NIC H2D/D2H | `mlx5_0/4` | `DUAL_PATH_VL_P2D=2`、`DUAL_PATH_VL_D2P=2`（§4.5） | 2 | 2 | **low only** |
| 预留 | — | — | — | 3 | 3 | high + low |
| 存储 | NFS/RDMA（KV L3 读写） | CX7 `mlx5_1`（独立子网，已是现状） | — | — | — | — |
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

g6 的 KV cache 自 2026-09-29 起走 NFS/RDMA（`/mnt/nfsrdma`，
`192.168.11.5:/data`，`vers=3,proto=rdma,port=20049`，见 [pza100 环境](pza100.md)），
仍在 CX7 上，相当于论文里的 SNIC。VL 方案不需要为存储做任何改动。

仍需注意：

- CX7 子网由多台主机共用（SM 在 `h3`），其他主机的流量不受本方案控制。
- L3 读带宽由 `DUALPATH_L3_READ_BPS` 软件限速决定，存储侧已经相当于做了整形。
- CX7 位于 NUMA 1/3，而 GPU 在 NUMA 0/2，Host staging buffer 的 NUMA 绑定需要
  一起评估。
- g5 本机读 L3 直接走本地 NVMe，不经过任何网卡；只有 g6 的读写经过 CX7。

### 4.4 节点内流量（CNIC-centric）

| 流量 | 现状 | 让它进入 VL 体系的办法 |
|---|---|---|
| 同一对 GPU 的 NCCL（GPU0↔1、GPU2↔3） | PD 脚本默认 `DUAL_PATH_NCCL_FORCE_NIC=1`：在 HCA 内部回环，不经过交换机（[nccl-control.md](nccl-control.md)）；`=0` 时走 NVLink | `DUAL_PATH_VL_P_NCCL` / `DUAL_PATH_VL_D_NCCL` 能设上 SL，但回环是否经过 HCA 端口的 VL 仲裁未验证 |
| 跨对 NCCL（GPU0↔2 等） | 两种模式都走 `NET/IB` GDRDMA（`mlx5_0`→交换机→`mlx5_4`） | `DUAL_PATH_VL_P_NCCL` / `DUAL_PATH_VL_D_NCCL`；也可以改用 TP2（NVLink 对）× DP2，让 NCCL 完全不占网卡，但 32B 模型在 2 × 40 GB 上显存很紧 |
| HiCache H2D/D2H | 默认 `hicache`：`cudaMemcpyAsync`，走拷贝引擎 | `DUAL_PATH_CUDAMEMCPY=nic`：经配对 NIC 的 RDMA 回环，SL 与同进程的 PD 传输相同（§4.5）；见 [nic-h2d.md](nic-h2d.md) |

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

每条路径一个开关，取值 0–3。按 `opensm-qos-dualpath.conf` 配置 SM 后 SL0–3 →
VL0–3，取值同时是 SL 和 VL 编号。

当前 SM 的实际配置（探查 `results/ib-qos-probe/g5-20260929-015024` 与
`g5-20260930-235715` 一致，CX8 全部平面端口和 Q3400 端口相同）：OperVLs 为 VL0–1；
SL2VL 为 SL0 → VL0、SL1–7 → VL1、SL8–15 → VL15（会被丢弃）；low 表 VL0:0xA0、
VL1:0x70，high 表为空，VLHighLimit 为 0。也就是说，现在 SL1、2、3 都落在 VL1，
和 VL0 按 160:112 做加权轮转，没有严格优先级。不改 SM 就能做的隔离只有“两类流量
分到 VL0 和 VL1、各用各的缓冲和 credit”。

| 开关 | 流量 | 数据由谁发出 | 生效方式 |
|---|---|---|---|
| `DUAL_PATH_VL_P_NCCL` | Prefill 的 TP 内 NCCL（g5 节点内） | Prefill 进程 | `pd_prefill.sh` 导出 `NCCL_IB_SL` |
| `DUAL_PATH_VL_D_NCCL` | Decode 的 TP 内 NCCL（g6 节点内） | Decode 进程 | `pd_decode.sh` 导出 `NCCL_IB_SL` |
| `DUAL_PATH_VL_P2D` | P→D RDMA 写 | Prefill 进程 | `pd_prefill.sh` 导出 `MC_IB_SL` |
| `DUAL_PATH_VL_D2P` | D→P RDMA 写（Decode path） | Decode 进程 | `pd_decode.sh` 导出 `MC_IB_SL` |
| `DUAL_PATH_VL_P_NIC_H2D` | Prefill 的 NIC H2D/D2H | Prefill 进程 | 必须等于 `P2D`，不设时取 `P2D` 的值 |
| `DUAL_PATH_VL_D_NIC_H2D` | Decode 的 NIC H2D/D2H | Decode 进程 | 必须等于 `D2P`，不设时取 `D2P` 的值 |

规则：

- 六个都不设时不导出任何变量，所有 QP 都在 SL0，与之前的实验完全相同。
- 只要设了其中一个，就进入 QoS 模式：没设的 `P_NCCL`、`D_NCCL`、`P2D`、`D2P` 取 0，
  不会从别的开关继承。
- 旧的 `DUAL_PATH_VL_NCCL` 已删除，设置它会直接报错。TP4 的 NCCL 只在节点内，
  Prefill 的只在 g5、Decode 的只在 g6，所以按角色分开设置没有歧义。
- 只接受 0–3。SL 一旦映射到没有启用的 VL，报文会被直接丢弃，表现为传输卡死。
- 校验在 `dualpath_export_common_env` 中完成，配置错误时服务器不会启动。
  导出由 `pd_network_config.sh` 的 `dualpath_apply_ib_vl` 按角色完成，启动日志里
  会打印一行 `IB VL: role=... NCCL_IB_SL=... MC_IB_SL=...`。
- 结果元数据的 `dual_path_vl` 字段记录六个值，未启用时为 `null`。

为什么 RDMA 写能按方向分开：RDMA WRITE 的数据包使用发起方 QP 的 SL。Mooncake 的
被动端用自己进程的配置建立 QP，反方向发送时也复用这个 endpoint。因此 P→D 的 SL
由 Prefill 进程决定，D→P 的由 Decode 进程决定，对端 ACK 用的是对端的 SL。

为什么 NIC H2D 暂时不能单独设置：Mooncake 0.3.13.post1 通过 `std::call_once` 在每个
进程里只读一次 `MC_IB_SL`，建立连接（进入 RTR）时写入 QP。同一进程里的 PD 引擎和
NIC H2D 引擎只能拿到同一个 SL。镜像里没有重编 Mooncake 所需的 cmake 和 glog。
以后如果要分开，保留的方案是一个 `LD_PRELOAD` 垫片，拦截 Mooncake `engine.so` 对
`ibv_modify_qp` 的 RTR 调用，对本卡回环 QP（`dlid` 等于本端口 LID）改写 SL。

当前 SM（2 条 VL）下的取值：NCCL 留在 VL0，KV 放到 VL1。两类流量各用各的缓冲和
credit，D→P 被 GPU PCIe 卡住时不会再堵住 NCCL；但 160:112 的加权轮转没有优先级。

```bash
DUAL_PATH_VL_P_NCCL=0 DUAL_PATH_VL_D_NCCL=0 DUAL_PATH_VL_P2D=1 DUAL_PATH_VL_D2P=1 \
  scripts/pd_benchmark.sh ...
```

按 `opensm-qos-dualpath.conf` 启用 4 条 VL 后的取值（与论文的分类一致）：

```bash
DUAL_PATH_VL_P_NCCL=1 DUAL_PATH_VL_D_NCCL=1 DUAL_PATH_VL_P2D=2 DUAL_PATH_VL_D2P=2 \
  scripts/pd_benchmark.sh ...
```

想单独观察 D→P 时可以设 `DUAL_PATH_VL_D2P=3`。VL3 在 high 表里，只适合做对照实验。

## 5. 分阶段实施

| 阶段 | 内容 | 需要 | 风险 |
|---|---|---|---|
| 0 探查 | 两节点分别 `sudo bash scripts/infra/ib_qos_probe.sh`；结果写入 `results/ib-qos-probe/` | root | 只读 |
| 1 打标 | 确认当前 SL2VL 中 SL1/2 映射到运行中的 VL 后，设置 `DUAL_PATH_VL_*`（§4.5），用 `perfquery -X/-S`（按 SL 的收发计数）确认 NCCL 和 Mooncake 分别计入预期的 SL | 已实现，无 SM 改动 | 若 SL 指向非运行 VL，报文会被丢弃，必须先看阶段 0 的结果 |
| 2 存储分离 | 已完成：g6 经 CX7 挂载 NFS/RDMA（§4.3） | — | — |
| 3 SM QoS | 合并 `opensm-qos-dualpath.conf`，复查全部相关端口的表 | SM 管理权限、与朋志协调 | 共享子网；OperVLs 变化可能导致断链 |
| 4 节点内 | NIC H2D 已实现（[nic-h2d.md](nic-h2d.md)）；待验证 HCA 内部回环是否经过 VL 仲裁 | 实验 | 回环不受保护时只能靠限速 |
| 5 验证 | 按 §6 的实验矩阵 | — | — |

回滚顺序：先去掉所有 `DUAL_PATH_VL_*`（下一次启动即生效，所有流量回到 SL0/VL0，
而 VL0 在 high 表里），再按需恢复 SM 的原配置文件。

## 6. 验证实验

### 6.0 已完成：当前 2 条 VL 下的 TP4 干扰对照（Job 649 / 650）

不改 SM，TP4 探针放在 VL0、Mooncake 流量放在 VL1（`FORCE_NIC=0`），与全部在 VL0
对比。详细数据见 [nic-test/README.md](../nic-test/README.md) 的“VL 隔离对照”：

| 干扰流量 | 受影响的一侧 | 无 VL | 有 VL |
|---|---|---|---|
| D→P 跑满 | Decode（g6，跨对 NCCL 与 D→P 共用发送端口） | 32.2 倍 | **1.12 倍** |
| D→P 跑满 | Prefill（g5，NCCL 与 D→P 争用同一块 GPU 的 PCIe） | 4.09 倍 | 3.58 倍 |
| NIC 回环 H2D 跑满 | Decode / Prefill | 3.72 / 1.43 倍 | 3.90 / 1.82 倍 |

- credit 反压造成的队头阻塞，两条 VL 就能解决，不需要严格优先级。
- GPU PCIe 上的争用和 HCA 内部回环，VL 都管不到，只能靠限速或调度来避开。

### 6.1 已完成：端到端 1P1D 对照（Job 651–675）

结论：VL 能消除 D→P 对 decode step 的拖慢，但这种干扰本来就只在 D→P 传输的
几十毫秒内出现，摊到 JCT 上不到 1%，所以 VL 基本不改变 JCT。原因如下：

- D→P 和 NIC H2D 都是单次突发（每次 55–80 ms / 75–150 ms），都处在发起请求自己的
  TTFT 关键路径上，只占该请求 TTFT 的 0.6–5.4%。
- `KVTRANS=full` 下，请求自己的 forward 不会和自己的 D→P、H2D 重叠；能被干扰的
  只有恰好同时在跑的其他请求，干扰也只持续一个传输窗口。
- D→P 的时间占空比只有 2.6–8.3%，6.5–13.4% 的 decode step 会与它重叠。无 VL 时，
  这些 step 每步多 3–11 ms，折算到每个 agent 上只占 JCT 的 0.16–0.86%。

配置：TP4、32K、`KVTRANS=full`、L3 限速 8 GB/s、NFS/RDMA、NIC H2D stream，
hostpin 镜像。VL 配置为 `P_NCCL=D_NCCL=0`、`P2D=D2P=P_NIC_H2D=D_NIC_H2D=1`。
651–663 使用默认 32K 数据集（cache hit 0.88），664–675 使用从 trace 255 起的
94 turns 数据集（cache hit 0.95）。JCT 取最慢轨迹的完成时间。每个点只跑了 1 次。

表中列的含义：

- D→P、D 侧 H2D：每个请求在最慢 TP rank 上的耗时，括号内为占平均 TTFT 的比例。
- 占空比：D→P 窗口并集占整个 run 时长的比例。
- 重叠 step：与其他请求 D→P 窗口重叠的 decode step（ITL）的占比，以及它们相对
  同一 run 中不重叠 step 的平均多出耗时。
- 折算：这些多出耗时的总和除以 agent 数，再除以 JCT。

| Job | 路径 | agents | VL | JCT (s) | TTFT (s) | D→P | D 侧 H2D | 占空比 | 重叠 step | 折算 |
|---|---|---|---|---|---|---|---|---|---|---|
| 651 | decode | 8 | off | 290.6 | 2.61 | 69 ms (2.7%) | 101 ms (3.9%) | 4.0% | 6.5%，+4.4 ms | 0.53% |
| 652 | decode | 8 | on | 286.4 | 2.66 | 69 ms (2.6%) | 102 ms (3.8%) | 4.5% | 6.7%，+0.8 ms | 0.10% |
| 655 | dynamic | 8 | off | 273.8 | 1.94 | 78 ms (4.0%) | 102 ms (5.2%) | 2.6% | 3.3%，+10.2 ms | 0.68% |
| 656 | dynamic | 8 | on | 270.2 | 1.90 | 78 ms (4.1%) | 103 ms (5.4%) | 2.7% | 3.7%，+1.7 ms | 0.13% |
| 658 | decode | 16 | off | 481.4 | 7.75 | 70 ms (0.9%) | 105 ms (1.4%) | 5.5% | 9.1%，+3.6 ms | 0.37% |
| 659 | decode | 16 | on | 467.5 | 7.26 | 68 ms (0.9%) | 104 ms (1.4%) | 5.5% | 9.7%，−0.4 ms | 0 |
| 662 | dynamic | 16 | off | 425.3 | 5.42 | 75 ms (1.4%) | 107 ms (2.0%) | 3.3% | 4.8%，+4.7 ms | 0.30% |
| 663 | dynamic | 16 | on | 419.9 | 5.12 | 78 ms (1.5%) | 107 ms (2.1%) | 3.4% | 5.6%，−1.4 ms | 0 |
| 666 | decode | 8 | off | 536.7 | 4.32 | 59 ms (1.4%) | 76 ms (1.8%) | 7.7% | 10.0%，+3.4 ms | 0.29% |
| 669 | decode | 8 | on | 535.2 | 4.34 | 57 ms (1.3%) | 76 ms (1.8%) | 7.5% | 10.5%，−0.1 ms | 0 |
| 665 | dynamic | 8 | off | 321.0 | 1.80 | 58 ms (3.2%) | 84 ms (4.7%) | 6.1% | 8.2%，+4.7 ms | 0.53% |
| 668 | dynamic | 8 | on | 312.8 | 1.73 | 59 ms (3.4%) | 85 ms (4.9%) | 6.3% | 9.7%，−0.8 ms | 0 |
| 672 | decode | 16 | off | 1111.7 | 10.33 | 63 ms (0.6%) | 81 ms (0.8%) | 8.3% | 12.6%，+3.1 ms | 0.16% |
| 675 | decode | 16 | on | 1029.3 | 9.49 | 57 ms (0.6%) | 78 ms (0.8%) | 8.1% | 13.4%，+0.1 ms | 0.01% |
| 671 | dynamic | 16 | off | 640.8* | 3.75 | 60 ms (1.6%) | 85 ms (2.3%) | 6.6% | 11.2%，+10.9 ms | 0.86% |
| 674 | dynamic | 16 | on | 563.6 | 4.12 | 59 ms (1.4%) | 85 ms (2.1%) | 7.3% | 12.8%，+4.5 ms | 0.46% |

\* 671 中轨迹 `32k-0256` 第 43 轮卡在 Prefill bootstrap，客户端等满 300 s 超时后
重试，JCT 因此多出 300 s。这一对不能用于比较 JCT。

Dynamic 路径上，Prefill 侧还有一次 NIC H2D，每次 105–151 ms，与 Decode 侧的 H2D
一样只出现在发起请求自己的 TTFT 里。

**VL 改变了什么。** 在所有 8 对中，与 D→P 重叠的 decode step 多出的耗时从
+3.1–10.9 ms 降到约 0（只有 674 剩 +4.5 ms）。>60 ms 的长 step 也明显减少：
Decode 路径从 1.2–7.3% 降到 0.03–1.9%，Dynamic 路径从 11.8–13.3% 降到
2.1–4.3%（统计方法见 `nic-test/analyze_real_d2p.py`）。这与 Job 650 中“D→P
跑满时 Decode 的 32 倍拖慢降到 1.12 倍”是同一个机制。

**VL 没有改变什么。**

- D→P 自身的耗时不变（瓶颈是 g5 GPU 的 PCIe 写入），NIC H2D 的耗时也不变
  （HCA 内部回环，见 §6.0）。
- 被 D→P 重叠的 Prefill forward 只有 3–53 个，每 token 耗时与不重叠的 forward
  相同。

**JCT 的变化在噪声范围内。** 按机制估算，VL 能省下的 JCT 是上表“折算”列中
off 与 on 的差值，在 0.15–0.64% 之间。实际测得的 JCT 变化（VL 相对 off）如下：

| 路径 | 651 系列 a8 | 651 系列 a16 | trace255 a8 | trace255 a16 |
|---|---|---|---|---|
| decode | −1.4% | −2.9% | −0.3% | −7.4% |
| dynamic | −1.3% | −1.3% | −2.6% | 不可比（671 有 300 s 卡顿） |
| prefill（没有 D→P） | +0.9% / +2.5%（653 对 654 / 657） | +4.4% | 0.0% | −2.6% |

完全没有 D→P 的 Prefill 路径，单次 run 之间也有 −2.6% 到 +4.4% 的波动，Decode
和 Dynamic 的大多数差值都落在这个范围内。

672 对 675 的 −7.4% 超出了这个范围，但不是 D→P 干扰造成的：两次 run 的 L3 读取
总耗时都是 710 s（串行），JCT 的差值等于读取之间空闲时间的差值。这个空闲主要
发生在“下一个请求已到达，但还没进入 L3 读取队列”这一段，p50 分别为 223 ms 和
150 ms。其中只有约 11 ms 能用 VL 解释（最慢 rank 的 D→P 少 6 ms，rank 间偏差少
5 ms），其余部分目前没有打点，原因待查。

复现：

```bash
cd nic-test
python3 summarize_vl_pd.py 651 652 655 656 658 659 662 663 666 669 665 668 672 675 671 674
python3 analyze_real_d2p.py <run_dir> <decode-0.log> --prefill-log <prefill-0.log> --baseline <prefill-path run_dir>
```

### 6.2 待做的 perftest 矩阵

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
