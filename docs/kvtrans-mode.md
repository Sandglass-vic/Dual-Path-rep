# `DUAL_PATH_KVTRANS=layer|full`：前缀 KV 的加载形态

状态：已实现并通过 32K / 8 agents 的四组完整 replay（Job 638–641，见第 11 节）。
**运行脚本默认 `DUAL_PATH_KVTRANS=full`**，默认镜像为
`sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-l3bps-20260929-nich2d-async-kvtrans.sqsh`
（sha256 `608bb7e1…`）。补丁 `scripts/infra/patch_sglang_kvtrans.py`，运行时模块
`scripts/infra/dualpath_kvtrans.py`（装到 `sglang/srt/utils/`），镜像由
`scripts/infra/build_kvtrans_image.sh` 基于
`sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-l3bps-20260929-nich2d-async.sqsh` 构建。

## 1. 目标与开关

Prefill path 和 Decode path 都支持两种形态，由同一个变量控制：

| 形态 | 含义 |
|---|---|
| `layer` | 前缀 KV 按层送进 Prefill HBM，与 forward 逐层重叠 |
| `full` | 前缀 KV 全部送进 Prefill HBM 之后，请求才进入调度，forward 不再逐层等待 |

| `DUAL_PATH_KVTRANS` | Prefill path | Decode path |
|---|---|---|
| `full`（**运行脚本默认**） | `full` | `full` |
| `layer` | `layer` | `layer` |
| 未设置（只在直接启动 server 时出现） | `layer`（HiCache 原样） | `full`（旧 D→P 实现） |

- `pd_benchmark_lib.sh` 总是导出这个变量，缺省为 `full`，并写进 metadata 的 `dual_path_kvtrans`；只接受 `layer` 和 `full`。
- 镜像内部“未设置”的行为与旧镜像一致，用来复现历史结果：直接启动 server 时不设这个变量即可。
- 选 `full` 作为默认的原因见第 11 节：两种形态的 JCT 差异在 1% 以内，`full` 的同步点更简单，Decode 侧不需要按层发送，也不会出现 `layer` 带来的 TPOT 上升。
- 只有这一个开关，没有限速或 flush 开关：flush 始终开启（见 5.2）。
- Prefill 在 REVERSE_FETCH 里带上自己的取值，Decode 发现与自己不同就让该请求失败，不做静默兼容。
- 设计要求：同形态下两条路径的数据流和同步点对齐，性能差异只来自 D→P 这一跳和 scatter 在哪一侧执行。

## 2. 现状（开关未设置）

### 2.1 Prefill path（`layer`）

```text
scheduler._prefetch_kvcache
  -> HiRadixCache.prefetch_from_storage       # 分配 host pool，启动预取线程
  -> CacheController._generic_page_get        # 逐页 get + set_from_flat_data_page（scatter）
  -> check_prefetch_progress                  # TP MIN consensus，_insert_helper_host 插入 host 节点
  -> get_new_batch_prefill
       -> add_one_req -> init_load_back       # 分配 HBM，加入 load_queue
       -> ready_to_load_host_cache -> start_loading
            load_stream 逐层 load_to_device_per_layer + load_events[i].record()
  -> forward：get_key_buffer(layer) -> LayerDoneCounter.wait_until(layer)
```

- 按层重叠只发生在 Host→HBM 这一段。L3 页包含所有层，要等整个请求读完。
- `LayerDoneCounter` 只有 3 个轮转槽位，与“正在执行的 batch”绑定，不能给调度前的请求长期占用。
- HiCache 里的 `transfer_hicache_all_layer` 只用于 D2H 备份（`backup_from_device_all_layer`），H2D 没有整块加载的入口，也没有“整块加载完再准入”的调度逻辑，所以 `full` 需要自己加调度部分，拷贝本身复用 `load_to_device_per_layer`。

### 2.2 Decode path（`full`）

```text
Decode: _pre_alloc -> start_decode_l3_prefetch（wait_complete，读进 PD Host staging，layer_first）
     -> 读完 -> _dualpath_init_receiver -> bootstrap（携带 decode_l3_hit_tokens）
Prefill: pop_bootstrapped -> _prepare_decode_l3_prefix
     -> gloo MIN(hit) -> allocator.alloc(hit) -> start_decode_l3_reverse（REVERSE_FETCH）
Decode: _handle_decode_l3_reverse_fetch -> _send_kvcache_generic（同步，所有层所有页，写进 Prefill HBM）
     -> REVERSE_COMPLETE
Prefill: poll 在所有 rank 回 COMPLETE 前一直返回 Bootstrapping -> 放行 -> forward 只算 suffix
```

实测（Job 619）：`decode_scatter` 0.194 s，`d_to_p_transfer` 0.068 s，两者都在关键路径上。

## 3. 统一的阶段模型

| 段 | 内容 | 同步点 |
|---|---|---|
| S1 | L3 读取，并 scatter 到某个 host 缓冲区 | — |
| S2 | 确定命中数（TP MIN） | **A：命中数确定** |
| S3 | 前缀进入 Prefill 可见的位置 | — |
| S4 | Host→HBM | **B：前缀可用** |
| S5 | Prefill forward（只算 suffix） | — |

|  | Prefill `full` | Decode `full` | Prefill `layer` | Decode `layer` |
|---|---|---|---|---|
| S1 目标缓冲区 | Prefill host pool | Decode staging | Prefill host pool | Decode staging |
| S3+S4 | HiCache 整块 H2D | D→P 整块 RDMA 写 Prefill HBM | HiCache 按层 H2D | D→P **按层** RDMA 写 Prefill HBM |
| 传输何时开始 | 预取完成后，调度前 | REVERSE_FETCH 后 | 组 batch 时 | REVERSE_FETCH 后 |
| 同步点 B | 调度前：finish event 在 TP 内取 MIN | 调度前：所有 rank 收到 COMPLETE | forward 内：逐层 CUDA event | forward 内：逐层远端标志（`cuStreamWaitValue32`） |
| 准入条件 | 整块完成 | 整块完成 | 预取完成 | 第 0 层已到达 |
| forward 是否等待 | 否 | 否 | 是 | 是 |

两种 `layer` 在 forward 里用同一个等待钩子：`get_key_buffer` / `get_value_buffer` 调用的 `LayerDoneCounter.wait_until`。

## 4. Prefill path `full`

整块加载复用 HiCache 的 host 节点和 `load_to_device_per_layer`（hicache、kernel、`DUAL_PATH_CUDAMEMCPY=nic` 都走这个入口），只把加载时机从“组 batch 时”提前到“预取完成时”，并把完成判断挪到调度前。

```text
get_new_batch_prefill
  -> check_hicache_events -> dualpath_full_poll        # 每轮一次：所有待定预加载的 event.query()，一次 gloo MIN
  -> 遍历 waiting queue
       check_prefetch_progress == True
       -> dualpath_full_preload(req, max_prefill_tokens)
            首次：match_prefix -> _dualpath_full_load_back -> CacheController.dualpath_full_load
                  （load_stream 上逐层 H2D，记录请求级 finish event，ack 进 ack_load_queue）
                  + preload lock；返回 False
            之后：返回“所有 rank 都完成”
       -> 为 True 才 add_one_req（此时 match_prefix 命中 device 节点，host_hit_length=0）
       -> 准入成功后 dualpath_full_admitted：释放 preload lock，记录 H2D 计时 event
```

- **不占 `LayerDoneCounter`、不进 `load_queue`**：否则会被同一轮其他请求的 `start_loading` 合并，batch 拿到的 consumer index 与实际加载对不上，还会触发 `update_producer` 的“槽位未完成”断言。
- **两把锁**：
  - `load_back` 自带的锁由 `loading_check` 在 finish event 完成时释放，沿用原逻辑；
  - preload lock 从加载开始一直持有到准入成功，防止“加载完成到准入”之间节点被驱逐。
  - `add_one_req` 在准入时会对 `req.last_node` 加自己的锁，之后才释放 preload lock。
  - 准入失败（NO_TOKEN 等）时保留 preload lock，下一轮直接准入，不会退化成按层加载。
  - 请求在 waiting queue 里被 abort 时，`release_aborted_request` 释放 preload lock。
- **HBM 上限，防止死锁**：已预加载但尚未准入的前缀都被锁住，不可驱逐。所以同时持有的预加载 token 总数不超过 `max_prefill_tokens`（至少允许一个）。否则排队请求可能占满 HBM，队首请求因为 NO_TOKEN 永远进不去。
- **TP 一致**：待定预加载列表按开始顺序排列，各 rank 的调度是确定性的，列表一致；每轮只做一次 gloo MIN，与待定数量无关。
- **回退**：命中太短（低于 `load_back_threshold`）或 HBM 不够时，`dualpath_full_preload` 返回 True，交给原来的按层路径处理，不丢请求。
- **计时**：预加载的起止 event 追加到请求的 `dualpath_h2d_event_pairs`。该 batch 的 consumer index 是 -1，所以 `prefill_host_to_device` 完整计入关键路径，与 Decode `full` 的 `d_to_p_transfer` 口径一致。
- **额外开销**：只有存在待定预加载的调度轮才做一次 gloo all-reduce；准入比“预加载完成”最多晚一个调度轮（毫秒级）。

## 5. Decode path `layer`：按层 RDMA 直接写进 Prefill HBM

与 Prefill HiCache 的按层加载一一对应，区别只有两点：搬运方从本地 `load_stream` 换成 Decode 的 RDMA；完成信号从本地 CUDA event 换成 Decode 写进来的标志。

### 5.1 协议

```text
Decode:  start_decode_l3_prefetch（读进 staging）-> bootstrap(hit_tokens)             # 与 full 相同
Prefill: _prepare_decode_l3_prefix：gloo MIN(hit) -> 分配前缀 HBM 页
         -> start_decode_l3_reverse：分配标志槽并清零
         -> REVERSE_FETCH：原 7 帧 + [MODE_TAG, kind(layer|full), 标志槽 host 地址]
         -> poll：标志 >= 1（第 0 层已到达）即视为就绪，poll_and_all_reduce 取 TP 共识
         -> 准入：dualpath_layer_admit 启动看门狗计时，req.dualpath_kvtrans_flag_ptr = 标志的 device 地址
         -> forward：tp_worker 把 batch 里各请求的标志地址交给 LayerDoneCounter
                     wait_until(i) 在当前 stream 上 cuStreamWaitValue32(flag, i+1, GEQ|FLUSH)
Decode:  decode_thread 把 REVERSE_FETCH 放进队列，专用发送线程处理（full 同样走这个线程）
         每层：同步写该层所有页的 K、V -> 4 B RDMA READ 回读 K、V 尾部 -> 4 B 写标志 = i+1
         -> REVERSE_COMPLETE
Prefill: 收到 COMPLETE -> ack 该 room；sender.clear() 时释放标志槽（进入隔离期）
```

- **退回 `full` 的情况**：Prefill 没开 HiCache（没有 layer counter）、MLA、`transfer_infos[room]` 不止一个 Decode rank、标志槽耗尽。退回时 REVERSE_FETCH 的 kind 为 `full`，Decode 按原来的整块方式发送。
- **准入等第 0 层**：刚发出 REVERSE_FETCH 就准入的话，Decode 端的响应延迟会让 Prefill 整条计算 stream 停在第 0 层，挡住同 batch 和后续 batch。等第 0 层落地后，GPU 只会在“传输比计算慢”的那几层上等待。
- **发送线程**：原来 D→P 在 `decode_thread` 里同步执行，会挡住其他 room 的 KV 状态消息。现在 `layer` 和 `full` 都交给单独的线程，FIFO 顺序不变。

### 5.2 标志、flush 和 GPU 等待

- **标志区**：Prefill 每个 TP rank 一块 memfd（512 个槽）。
  - 布局：`flags[u32] | target[u32] | timedout[u32] | pad | deadline[f64]`；
  - 用 `cudaHostRegister(Portable|Mapped)` 注册给 CUDA，用 `cuMemHostGetDevicePointer` 取 device 地址；
  - 用 `batch_register` 注册到 PD 的 Mooncake engine，只注册 flags 这一段。
- **Decode 端常量**：注册一个 `[1..L]` 的 u32 数组作为标志写的源，另有一个 64 B 的 scratch 作为 READ 的目的。
- **每层的顺序**：
  1. `batch_transfer_sync_write` 写完该层 K、V 的所有块；
  2. `batch_transfer_sync_read` 回读 K 和 V 目的区域的最后 4 B（flush，始终开启）；
  3. 写 4 B 标志 = i+1。
- **为什么要 flush**：RC 写收到 ACK 只说明对端 HCA 收到了数据，不保证已经写进 GPU。READ 由同一块 Prefill NIC 发起 DMA 读，PCIe 规定它不能越过这块 NIC 之前的 posted write，所以 READ 完成就说明这一层的数据已经落进 HBM，之后写的标志不会跑到数据前面。每层多一次约 10 µs 的往返，64 层不到 1 ms。
- **GPU 等待**：通过 ctypes 调用 `libcuda.so.1` 的 `cuStreamWaitValue32_v2`，flags 为 `GEQ`；设备支持 `CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES` 时再加上 `FLUSH`。
  - 只往 stream 里排一个等待操作，不阻塞 CPU；
  - K 和 V 都会调用 `wait_until`，但每层只排一次（`dualpath_remote_waited` 去重）；
  - 每个 batch 在 `set_hicache_consumer` 旁边重设标志列表，本 batch 没有这类请求时开销为零。

### 5.3 避免死锁

| 风险 | 处理 |
|---|---|
| Decode 死掉或卡住，Prefill GPU 永远停在 wait 上 | 看门狗**子进程**共享 memfd，每 5 ms 扫描一次；准入后 30 s 仍未收齐，就把标志写成 L 并置 `timedout`，GPU 继续执行；`sender.poll` 看到 `timedout` 后让请求失败 |
| 看门狗拿不到 GIL | 看门狗是独立进程，不是线程。主线程可能卡在 kernel launch 里（stream 被 wait 阻塞、launch 队列满），而 launch 时 PyTorch 不释放 GIL，同进程里的 Python 线程就跑不起来 |
| Decode 发送出错 | Decode 先写标志 = L 解除 GPU 等待，再发 COMPLETE(Failed)；Prefill 标记失败，inflight 队列按原逻辑 abort |
| 迟到的标志写进已复用的槽 | 释放时先写成 L，进入隔离期；收到该 room 的 COMPLETE 且过了 2 s，或者过了 120 s，才回收 |
| 回收时 GPU 上还排着等旧槽的 wait | 同上：释放时先写成 L，且至少隔离 2 s |
| 控制 socket 并发 | 见第 6 节 |

## 6. 控制 socket 锁（修复提交 8aa81fb）

8aa81fb 给 `MooncakeKVManager._connect` 返回的 socket 包了一层锁，用来防止 Decode L3 反向消息和 transfer worker 在同一个 ZMQ PUSH socket 上交错发帧。问题有三个：

- **全局一把锁**：`_dualpath_send_lock` 是类属性，所有 endpoint 的发送都串行在这一把锁上；而且持锁期间调用的 `send_multipart` 会阻塞（对端未连上或达到 HWM）。只要一个对端慢，所有对端的状态消息都会被拖住。
- **包装只暴露 `send_multipart`**：调用其他 socket 方法会报 `AttributeError`。
- **帧数写死**：REVERSE_FETCH 要求 `== 7`、COMPLETE 要求 `== 4`，协议追加字段后消息会被当成畸形丢掉。

修复（仍在 `patch_mooncake_control_socket_safety` 里）：

- 每个 `(endpoint, is_ipv6)` 对应一个 `_DualPathLockedSocket`，各自持有一把锁，只串行同一个 socket 上的发送；
- 类级的 `_dualpath_socket_guard` 只保护 socket 的创建（双重检查），发送时从不持有它；
- 包装类通过 `__getattr__` 透传其他方法；
- 帧数校验改为 `< 7` / `< 4`，允许在末尾追加字段。

## 7. 计时

- `full`：Prefill 的 `prefill_host_to_device` 取预加载的起止 event；Decode 的 `d_to_p_transfer` 不变。两者都全部计入关键路径。
- `layer`：`d_to_p_transfer_raw_seconds` 仍是整个按层发送的时长，其中大部分与 forward 重叠。新增诊断字段 `d_to_p_first_layer_seconds_diagnostic`（第 0 层到达用时），这部分才在关键路径上。另外新增 `kvtrans_mode` 字段。summarizer 暂未区分两种形态，比较 `layer` 的 TTFT 时要用首层字段扣除重叠部分。

## 8. 改动清单

| 文件 | 改动 |
|---|---|
| `scripts/infra/dualpath_kvtrans.py`（新增） | 模式解析、`RemoteLayerFlags`（memfd 标志区、槽位与隔离期、看门狗子进程）、`LayerSendConstants`、`wait_remote_layer` |
| `scripts/infra/patch_sglang_kvtrans.py`（新增） | 幂等补丁，必须在 `patch_sglang_benchmark_timing.py` 之后执行 |
| ↳ `managers/cache_controller.py` | `LayerDoneCounter` 远端标志等待；`dualpath_full_load` |
| ↳ `mem_cache/hiradix_cache.py` | `dualpath_full_preload` / `_poll` / `_admitted` / `_release` |
| ↳ `managers/scheduler.py` | 准入门控 |
| ↳ `managers/tp_worker.py` | 每个 batch 设置标志列表 |
| ↳ `disaggregation/mooncake/conn.py` | 标志槽分配、REVERSE_FETCH 追加帧、首层就绪、专用发送线程、按层发送与 flush、失败解除、超时检查 |
| ↳ `disaggregation/prefill.py`、`decode.py`、`scheduler_output_processor_mixin.py` | 准入时启动看门狗计时、首层计时 |
| `scripts/infra/patch_sglang_benchmark_timing.py` | 控制 socket 锁修复（第 6 节） |
| `scripts/infra/build_kvtrans_image.sh`（新增） | 基于 nich2d-async 镜像构建 `-kvtrans` 镜像 |
| `scripts/tests/runtime/pd_benchmark_lib.sh`、`scripts/pd_benchmark.sh` | `DUAL_PATH_KVTRANS` 校验、export、透传，metadata 字段 `dual_path_kvtrans` |

用法（`pd_alloc.sh` / `pd_benchmark.sh` 的默认镜像已经是 kvtrans 镜像）：

```bash
scripts/infra/build_kvtrans_image.sh              # 重新构建时使用，输出 ...-nich2d-async-kvtrans.sqsh
scripts/pd_benchmark.sh ...                       # 默认 DUAL_PATH_KVTRANS=full
DUAL_PATH_KVTRANS=layer scripts/pd_benchmark.sh ...
```

## 9. 验证计划

1. **正确性**：1P1D，Prefill path 和 Decode path 各跑 `layer` 和 `full`，共 4 种组合，再加开关未设置的对照。同一轮对话的贪心输出要逐 token 一致，命中 token 数要一致。
2. **同步点**：打开 `DUALPATH_TTFT_DETAIL`，检查：
   - `full` 下 forward 开始时间晚于预加载的 finish event 或 COMPLETE；
   - `layer` 下准入晚于第 0 层标志，D→P 完成时间落在 forward 窗口内。
3. **故障注入**：让 Decode 在第 k 层停止发送，确认 30 s 后看门狗解除 GPU 等待、请求以失败结束、同 batch 的其他请求输出正常。
4. **性能**：32K、8 agents、rep1，四种组合各一组。
   - 同形态下两条路径的 TTFT 差，去掉排队之后应不超过约 0.1 s；
   - Decode `layer` 的 `queue_wait` 应明显低于 Decode `full`；
   - `prefill_computation` 的增量就是按层 D→P 对 TP 通信的干扰代价（见 Job 627：写 GPU 的 RDMA 会让 all-reduce 慢到 3.6 倍）。

## 10. 预期与风险

| 组合 | 关键路径上与传输相关的部分 | 相对当前 |
|---|---|---|
| Prefill `layer` | 读取 + scatter + 第 0 层 H2D | 不变 |
| Prefill `full` | 读取 + scatter + 整块 H2D（约 64 ms） | TTFT 约 +60 ms |
| Decode `full` | 读取 + scatter + D→P（68 ms） | 不变 |
| Decode `layer` | 读取 + scatter + 第 0 层 D→P（约 1～2 ms）+ 干扰 | TTFT 约 −15～−65 ms |

- **干扰集中**：按层 D→P 与 Prefill 自己的 all-reduce 完全重叠，是 Job 627 里最差的形态。在 VL 隔离（[vl-traffic-isolation.md](vl-traffic-isolation.md)）上线前，净收益可能被干扰吃掉大半。
- **flush 的前提**：flush 依赖“READ 和之前的写走同一块 Prefill NIC”。Mooncake 按目的 GPU 选 NIC；如果它把同一块 GPU 的写分散到多块 NIC 上，READ 就只能保证其中一块 NIC 的写已落地。GPU 端的 `CU_STREAM_WAIT_VALUE_FLUSH` 能补一部分，但仍需要验证（第 9 节第 1 条）。
- **看门狗超时**：固定 30 s。正常的按层发送在 1 s 以内完成；只有 Decode 故障时，Prefill GPU 才会被挡住这么久。
- **标志槽**：512 个，耗尽时退回 `full` 并打 warning。

## 11. 实测结果（Job 638–641）

配置：32K、8 agents、rep1，L3 限速 8 GB/s，NFS/RDMA，`DUAL_PATH_CUDAMEMCPY=nic`（stream 提交），NCCL 允许走 NVLINK（`DUAL_PATH_NCCL_FORCE_NIC=0`）。四组都完成了 208/208 轮、50,811 个输出 token，cache hit 都是 88.18%，没有触发看门狗超时，也没有协议不一致。

| 路径 | KVTRANS | Job | JCT | 平均 TTFT | 平均 TPOT | TPOT p90 |
|---|---|---|---|---|---|---|
| Prefill | layer | 638 | 265.10 s | 2.473 s | 31.29 ms | 34.50 ms |
| Prefill | full | 640 | 263.81 s | 2.537 s | 30.92 ms | 34.29 ms |
| Decode | layer | 639 | 276.97 s | 2.563 s | 33.29 ms | 38.65 ms |
| Decode | full | 641 | 275.23 s | 2.781 s | 31.56 ms | 35.18 ms |

TTFT 分解（关键路径均值，秒）：

| 阶段 | P-layer | P-full | D-layer | D-full |
|---|---|---|---|---|
| queue_wait | 1.111 | 1.187 | 1.153 | 1.388 |
| L3 读取 | 0.605 | 0.605 | 0.605 | 0.605 |
| scatter | 0.143 | 0.149 | 0.184 | 0.168 |
| Prefill H2D | 0.004（原始 0.119） | 0.098 | — | — |
| D→P | — | — | 0.103（首层 0.0016） | 0.064 |
| Prefill 计算 | 0.310 | 0.300 | 0.339 | 0.294 |

- **两种形态都按设计生效**：Prefill `full` 的 H2D 全部进入关键路径（4 ms → 98 ms）；Decode `layer` 的 800 条 rank 级记录全部走按层发送，首层平均 1.6 ms 到达。
- **为什么 full 和 layer 结果接近**：TTFT 的 70% 以上是排队和 L3 读取（L3 限速是瓶颈），KVTRANS 只影响 64～100 ms 的传输段；JCT 的约 75% 是 decode（约 6350 token × 31 ms），每个 agent 26 轮累计的差异只有 1.6～2.6 s，约为 JCT 的 1%，在单次运行的波动范围内。
- **Decode `layer` 的收益被抵消**：关键路径上的 D→P 省了约 62 ms，但按层同步发送让 D→P 总时长从 64 ms 涨到 105 ms，Prefill 计算慢了约 45 ms（按层 RDMA 写 GPU 与 TP 通信重叠）。D-layer 比 D-full 快的 0.22 s TTFT 主要来自 `queue_wait`，只跑了一次，不能算作 `layer` 的收益。汇总脚本还没有区分形态，表里 D-layer 的 D→P 0.103 s 包含了与计算重叠的部分。
- **Decode `layer` 的 TPOT 上升**：TPOT p10 四组相同（约 26.6 ms），没有其他请求在传输时 decode step 都是约 27 ms，差异只出现在“有请求在预分配或传输”的时段。可能的原因是按层发送每请求 192 次 Mooncake 调用，每次返回都要重新拿 GIL，打断 Decode 调度线程；未验证。
- **结论**：运行脚本默认使用 `full`。`layer` 保留为可选项，要让它有收益，需要按层组发送（减少调用次数），并在 L3 不是瓶颈的场景下重新评估。
