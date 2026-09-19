# Dynamic L3 Read Path 设计

本文设计在现有 Prefill-owned 和 Decode-owned L3 read 之外，引入 request-scoped
的 `dynamic` read path。Dynamic 模式下，每个 request 单独选择 Prefill 或 Decode
作为 L3 KV 读取方；两侧通过同一个不可变 decision 协调工作。

**实现状态**：§1–§9 已实现（阶段一至三），见 `srt/utils/dualpath_read_path.py`。
尚未实验验证。与本文原稿的两处偏差见 §14。

## 1. 目标与约束

目标：

- 保留静态 `prefill` 和 `decode` 行为；
- 新增 `dynamic` 配置模式；
- 默认在**每条 trajectory 内部**交替选择 Prefill/Decode；
- 一个 request 只能由一侧执行 L3 read；
- Prefill 和 Decode 必须使用同一个 request-scoped path decision；
- 复用现有 Prefill scheduler、`DecodePreallocQueue`、Mooncake transfer 和 TP
  completion consensus；
- path 选择、L3 timing、`cached_tokens` 和结果汇总使用同一份 request decision。

非目标：

- 第一阶段不实现基于实时带宽、队列长度或负载预测的复杂调度；
- 不复制现有两套 L3 pipeline；
- 不允许同一 request 同时从 Prefill 和 Decode 读取 L3。

### 硬约束

以下四条来自现有代码，违反任何一条都会导致引擎卡死、跨轮命中失效或实验结论错误。

**C1. decision 必须在请求进入引擎前就已确定，引擎内不得等待。**
Prefill 的 L3 prefetch 触发点是 `_prefetch_kvcache()`，它在 `_add_request_to_queue()`
里被调用，而后者跑在 scheduler event loop 上：

```python
elif self.disaggregation_mode == DisaggregationMode.PREFILL:
    self._prefetch_kvcache(req)                      # <- 决策必须在此刻已可用
    self.disagg_prefill_bootstrap_queue.add(req, ...)
```

在这里做任何阻塞等待都会停住整个引擎而不只是当前请求。而且此刻 bootstrap 握手
尚未发生，Prefill 与 Decode 之间还没有任何通道，"向对方查询"在物理上就不成立。

**C2. L3 写入归属必须始终留在 Decode。**
`hiradix_cache._inc_hit_count()` 的注释即为 "Prefill is an L3 reader only; Decode
publishes the full context"，由 `DUALPATH_HICACHE_READ_ONLY=1` 保证。dynamic 只切换
**读**的归属；一旦让 prefill-owned request 也写 L3，跨轮命中链会断。

**C3. 选路维度不能与 agent 身份混淆。**
见 §3.1 的反例：按全局 admission 序交替会退化成按 agent 静态分片。

**C4. 资源预算按峰值配置，不按均值。**
交替只保证总量上各占一半，突发时仍可能有连续多个同侧 request 在飞。Prefill 的
`HICACHE_SIZE_GB` 与 Decode 的 staging 池都要按"全部请求落在本侧"配置。Prefill
预取配额的计算方式见
[HiCache 与 DualPath](hicache-and-dualpath.md#host-pool-尺寸与-l3-prefetch-配额)。

## 2. 配置模型

```bash
DUALPATH_L3_READ_PATH=prefill
DUALPATH_L3_READ_PATH=decode
DUALPATH_L3_READ_PATH=dynamic
```

新增参数：

```bash
DUALPATH_DYNAMIC_PATH_POLICY=alternate_per_turn
DUALPATH_DYNAMIC_FAIL_MODE=fail_closed
```

| 配置 | 行为 |
|---|---|
| `prefill` | 所有 request 使用 Prefill L3 read |
| `decode` | 所有 request 使用 Decode L3 read |
| `dynamic` | 每个 request 单独决定 path |
| `alternate_per_turn` | 每条 trajectory 内按 turn 交替，轨迹间相位错开（默认） |
| `same_turn_lockstep` | 所有轨迹在同一 turn 同时切换，**仅用于诊断** |
| `fail_closed` | 请求未携带合法 decision 时拒绝该 request |

没有 `DUALPATH_DYNAMIC_DECISION_TIMEOUT`：decision 随请求一起到达（§4），引擎侧
不存在"等待 decision"这一状态，因此也不存在超时（约束 C1）。

`dynamic` 只能是配置层模式。request 最终只能得到 `prefill` 或 `decode`。

三类进程必须使用相同配置：

```bash
DUALPATH_L3_READ_PATH=dynamic scripts/pd_benchmark.sh prefill
DUALPATH_L3_READ_PATH=dynamic scripts/pd_benchmark.sh decode
DUALPATH_L3_READ_PATH=dynamic scripts/pd_benchmark.sh client
```

allocation 只保存节点、TP 和资源信息，不保存 request path；path 模式在启动
Prefill、Decode 和 client 时指定。

## 3. Request-scoped decision

建议新增独立 read-path 模块，避免继续在 scheduler 中直接判断环境变量。

```python
class ReadPath(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    DYNAMIC = "dynamic"


class ResolvedReadPath(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class ReadPathDecision:
    request_id: str
    sequence_no: int
    path: ResolvedReadPath
    policy: str
    epoch: str
    issued_at_ns: int
```

decision 必须满足：

- 一个 request 只生成一次；
- 生成后不可修改；
- retry、streaming chunk 和 TP rank 不重新分配 path；
- Prefill 和 Decode 看到的是同一份 decision；
- decision 携带 allocation/epoch，防止跨 allocation 误复用。

### 3.1 选路维度：为什么不能用全局 admission 序

按全局 admission 顺序交替（`sequence_no % 2`）在本项目的 workload 下会退化成
**按 agent 静态分片**。用 482 那轮的真实到达顺序（`Req Time Stats` 的 `start=`）
模拟该策略：

| trajectory | prefill | decode | prefill 占比 |
|---|---:|---:|---:|
| `32k-0000` | 1 | 25 | 4% |
| `32k-0001` | 26 | 0 | 100% |
| `32k-0002` | 1 | 25 | 4% |
| `32k-0003` | 24 | 2 | 92% |

每一轮的 4 个请求确实 2:2 均分，但均分被锁死在 agent 身份上。原因是各 agent 串行
推进（turn N+1 等 turn N 返回），每轮到达顺序是近似稳定的排列，周期 4 与交替周期 2
对齐，sequence 的奇偶性等价于 agent 的奇偶性。

后果是实测的其实是"两条轨迹全程 prefill vs 两条全程 decode"，而不是动态切换；每条
轨迹全程不发生 path transition，§10 的 transition 图没有数据点。这类缺陷不报错、
不崩溃，数据看上去完全正常，因此必须在设计层排除。

默认策略 `alternate_per_turn` 改为按 `(trajectory, turn)` 决策，让每条轨迹内部交替，
并用 trajectory 做相位错开，避免所有轨迹在同一 turn 同时切到同一侧：

```python
phase = stable_hash(trajectory_id) & 1
path = "prefill" if (turn_index + phase) % 2 == 0 else "decode"
```

全局 `sequence_no` 仍然分配并记录，但只作为审计信息，不参与选路。
`same_turn_lockstep` 保留为诊断选项，用于制造可控的退化对照。

### 3.2 谁来分配：客户端即全局 admission 点

不需要独立的 coordinator 进程、RPC 或共享文件锁。现有系统已经有一个天然的全局
admission 点：

- 客户端 `scripts/benchmark/replay_fake_agent_traces.py` 是**单进程 +
  `ThreadPoolExecutor`**（每 agent 一个 worker）；
- 它自己生成并完全控制 `rid`，请求体即
  `{"rid": effective_request_id, "input_ids": ..., ...}`；
- 它天然知道 `(trajectory, turn)`，而引擎侧并不知道。

因此 decision 在客户端分配，一个进程内的计数器加锁即可，既满足全局有序，又满足
约束 C1。

```python
with lock:
    sequence_no = next_sequence
    next_sequence += 1
decision = make_decision(request_id, sequence_no, trajectory_id, turn_index)
```

引擎侧不再有 `decide()`/`wait()`，只有一个纯解析与校验函数：

```python
def read_path_from_request(req, configured_path) -> ReadPathDecision:
    """静态模式返回固定 path；dynamic 模式解析请求携带的 decision。"""
```

这样就不存在"两侧不一致"的可能，也就不需要 `acknowledge()`、`wait()` 超时和跨进程
冲突检测。

## 4. Decision 传递

PD 模式下 Prefill 和 Decode **各自独立从 Router 收到完整的 `Req` 对象**
（`DecodePreallocQueue.add(req: Req)` 直接接收 `Req`，而不是从 bootstrap 握手里
还原）。所以 decision 只要放进请求体，两侧天然拿到同一份：

```json
{
  "rid": "...",
  "dualpath_read_path": "prefill",
  "dualpath_path_sequence": 12,
  "dualpath_path_trajectory": "32k-0001",
  "dualpath_path_turn": 19,
  "dualpath_path_epoch": "allocation-470"
}
```

```text
Client (单进程, 全局 admission 点)
    |
    | 分配 decision
    |
    +--> Router --+--> Prefill: Req(含 decision)
                  |
                  +--> Decode:  Req(含 decision)
```

实现上需要给 `GenerateReqInput` 增加字段并在 `Req` 构造时透传，Router 转发请求体
时不做字段裁剪。仓库已经通过 `patch_sglang_benchmark_timing.py` 改过 `Req` 的构造
（`extra_key` 即为一例），沿用同一注入点即可。

### TP rank 一致性不需要额外协调

`Scheduler.recv_requests()` 只由 `attn_tp_rank == 0` 收取请求，随后用
`broadcast_pyobj()` 广播给所有 TP rank。decision 在请求对象里，所有 rank 天然一致。

**不要**在 `_add_request_to_queue` 这类非集合通信点新增 per-request 广播来同步
decision：既增加关键路径延迟，又引入死锁面——一旦某个 rank 的请求集合与其他 rank
不一致就会挂死。

## 5. Request 状态机

```text
CREATED + PATH_DECIDED          (客户端，请求发出前)
  -> PREFILL_READY + DECODE_READY
  -> PATH_OWNER_PREFETCH
  -> L3_PREFIX_READY
  -> PD_TRANSFER
  -> COMPUTE
  -> RESPONSE
  -> RELEASED
```

`PATH_DECIDED` 不是引擎侧的一个独立阶段。请求到达引擎时 decision 必定已存在，否则
直接 fail-closed（约束 C1）。

建议在 request 上保存：

```python
req.dualpath_read_path
req.dualpath_path_sequence
req.dualpath_path_trajectory
req.dualpath_path_turn
req.dualpath_path_epoch
req.dualpath_path_owner
req.dualpath_l3_prefetch_started
req.dualpath_l3_prefetch_completed
```

状态转换由 scheduler/owner 管理，TP rank 只报告本地状态并参加已有 TP consensus。

## 6. 两条 request 路径

### 6.1 Prefill-owned request

当 `decision.path == prefill` 时：

1. Prefill scheduler 发起 L3 prefetch；
2. Prefill 执行 hash、exists、TP hit consensus、L3 read 和 scatter；
3. Decode 不发起 Decode L3 prefetch；
4. Prefill 通过 P→D 传输 durable prefix；
5. Decode 等待 Prefill prefix 和 transfer 完成。

该路径继续复用现有 Prefill `HiRadixCache`/prefetch 流程。

### 6.2 Decode-owned request

当 `decision.path == decode` 时：

1. Decode `DecodePreallocQueue` 发起 L3 prefetch；
2. Decode 执行 hash、exists、TP hit consensus、L3 read 和 scatter；
3. Decode 将 durable prefix 放入 Decode Host staging/HBM；
4. Decode 通过 D→P 将命中 prefix 发送给 Prefill；
5. Prefill 只计算 L3 miss suffix；
6. Prefill 将 suffix 通过 P→D 返回 Decode。

该路径继续复用现有 Decode scheduler-owned prefetch、D→P/P→D 和 completion sync。

## 7. 统一策略接口

Dynamic 只负责选 path，不应复制两套读取实现：

```python
class ReadPathStrategy:
    def prepare(self, req, decision): ...
    def start_l3_read(self, req, operation): ...
    def complete(self, req, operation): ...


class PrefillReadStrategy(ReadPathStrategy): ...
class DecodeReadStrategy(ReadPathStrategy): ...
```

统一入口（非阻塞，满足约束 C1）：

```python
decision = read_path_from_request(req, CONFIGURED_PATH)
strategy = strategy_registry.for_path(decision.path)
strategy.prepare(req, decision)
```

推荐模块结构：

```text
srt/utils/dualpath_read_path.py
  ReadPath / ResolvedReadPath / ReadPathDecision
  read_path_from_request()     # 解析与校验，无 I/O、无等待
  policy                       # alternate_per_turn / alternate_global
  strategy                     # Prefill/Decode path strategy
  request_state                # 状态与转换校验
```

### 注入方式

不要用 `replace_once()` 把 dynamic 逻辑逐段塞进 scheduler 函数体。该函数要求匹配数
恰好为 1，任何上游改动或 partial patch 基底都会让它直接抛异常，这类问题此前已经
踩到过。

正确做法在仓库里已有范式：`write_ttft_v3_helper()` 直接写出一个完整的新模块
`srt/utils/dualpath_ttft.py`。dynamic 应照此办理——逻辑全部放进新模块，patch 脚本
只保留极少量调用点注入（`_prefetch_kvcache`、`DecodePreallocQueue.add`、
`Req.__init__` 字段透传、timing emit）。

## 8. L3 read ownership

每个 request 必须满足：

```text
非 owner 侧不得发起 L3 read
```

注意这是**互斥**而不是 XOR：冷 cache 首轮和任何真实 miss 都会让两侧 `read_tokens`
同时为 0，这属于正常情况。判定只看 owner 归属，不看是否命中。

写入归属不参与 dynamic（约束 C2）：无论 decision 是哪一侧，L3 的 backup/publish
始终由 Decode 完成，Prefill 保持 `DUALPATH_HICACHE_READ_ONLY=1`。

进入 prefetch 前增加 request-level guard：

```python
if req.dualpath_l3_prefetch_started:
    raise RuntimeError(f"Duplicate L3 prefetch: {req.request_id}")

req.dualpath_l3_prefetch_started = True
req.dualpath_l3_prefetch_owner = side
```

必须拒绝以下情况：

- 请求未携带 decision，或 decision 字段不合法；
- `epoch` 与当前 allocation 不符；
- 同一 request 在同一侧重复启动 L3 read；
- path owner 与 timing marker 不一致。

decision 随请求传递（§4）之后，"两侧拿到不同 decision"已不可能发生，无需运行期
跨进程比对。保留 owner 与 timing marker 的一致性校验作为兜底。

默认采用 fail-closed，避免 fallback 导致双读或错误归因。

## 9. 现有代码的改造点

### `pd_runtime.sh` 和启动脚本

path 校验扩展为：

```bash
[[ $DUALPATH_L3_READ_PATH == prefill ||
   $DUALPATH_L3_READ_PATH == decode ||
   $DUALPATH_L3_READ_PATH == dynamic ]]
```

同时导出 dynamic 参数，并记录：

```text
configured_l3_read_path=dynamic
dynamic_path_policy=alternate_per_turn
dynamic_fail_mode=fail_closed
```

### 资源预算

按约束 C4，两侧都要按"全部请求落在本侧"配置，不能因为交替而砍半：

- Prefill：`HICACHE_SIZE_GB`（默认 40，A100 40GB 下给出约 413k tokens 的预取配额）；
- Decode：`DUALPATH_DECODE_PD_HOST_STAGING_GB` 与 `DECODE_HICACHE_RATIO`。

Prefill 预取配额 `0.8 * (host pool - device pool)` 的推导见
[HiCache 与 DualPath](hicache-and-dualpath.md#host-pool-尺寸与-l3-prefetch-配额)。
该配额被耗尽时 `prefetch_from_storage()` 会静默跳过，dynamic 下这会被误读成"选路
没生效"，排查时先看 `DUALPATH_L3_PREFETCH_SKIP`。

### `ttft_v3` timing helper

当前 helper 的模块级 `READ_PATH` 只能表示静态配置，不能表示 dynamic request
的实际 path。应将 path 改为 emit 时传入：

```python
def emit(logger, *, rid, side, tp_rank, resolved_path, values=None):
    payload = {
        "configured_path": CONFIGURED_PATH,
        "path": resolved_path,
        ...,
    }
```

结果中同时保留 `configured_path` 和 request 实际的 `path`。

### Prefill `HiRadixCache`

现有 `ongoing_prefetch` 已保存 request-scoped operation 和 `extra_key`。应将
`ReadPathDecision` 一并保存，确保 prefetch 完成、revoke、cancel 都使用同一 decision。

### Decode `DecodePreallocQueue`

现有 `PrefetchOperation` 增加：

```python
operation.read_path_decision = decision
```

scheduler enqueue 前检查：只有 `decision.path == decode` 才能创建 Decode L3
prefetch operation。

### Scheduler 分支

避免继续写：

```python
if DUALPATH_L3_READ_PATH == "decode":
    ...
```

统一改为：

```python
decision = read_path_from_request(req, CONFIGURED_PATH)   # 纯解析，不阻塞

if decision.path == ResolvedReadPath.DECODE:
    schedule_decode_l3_prefetch(req, decision)
else:
    schedule_prefill_owned_request(req, decision)
```

注意当前 patch 生成的 Prefill 分支是按进程级环境变量提前 return 的：

```python
dualpath_decode_l3_read = os.environ.get("DUALPATH_L3_READ_PATH", "prefill") == "decode"
if dualpath_decode_l3_read:
    return
```

dynamic 下必须改成按 `decision.path` 判断，且该判断要在
`_prefetch_kvcache()` 内、`req.init_next_round_input()` 之后立即完成，不得引入等待。

## 10. Timing、统计和绘图

Dynamic run 的 `run.json` 保存配置级信息：

```json
{
  "configured_l3_read_path": "dynamic",
  "dynamic_path_policy": "alternate_per_turn",
  "resolved_l3_read_paths": ["prefill", "decode"]
}
```

每个 request 保存实际 path：

```json
{
  "l3_read_path": "prefill",
  "configured_l3_read_path": "dynamic",
  "path_sequence_no": 12,
  "prefill_l3_read_tokens": 31744,
  "decode_l3_read_tokens": 0
}
```

`cached_tokens` 必须按 request 实际加载路径计算：

```python
cached_tokens = (
    row["prefill_l3_read_tokens"]
    if row["l3_read_path"] == "prefill"
    else row["decode_l3_read_tokens"]
)
```

每个 request 必须满足（互斥，不是 XOR）：

```text
l3_read_path == "prefill"  ->  decode_l3_read_tokens == 0
l3_read_path == "decode"   ->  prefill_l3_read_tokens == 0
```

不要断言"必有一侧大于 0"：冷 cache 首轮和任何真实 miss 都会让两侧同时为 0，那是
正常结果。校验只针对非 owner 侧。

汇总器不能再要求整个 run 只有一个 L3 path，应分别统计：

```text
dynamic_prefill_owned_requests
dynamic_decode_owned_requests
dynamic_prefill_l3_read_tokens
dynamic_decode_l3_read_tokens
dynamic_prefill_l3_read_seconds
dynamic_decode_l3_read_seconds
```

建议图表：request sequence/path、两侧 request 数量、两侧 L3 bytes、两侧带宽、
按 resolved path 着色的 TTFT，以及 path transition 处的 scheduler wait。

还应补一张**每条 trajectory 的 path 分布**图，用于直接暴露 §3.1 那类退化：如果某条
轨迹的 prefill 占比接近 0% 或 100%，说明选路维度出了问题，而不是结果本身。

### 可比性警告

dynamic run 里的 per-path 数字**不能直接和静态 run 的对应数字比较**。两条路径在同一
run 内共享 Lustre 带宽和 NIC，decode 路径还额外带有 D→P 反向传输和 completion sync
（482 那轮 `decode_l3_sync_wait_seconds_total` 为 43.17s）。把 dynamic 下的
`dynamic_prefill_l3_read_seconds` 与静态 prefill run 相比，会把相互干扰误读成路径
本身的开销差异。需要对比时应使用同等并发下的静态 run 作为基线，并显式标注干扰项。

## 11. 异常和并发语义

### Retry

retry 不增加 sequence，也不重新选择 path：

```text
request 12 -> prefill
retry request 12 -> prefill
```

这条只能在**客户端**实现。当前客户端在 stream 重试时会改写 rid：

```python
effective_request_id = f"{request_id}-stream-retry-{stream_retry_count}"
```

对引擎而言那是一个全新的、rid 不同的请求，引擎无从识别它是重试。因此客户端重试时
必须复用原 request 的 decision 原样下发，而不是重新分配。同理，decision 也不应该从
rid 反推——rid 会变。

### Cancel

cancel 后不重用 request ID，也不回收 sequence 号。未完成的 prefetch 必须经过现有
cancellation-safe release 流程，不能提前释放仍被 I/O 使用的 Host buffer。

### 并发 request

path 由 `(trajectory, turn)` 决定，与并发顺序无关。同一 trajectory 的相邻 turn 天然
交替：

```text
32k-0001 turn 18 -> prefill
32k-0001 turn 19 -> decode
32k-0001 turn 20 -> prefill
```

request 0 尚未完成不影响 request 1 使用 Decode path。全局 `sequence_no` 仍按客户端
admission lock 顺序单调递增，但只用于审计。

### decision 缺失

默认 fail-closed：请求未携带合法 decision 时直接失败并记录明确错误，不静默 fallback
到静态 path。不存在超时状态（约束 C1）。

### TP rank 不一致

decision 随请求对象经 `broadcast_pyobj()` 到达所有 rank，天然一致，无需额外协调。
各 rank 不得自行推导或分配 path；解析函数必须是纯函数，不得读取进程级可变状态。

## 12. 实施阶段

### 阶段零：离线验证选路策略

在改任何引擎代码之前，先用历史 run 的 `rounds.jsonl` 把候选策略跑一遍，输出每条
trajectory 的 path 分布。§3.1 那张退化表就是这样得到的。这一步成本接近零，却能挡住
最难事后发现的一类错误。

### 阶段一：统一静态模式

让现有 `prefill` 和 `decode` 都通过 `ReadPathDecision` 和 strategy 接口执行，
但 decision 由静态配置直接给出。验收静态行为、日志和 timing 与当前版本一致。

按 §7 的注入方式先把新模块落地，再逐个替换调用点；不要在这一步改动 L3 pipeline
本身。

### 阶段二：加入 dynamic decision

实现客户端 decision 分配、请求体字段透传、引擎侧解析与校验、dynamic 日志。先只验证
两侧解析出的 decision 一致，不改变实际 L3 pipeline——此时 dynamic 的行为应与当前
静态模式完全相同。

### 阶段三：切换 L3 owner

让 Prefill/Decode 根据 request decision 选择 L3 owner，复用现有两套 prefetch 实现，
禁止复制逻辑。

### 阶段四：统计和故障验证

至少验证：

- 每条 trajectory 内部相邻 turn 交替，且各轨迹 prefill 占比都接近 50%；
- 不同 trajectory 的相位有错开，不是所有轨迹在同一 turn 同时切换；
- retry 不切换 path（客户端复用 decision）；
- 每个 request 只有 owner 侧产生 L3 marker，非 owner 侧 `read_tokens` 恒为 0；
- 冷 cache 首轮两侧均为 0 不触发校验失败；
- 请求缺失 decision 时 fail-closed 并有明确错误，不死锁；
- TP4 所有 rank decision 一致；
- 无 `DUALPATH_L3_PREFETCH_SKIP ... reason=rate_limited`（否则预算配置有误，
  见约束 C4）；
- buffered/O_DIRECT 和 XFS/NFS 均可运行；
- Decode L3 completion sync 不影响 Prefill-owned request。

## 13. 最终运行模型

客户端（唯一的 admission 点，知道 trajectory 与 turn）：

```python
def send_turn(trajectory_id, turn_index, request_id, payload):
    decision = assign_decision(trajectory_id, turn_index, request_id)
    payload.update(decision.as_request_fields())
    return post_with_retry(payload, decision)     # 重试复用同一 decision
```

引擎两侧（纯解析，不阻塞、不协调）：

```python
def handle_request(req):
    decision = read_path_from_request(req, CONFIGURED_PATH)
    req.dualpath_decision = decision
    strategy = strategy_registry.for_path(decision.path)
    return strategy.run(req, decision)
```

核心原则：

1. `dynamic` 只负责选择，不负责实现具体读取；
2. 每个 request 只有一个不可变 path decision；
3. decision 在客户端分配、随请求下发，两侧解析而不协商；
4. 引擎内任何位置都不得为获取 decision 而阻塞（约束 C1）；
5. L3 read owner 是 request-scoped，而不是 process-scoped；
6. L3 写入归属始终在 Decode，不随 dynamic 改变（约束 C2）；
7. 选路维度是 `(trajectory, turn)`，不是全局 admission 序（约束 C3）；
8. 静态 `prefill` / `decode` 通过同一套 strategy 接口保留；
9. timing、统计和 `cached_tokens` 都按 resolved path 计算；
10. decision 缺失时默认 fail-closed，避免双读和错误归因。

## 14. 实现与本文原稿的偏差

**decision 的载体是 rid，不是新增请求字段。** §4 原本要给 `GenerateReqInput` 加字段并
依赖 Router 透传。实测发现客户端已经把 `(trajectory, turn)` 编码进它完全掌控的 `rid`，
而 Router 把同一个 rid 交给两侧引擎、`recv_requests()` 又广播给所有 TP rank。于是
`resolve(rid)` 这个纯函数天然在两侧、所有 rank 上给出同一结果，不需要改 `GenerateReqInput`、
`tokenizer_manager`，也不依赖 Rust Router 是否保留未知字段（后者无法在本地验证）。

§11 曾担心"rid 会变所以不能反推"——客户端 stream 重试会把 rid 改成
`<rid>-stream-retry-N`，但解析用的正则只取 `(trajectory, turn)`，后缀不影响结果。
已验证 104 条真实 rid 加后缀后决策不变。

**相位函数用轨迹编号，不用 hash。** §3.1 写的是 `stable_hash(trajectory_id) & 1`。实测
`crc32` 对 `32k-0000`..`32k-0003` 给出**相同**的奇偶位，四条轨迹会全部同相，正好制造出
§3.1 想避免的那种退化。改用轨迹编号的奇偶（`32k-0001` → 1），相邻轨迹严格反相。

## 15. 实现细节

新模块 `srt/utils/dualpath_read_path.py`（由 `write_read_path_helper()` 生成，沿用
`write_ttft_v3_helper()` 的范式）：

```python
resolve(rid) -> "prefill" | "decode"     # 纯函数，LRU 缓存，永不抛异常
is_decode_read(rid) -> bool
decode_capable() -> bool                  # 启动期能力开关
```

五个原先读 `DUALPATH_L3_READ_PATH` 的运行时开关点已全部改为按请求决策：

| 文件 | 改法 |
|---|---|
| `managers/scheduler.py` | `dualpath_is_decode_read(req.rid)` |
| `disaggregation/prefill.py` | `if not dualpath_is_decode_read(req.rid)` |
| `disaggregation/decode.py` | `dualpath_is_decode_read(decode_req.req.rid)` |
| `disaggregation/decode_kvcache_offload_manager.py` | `dualpath_is_decode_read(req.rid)` |
| `disaggregation/mooncake/conn.py` | `dualpath_decode_capable()`（启动期能力，非按请求） |

构建期断言会遍历这五个文件，若任何一个仍出现 `DUALPATH_L3_READ_PATH` 就直接失败。

TTFT V3 记录同时输出 `configured_path`（run 模式）与 `path`（该请求实际走的路径），
`run.json` 增加 `dynamic_path_policy`。

无法解析的 rid（warmup、`HEALTH_CHECK_*`）回退到 `prefill` 而非 fail-closed——让健康检查
失败会直接打断 run，代价远大于收益。
