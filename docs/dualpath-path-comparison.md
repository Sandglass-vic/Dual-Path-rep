# DualPath Lustre read path 三路径对比：32K / 8 agent / TP8 / O_DIRECT

Job `69352`，1P1D、Qwen2.5-32B、TP8、`DUAL_PATH_O_DIRECT_READ=1`、
`HICACHE_SIZE_GB=0`（host pool 由 `HICACHE_RATIO=1.35` 导出）、
`alternate_per_turn`。三个 run 共用同一 allocation 与同一镜像
`sglang-file-l3-dual-staging-69327.sqsh`，各 208 请求、8 条 trajectory、26 轮。

对比图：`results/pd-case-comparison/dualpath-8agent-comparison.svg`（产物不入库，复现命令见文末）

## 结果总览

| | Prefill path | Decode path | **Dynamic** |
|---|---:|---:|---:|
| trajectory_jct | 170.09 s | 175.38 s | **110.87 s** |
| TTFT mean | 4.135 s | 4.407 s | **1.657 s** |
| TTFT p50 | 4.254 s | 4.564 s | **1.363 s** |
| TTFT p90 | 7.992 s | 8.277 s | **2.923 s** |
| TTFT p99 | 8.717 s | 9.054 s | **5.130 s** |
| TPOT mean | **9.09 ms** | 8.99 ms | 10.53 ms |
| 请求延迟 mean | 6.317 s | 6.569 s | **4.137 s** |
| Lustre 读聚合带宽 | 7.59 GB/s | 7.58 GB/s | **14.79 GB/s** |
| decode 吞吐 | 293 tok/s | 285 tok/s | **443 tok/s** |

Dynamic 的 TTFT 快 **2.49x**、JCT 快 **1.53x**，代价是 TPOT 慢 16%。
两条静态路径几乎等价（TTFT 差 6.6%），说明 GAIA 上 Prefill 侧与 Decode 侧
读取 Lustre 的能力对称。

## 1. KV 读数据量与带宽

三个 run 读取的 KV 总量完全相同，dynamic 精确对半分，因此是干净对照：

| run / 侧 | 请求数 | tokens | GiB | 中位 GB/s |
|---|---:|---:|---:|---:|
| Prefill path / Prefill 侧 | 200 | 3,874,176 | 945.8 | 7.59 |
| Decode path / Decode 侧 | 200 | 3,874,176 | 945.8 | 7.58 |
| Dynamic / Prefill 侧 | 100 | 1,937,472 | 473.0 | 7.54 |
| Dynamic / Decode 侧 | 100 | 1,936,704 | 472.8 | 7.24 |

**每个请求只在一侧读取**：208 个请求中两侧同时读取的有 0 个，100 个走
Prefill、100 个走 Decode、8 个是 turn 1 无前缀。因此单请求读取耗时几乎不变
（dynamic 0.708 s vs 静态 0.681 s，仅慢 4%），dynamic **没有加速任何一次读取**。

变的是系统吞吐：两侧并发读取，聚合带宽 `7.54 + 7.24 = 14.79 GB/s`，是单路径的
1.95x，接近理论 2x，说明两条路径几乎完全独立、不争抢 Lustre。

瓶颈不在 Lustre：单节点 TP8 的 8 条读取流只拿到 7.6 GB/s（7.1 GiB/s），而
[platform.md](platform.md) 的 fio 标定单节点 8 jobs 即可读 10.77 GiB/s、
64 jobs 达 32.37 GiB/s。读取并行度由 **TP rank 数**决定，不是 agent 数，所以
加 agent 不会提高带宽，加节点才会。

这一点由独立的微基准交叉验证：`io-test` Job `69346`/`69347`
（见 [io-test/RESULTS.md](../io-test/RESULTS.md)）在两个空节点上只跑文件 I/O，
不带引擎和 GPU，测得 O_DIRECT 双节点聚合为单节点的 **1.935x**、每节点保持率
97.5%。与本文在真实推理负载下测到的 1.95x 一致，说明两条路径确实互不争抢，
dynamic 的聚合带宽不是测量口径造成的假象。

带宽口径：`tokens x 262144 B / lustre_read_seconds / 1e9`，即 **GB/s**（字段名
`*_effective_bandwidth_gbps` 实为字节，非 Gbit/s），分母是纯读取不含 scatter。
取**中位数**而非 `sum(tokens)/sum(seconds)`：后者是时间加权，少数慢请求会把
结果压低约 30%。

## 2. TTFT 为什么快 2.49 倍：全部来自 queue wait

| 阶段 [秒] | Prefill | Decode | Dynamic | Dyn−Pre |
|---|---:|---:|---:|---:|
| **queue wait** | **3.159** | 3.435 | **0.617** | **−2.543** |
| prefill 侧读取 | 0.655 | 0 | 0.331 | −0.324 |
| decode 侧读取 | 0 | 0.656 | 0.350 | +0.350 |
| prefill scatter | 0.130 | 0 | 0.066 | −0.064 |
| decode scatter | 0 | 0.130 | 0.067 | +0.067 |
| prefill 计算 | 0.054 | 0.045 | 0.050 | −0.004 |
| P→D 传输 | 0.028 | 0.006 | 0.026 | −0.003 |
| 其他 + misc | 0.108 | 0.134 | 0.126 | +0.018 |
| **TTFT** | **4.135** | 4.407 | **1.657** | **−2.477** |

TTFT 总差 2.477 s，queue wait 一项就贡献 2.543 s（103%）。读取相关各项两两抵消
（Prefill 侧省 0.324，Decode 侧加 0.350），净效应约为零。

**dynamic 图上两段读取不能读作"一个请求先后读了两次"。** 每请求只读一侧，
另一侧恒为 0，因此两段均值之和恰等于每请求实际读取耗时的均值
（0.3307 + 0.3500 = 0.6808，与直接按请求求均值完全相等）。数值正确，但视觉
上暗示了串行，解读时需注意。

### queue wait 为什么降 5.1 倍

不能用 M/M/1 的 `W ∝ ρ/(1−ρ)` 解释：这是**闭环**系统，8 个 agent 各自串行
发请求，在途请求数被钉死，到达率是结果而非输入；引擎还会批处理，也不是单
服务器。

**Prefill 引擎串行，容量约 1.16 req/s。** 把引擎为一个请求独占的阶段相加得服务
时间 `S_p`（before_read + lustre_read + scatter + H2D + computation）。按
Little's Law，处于服务中的请求数为 `X · S_p`：

| | 吞吐 X | S_p | X·S_p | 容量 1/S_p | 利用率 |
|---|---:|---:|---:|---:|---:|
| 静态 4 agent | 1.044 | 0.850 | 0.887 | 1.177 | 89% |
| 静态 8 agent | 1.197 | 0.866 | **1.037** | 1.155 | **104%** |
| Dynamic 8 agent | 1.815 | 0.462 | 0.838 | 2.166 | 84% |

`X·S_p` 全部落在 1 附近，证实引擎**同时只服务一个请求**。静态两档的 `S_p` 几乎
相同，是同一条硬容量线；8 agent 实测 1.197 已顶到甚至略超（104%，超出部分是
chunked prefill 的轻微重叠），即**静态 8 agent 处于饱和**。Dynamic 把一半请求的
读取卸到对侧，`S_p` 降到 0.462 s，容量抬到 2.166 req/s，重新脱离饱和。

**queue 是恒等式里的残差。** 闭环下 `queue = N/X − S_total` 严格成立
（三个 case 误差均为 0.0000）。4→8 agent 的账：

```text
agent 数    4     →  8        x2.00
吞吐 X      1.044 →  1.197    x1.15   <- 撞上容量 1.155，涨不动
N/X         3.831 →  6.681    x1.74   (+2.85 s)
服务时间    3.003 →  3.521    x1.17   (+0.52 s)
queue       0.828 →  3.159    x3.82   (+2.33 s)
```

N 翻倍而 X 只能涨 15%，N/X 多出的 2.85 s 里服务阶段只吸收 0.52 s，剩下 2.33 s
全部压进队列；queue 基数只有 0.828 s，吸收 2.33 s 即 3.8 倍。**超线性不是特殊的
排队效应，而是小基数项承担了大项增量的全部差额。** 引擎若未饱和、X 能跟着 N 涨，
N/X 不变，queue 就不会涨。

Dynamic 的反向操作：容量翻倍后 N/X 从 6.681 降到 4.407（省 2.27 s），服务时间反
增 0.27 s（生成 batch 变大），净 2.54 s 全从队列出，正是 3.159 − 0.617。

**收益有硬上限**：queue 最低为 0 时 `X = N/S_total = 8/3.79 = 2.11 req/s`，已贴近
容量 2.166。Dynamic 当前的 1.815 距此仅剩 16%，8 agent 下这条路基本走到头；再要
提升须压缩 `S_total`，其中生成阶段占 2.48 s，是下一个大头。

### 用 Little's Law 交叉验证

先验证闭环：

| | 吞吐 X | 延迟 R | N = X·R | agent 数 |
|---|---:|---:|---:|---:|
| Prefill path | 1.197 req/s | 6.317 s | 7.56 | 8 |
| Dynamic | 1.815 req/s | 4.137 s | 7.51 | 8 |

两边都收敛到 7.5。再对每个阶段用 `L = X · T` 拆开这 7.5 个在途请求：

| 阶段 | Prefill 在途 | Dynamic 在途 |
|---|---:|---:|
| **queue wait** | **3.78** | **1.12** |
| Lustre 读取 | 0.78 | 1.24 |
| scatter | 0.16 | 0.24 |
| prefill 计算 | 0.06 | 0.09 |
| P→D 传输 | 0.03 | 0.05 |
| 生成阶段 | 2.61 | **4.50** |
| 合计 | 7.43 | 7.23 |

静态下 7.5 个在途请求有 **3.78 个（51%）单纯在排队干等**，dynamic 只剩
1.12 个（15%）。在途总数固定、各阶段服务时间基本不变，系统处理更快时多出来的
请求无处可去，只能从队列被抽走填进服务阶段——**队列是承受全部调整的残差项**，
所以它的变化幅度（5.1x）远大于吞吐本身（1.52x）。

这也给出了收益上限：queue wait 最多降到 0，dynamic 已用掉 3.78 个里的 2.66 个，
剩余空间有限。

## 3. TPOT 为什么变差 16%

**不是因为 Decode 引擎多做了读取。** 反例：Decode path 下 Decode 引擎承担
**全部** Lustre 读取，TPOT 却是三者最低（8.99 ms）。

也不是 TTFT/TPOT 边界算错，三重验证：

- 客户端 TTFT 与服务端 V3 分解完全相等（1.6574 = 1.6574），
  `accounting_error_rounds=0`，最大误差 `0.00e+00`；
- `TTST − TTFT` 即 `itl[0]`，dynamic 是 9.66 ms，**三者最小**。若 TTFT 少算，
  这个间隔会膨胀；
- dynamic 的 `itl[1:]` 均值才是 10.53 ms，说明整个生成阶段都略慢，不是开头多
  了一块。

真正原因是 **decode 批次并发度**：

| | decode 吞吐 | 单请求速率 | 平均并发 | TPOT |
|---|---:|---:|---:|---:|
| Prefill path | 293 tok/s | 110 tok/s | 2.66 | 9.09 ms |
| Decode path | 285 tok/s | 111 tok/s | 2.56 | 8.99 ms |
| Dynamic | 443 tok/s | 95 tok/s | **4.67** | 10.53 ms |

dynamic 的 TTFT 短，请求更快穿过 prefill 进入生成阶段，Decode 引擎同时生成的
请求从 2.66 涨到 4.67，batch 大了近一倍，单 token 延迟自然上升。这与上一节
Little's Law 拆出的"生成阶段在途 4.50"独立吻合。

**TPOT 变差是 dynamic 加速的结果，不是代价的根源**：换来 decode 吞吐 +51%。
Decode path 的 TPOT 最低也是同理——它 TTFT 最长、请求进入生成最慢、并发最低。

## 4. 适用条件

dynamic 的收益取决于**两条路径的存储对称性**，不是策略本身的属性。GAIA 上两侧
都直连 Lustre，单侧带宽几乎相同（7.54 vs 7.24 GB/s），分摊才有收益。

pza100 上结论相反（见 [pza100.md](pza100.md)：dynamic 6.471 s 接近
静态 Decode 6.771 s，明显劣于静态 Prefill 1.874 s）：那里 Decode 侧读取要穿
NFS，两侧不对等，分摊反而放大了慢的一侧。

收益还随并发放大。静态的 queue wait 随 agent 数超线性增长，其在 TTFT 中的占比
从 4 agent 的 46% 升到 8 agent 的 76%，而 dynamic 稳定把它压到约 1/5：

| | TTFT 加速 | 静态 queue wait | dynamic queue wait | 比值 |
|---|---:|---:|---:|---:|
| 4 agent | 1.27x | 0.828 s | 0.167 s | 4.96x |
| 8 agent | **2.49x** | 3.159 s | 0.617 s | 5.12x |

## 5. 已知问题

4 agent 的两次 dynamic run 出现过 Prefill 侧偶发读取停顿（单请求 2--5 s，
带宽跌到 0.1--2 GB/s），静态路径 200 个请求零发生。两次形态不同：一次是 8 个
rank 齐慢且集中在 turn 2--3，一次是 rank 1/3 落后且散在后期轮次。本次 8 agent
三个 run 合计只有 2 个请求低于 2.5 GB/s，问题基本未复现，倾向于是偶发争用而非
系统性缺陷。一个待验证的假设是 dynamic 下两侧同时读写同一批 KV 文件引发 Lustre
锁争用；证实需要采集 LDLM 锁统计，或做不重叠 key namespace 的对照实验。

## 复现

```bash
cd /labhome/sezhou/dual-path/scripts/plot
python3 plot_pd_case_comparison.py \
  --case 69352 1p1d-32k-a8-r1-prefill-new \
  --case 69352 1p1d-32k-a8-r1-decode-new \
  --case 69352 1p1d-32k-a8-r1-dynamic-new \
  --name dualpath-8agent-comparison \
  --title "DualPath Lustre read path comparison - 1P1D, 32K, 8 agents, TP8, O_DIRECT"
```
