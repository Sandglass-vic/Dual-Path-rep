# 项目目录与运行环境约定

> 返回 [文档索引](README.md)

## pza100 当前目录

```text
/data/home/yongxiao/dual-path/       Git 根目录
├── docs/
├── scripts/
└── results/                         仓库侧图表/汇总（gitignore）

/data/home/yongxiao/mydata/          g5 本地、g6 经 IPoIB NFS 共享
├── images/
├── models/
├── test/<allocation-id>/
├── traces/
└── test/dataset/
```

```bash
export CODE_HOME=/data/home/yongxiao/dual-path
export RUNTIME_HOME=/data/home/yongxiao/mydata
export IMAGE_DIR=$RUNTIME_HOME/images
export MODEL_DIR=$RUNTIME_HOME/models/Qwen2.5-32B
export DATASET_DIR=$RUNTIME_HOME/test/dataset
export TRACE_DIR=$RUNTIME_HOME/traces
export REPO_RESULT_DIR=$CODE_HOME/results
```

g5 是 NFS 源和登录/计算节点；g6 的共享目录依赖 g5 与 `ibs3`。不要把 GAIA
Lustre 的带宽、stripe 或失效隔离假设套用到该 NFS 路径。

## GAIA 历史目录划分

项目分为“随时可见的 Git 代码仓库”和“仅计算节点可见的运行数据目录”：

```text
/labhome/sezhou/dual-path/          Git 根目录，登录节点随时可查
├── docs/                           设计、平台和实验记录
├── scripts/
│   ├── benchmark/                  数据生成、replay、collect 与 summarize
│   ├── smoke/                      Mooncake Store/DFS smoke
│   ├── plot/                       画图脚本
│   ├── infra/                      SGLang patch 与镜像构建
│   ├── pd_alloc.sh                 默认值、salloc、节点角色与共享配置
│   ├── pd_benchmark.sh             节点自动识别与启动入口
│   ├── pd_plot.sh                  独立后处理入口
│   └── tests/                      runtime 辅助脚本和测试
├── results/                        仓库侧图表/汇总（REPO_RESULT_DIR；gitignore）
└── ...

/mnt/lustre/users/sezhou/dual-path/ 计算节点运行数据根目录
├── images/
├── models/
├── test/<allocation-id>/
│   ├── pd-servers/
│   │   ├── kv-cache/{file,l3-sync}/
│   │   └── *.log
│   └── <TASK_NAME>/
│       └── logs/                  # replay 原始数据、client 日志、画图输入
├── traces/
└── test/dataset/                  fake trajectory 数据集
```

**同名不同义**：`/labhome/sezhou/dual-path/results/` 只存 `pd_plot.sh` 生成的
CSV/SVG/JSON；Lustre 的 `test/<allocation-id>/<TASK_NAME>/logs/` 存 replay 原始
`rounds.jsonl` 等。二者通过 `REPO_RESULT_DIR` 与 `RESULT_DIR` 区分。旧名
`artifacts/` 已弃用。

allocation 与 client task 是两个命名层级。allocation ID 标识长期 server、server
日志和共享 L3 KV；`TASK_NAME` 标识一次 client replay、对应日志/结果和后处理产物。
Lustre 原始结果和 labhome 整理结果使用相同 task 层级：

```text
<task>/
├── JCT.svg / JCT.json                    # labhome task 级对比
├── data/ / csv/                          # task 级聚合
├── 32K/
│   └── agents-200/
│       ├── rep-1/
│       │   ├── data/                     # JSON/JSONL 原始及明细
│       │   ├── csv/                      # 处理后的表格
│       │   ├── longest-trajectory/       # 最长 agent 的逐 turn 图
│       │   └── all-trajectories/         # 全 agent 分布与 TTFT breakdown
│       └── rep-2/
└── 64K/...
```

Lustre 侧 `<task>/<dataset>/agents-<N>/rep-<R>/` 保存 replay 原始文件；client
Router 日志和 replay 原始数据位于 `test/<allocation-id>/<task>/logs/`。长期
Prefill/Decode 日志位于 `test/<allocation-id>/pd-servers/`，不会复制到 labhome。

所有代码和小型配置必须放在 `/labhome/sezhou/dual-path` 的子目录并由 Git 管理，确保登录节点随时可读。镜像、模型权重、SWE-bench checkout/sandbox、KV-cache、大型 trace 和原始结果放在 Lustre，不提交 Git。登录节点不能依赖 `/mnt/lustre` 或 `/labhome/sezhou/lustre` 可见；访问和生成运行数据时必须进入计算节点。

## GAIA 历史环境变量

```bash
export CODE_HOME=/labhome/sezhou/dual-path
export RUNTIME_HOME=/mnt/lustre/users/sezhou/dual-path
export IMAGE_DIR=$RUNTIME_HOME/images
export MODEL_DIR=$RUNTIME_HOME/models/Qwen2.5-32B
export SERVER_CACHE_DIR=$RUNTIME_HOME/test/<allocation-id>/pd-servers/kv-cache/file
export DATASET_DIR=$RUNTIME_HOME/test/dataset
export TRACE_DIR=$RUNTIME_HOME/traces
export RESULT_DIR=$RUNTIME_HOME/test/<allocation-id>/<TASK_NAME>/logs  # Lustre runtime
export REPO_RESULT_DIR=$CODE_HOME/results        # labhome 汇总/图表
export LOG_DIR=$RUNTIME_HOME/test/<allocation-id>/<TASK_NAME>/logs
```

`SERVER_CACHE_DIR` 是长期 Prefill/Decode server 使用的 allocation 级 HiCache L3
目录，不属于单次 client task。当前默认结构为：

```text
$RUNTIME_HOME/test/<allocation-id>/
├── pd-servers/kv-cache/
│   ├── file/
│   └── l3-sync/
└── <TASK_NAME>/logs/
```

client 完成或失败时只清理本机 Router/replay；不会自动删除该目录，因此同一
allocation 上的后续 client run 默认复用已经落盘的 KV。

需要 cold start 时，必须先停止 Prefill/Decode 并释放 allocation，再删除整个
allocation 目录，然后重新启动 server：

```bash
rm -rf "$RUNTIME_HOME/test/<allocation-id>/pd-servers/kv-cache"
```

`file/` 保存持久化 KV，`l3-sync/` 保存 Decode 各 TP rank 的完成 marker。不要在
server 运行期间直接删除 `file/`，否则可能与 HiCache metadata 或异步 backup 竞争。
