# 项目目录与运行环境约定

> 返回 [文档索引](README.md)

## 目录划分

项目分为“随时可见的 Git 代码仓库”和“仅计算节点可见的运行数据目录”：

```text
/labhome/sezhou/dual-path/          Git 根目录，登录节点随时可查
├── docs/                           设计、平台和实验记录
├── scripts/
│   ├── plot/                       画图脚本
│   ├── infra/                      SGLang patch 与镜像构建
│   ├── pd_*_benchmark.sbatch       正式 PD 矩阵入口
│   └── collect_benchmark_metrics.py
├── results/                        仓库侧图表/汇总（REPO_RESULT_DIR；gitignore）
└── ...

/mnt/lustre/users/sezhou/dual-path/ 计算节点运行数据根目录
├── images/
├── models/
├── kv-cache/
├── traces/
├── results/                        runtime 原始结果（RESULT_DIR；不在 git 仓库内）
├── logs/
└── test/dataset/                  fake trajectory 数据集
```

**同名不同义**：labhome 的 `results/` 存 collect 产出的 CSV/SVG/fullchain；
Lustre 的 `results/` 存 replay 原始 `rounds.jsonl` 等。二者通过
`REPO_RESULT_DIR` 与 `RESULT_DIR` 区分。旧名 `artifacts/` 已弃用。

所有代码和小型配置必须放在 `/labhome/sezhou/dual-path` 的子目录并由 Git 管理，确保登录节点随时可读。镜像、模型权重、SWE-bench checkout/sandbox、KV-cache、大型 trace 和原始结果放在 Lustre，不提交 Git。登录节点不能依赖 `/mnt/lustre` 或 `/labhome/sezhou/lustre` 可见；访问和生成运行数据时必须进入计算节点。

## 推荐环境变量

```bash
export CODE_HOME=/labhome/sezhou/dual-path
export RUNTIME_HOME=/mnt/lustre/users/sezhou/dual-path
export IMAGE_DIR=$RUNTIME_HOME/images
export MODEL_DIR=$RUNTIME_HOME/models/Qwen2.5-32B
export CACHE_DIR=$RUNTIME_HOME/kv-cache
export DATASET_DIR=$RUNTIME_HOME/test/dataset
export TRACE_DIR=$RUNTIME_HOME/traces
export RESULT_DIR=$RUNTIME_HOME/results          # Lustre runtime
export REPO_RESULT_DIR=$CODE_HOME/results        # labhome 汇总/图表
export LOG_DIR=$RUNTIME_HOME/logs
```
