# 实验记录、Git 与安全规范

> 返回 [文档索引](README.md)

## 平台基线与实验记录模板

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

## Git 管理

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

## 安全说明

参考脚本中包含明文 NGC API Key。该凭据已经暴露在普通文本中，应视为泄露：

1. 在 NGC API Keys 页面立即吊销；
2. 生成新 key；
3. 仅写入 `$HOME/.config/enroot/.credentials`；
4. 执行 `chmod 600`；
5. 检查 shell history、作业日志和版本库中是否还有副本；
6. 不在本文或后续脚本中填写真实 token。

内部 GitLab access token 采用相同规则，并确保 `.credentials`、`.env` 和本地 secret 文件被 `.gitignore` 排除。

本文不会记录登录节点 IP、个人凭据或未经当前环境验证的网络拓扑宣传资料。对于会影响结论的平台信息，以作业节点实时采集结果为准。
