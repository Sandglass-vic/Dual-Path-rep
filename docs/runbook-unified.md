# Unified 推理运行手册

> 返回 [文档索引](README.md)

## Phase 0：当前镜像与模型准备

目的：不引入 PD、HiCache 和 Mooncake，只验证镜像与 Qwen2.5-32B。

1. 申请 1 个节点；
2. 验证容器内可见 8 张 H100 和 Lustre；
3. 将一个 Qwen2.5-32B checkpoint 完整下载到 `$MODEL_DIR`；
4. 检查 tokenizer、config 和所有 weight shards；
5. 记录模型 revision、镜像 SHA-256 和 SGLang 版本。

模型约 65 GB，建议直接下载到 Lustre。以下 Python 方式不依赖 `hf` CLI 是否存在；正式运行时将 `<commit-sha>` 替换为固定的 Hugging Face revision：

```bash
srun --partition=GAIA --nodes=1 --ntasks=1 --gres=gpu:1 --time=01:00:00 \
  --container-image="$IMAGE_DIR/sglang:26.01-py3.sqsh" \
  --container-mounts=/mnt/lustre:/mnt/lustre \
  --no-container-mount-home \
  bash -lc '
    python - <<"PY"
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-32B",
    revision="<commit-sha>",
    local_dir="'"$MODEL_DIR"'",
)
PY
  '
```

若先验证未来 SWE-bench 所需的对话模型，将 repo id 和目录改为 `Qwen/Qwen2.5-32B-Instruct`。两种 checkpoint 不得写入同一目录。

当前 base 模型已下载完成：

```text
repo：Qwen/Qwen2.5-32B
revision：1818d35814b8319459f4bd55ed1ac8709630f003
目录：/mnt/lustre/users/sezhou/dual-path/models/Qwen2.5-32B
占用：约 62 GiB
```

验收条件：

- 容器可稳定启动；
- 8 张 GPU 可见；
- `python -c 'import sglang, torch'` 成功；
- 模型目录不依赖运行时联网补文件；
- 模型 revision 和文件校验信息已保存到 `manifests/`。

## Phase 1：单节点统一服务

目的：排除模型、长度、显存和镜像问题，建立 non-PD 基线。

推荐先使用 8 卡 TP，以减少权重显存压力并与论文 SGL(MC) 的 Qwen 配置一致。完整 Slurm/Enroot 启动命令：

```bash
srun --partition=GAIA \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:8 \
  --time=01:00:00 \
  --container-image="$IMAGE_DIR/sglang:26.01-py3.sqsh" \
  --container-mounts="/mnt/lustre:/mnt/lustre,$CODE_HOME:/workspace/dual-path" \
  --container-workdir=/workspace/dual-path \
  --no-container-mount-home \
  bash -lc '
    python -m sglang.launch_server \
      --model-path "'"$MODEL_DIR"'" \
      --served-model-name qwen2.5-32b \
      --host 0.0.0.0 \
      --port 8000 \
      --tp-size 8 \
      --dtype bfloat16 \
      --kv-cache-dtype auto \
      --context-length 4096 \
      --mem-fraction-static 0.80 \
      --trust-remote-code \
      --enable-metrics
  '
```

第一次不要启用 HiCache、Mooncake、PD、FP8 KV、speculative decoding 或自定义 attention backend。按以下顺序验证：

1. `context-length=4096`，单请求、确定性生成；
2. 同一 prompt 连续请求 10 次，确认无失败且输出稳定；
3. `context-length=32768`，单请求；
4. 当前目标完成后才扩到 64K 和并发请求。

基础检查：

```bash
curl -f http://<compute-host>:8000/health
curl -s http://<compute-host>:8000/v1/models

curl -s http://<compute-host>:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-32b",
    "prompt": "Write a Python function that returns the sum of two integers.",
    "temperature": 0,
    "max_tokens": 64
  }'
```

若使用 Instruct checkpoint，再增加 `/v1/chat/completions` 测试；base checkpoint 的主要正确性入口使用 completions，避免 chat template 差异干扰。

验收条件：

- `/health` 和 `/v1/models` 成功；
- completion 返回非空文本，finish reason 正常；
- `temperature=0` 的重复请求 token 序列一致；
- 4K 与 32K 输入均能完成，无 OOM、hang、NaN 或请求错误；
- 8 张 GPU 均加载模型且有合理显存占用；
- 保存完整启动参数、服务日志、模型 revision 和 GPU 峰值显存。
