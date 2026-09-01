#!/usr/bin/env bash
#SBATCH --job-name=qwen32b-pd-1p1d
#SBATCH --partition=GAIA
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/lustre/users/sezhou/dual-path/logs/pd-1p1d/slurm-%j.out

set -euo pipefail

CODE_HOME=${CODE_HOME:-/labhome/sezhou/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/mnt/lustre/users/sezhou/dual-path}
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3-mooncake0.3.9.sqsh}
MODEL_DIR=${MODEL_DIR:-$RUNTIME_HOME/models/Qwen2.5-32B}
LOG_DIR=${LOG_DIR:-$RUNTIME_HOME/logs/pd-1p1d/$SLURM_JOB_ID}
IB_DEVICE=${IB_DEVICE:-mlx5_0}
PREFILL_PORT=${PREFILL_PORT:-30000}
DECODE_PORT=${DECODE_PORT:-30000}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
ROUTER_PORT=${ROUTER_PORT:-8000}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-1200}

mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ ${#hosts[@]} -ne 2 ]]; then
  echo "Expected exactly two nodes, got: ${hosts[*]}" >&2
  exit 1
fi

export PREFILL_HOST=${hosts[0]}
export DECODE_HOST=${hosts[1]}
export MODEL_DIR LOG_DIR IB_DEVICE PREFILL_PORT DECODE_PORT
export BOOTSTRAP_PORT ROUTER_PORT SERVER_TIMEOUT

mkdir -p "$LOG_DIR"
printf '%s\n' "$PREFILL_HOST" >"$LOG_DIR/prefill_host"
printf '%s\n' "$DECODE_HOST" >"$LOG_DIR/decode_host"

step_pids=()
cleanup() {
  local pid
  for pid in "${step_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_url() {
  local name=$1
  local url=$2
  local deadline=$((SECONDS + SERVER_TIMEOUT))

  until curl --fail --silent --max-time 2 "$url/health" >/dev/null; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for $name at $url" >&2
      return 1
    fi
    sleep 5
  done
  echo "$name is healthy at $url"
}

echo "Job $SLURM_JOB_ID: prefill=$PREFILL_HOST decode=$DECODE_HOST"

srun --nodes=1 --ntasks=1 --nodelist="$PREFILL_HOST" \
  --gres=gpu:8 --exclusive --exact --mpi=none \
  --container-image="$IMAGE" \
  --container-mounts="/mnt/lustre:/mnt/lustre,$CODE_HOME:/workspace/dual-path" \
  --container-workdir=/workspace/dual-path \
  --no-container-mount-home \
  bash scripts/pd_prefill_router.sh &
step_pids+=("$!")
prefill_step_pid=$!

srun --nodes=1 --ntasks=1 --nodelist="$DECODE_HOST" \
  --gres=gpu:8 --exclusive --exact --mpi=none \
  --container-image="$IMAGE" \
  --container-mounts="/mnt/lustre:/mnt/lustre,$CODE_HOME:/workspace/dual-path" \
  --container-workdir=/workspace/dual-path \
  --no-container-mount-home \
  bash scripts/pd_decode.sh &
step_pids+=("$!")
decode_step_pid=$!

wait_for_url decode "http://$DECODE_HOST:$DECODE_PORT"
wait_for_url router "http://$PREFILL_HOST:$ROUTER_PORT"

response_file="$LOG_DIR/response.json"
curl --fail --silent --show-error --max-time 300 \
  -X POST "http://$PREFILL_HOST:$ROUTER_PORT/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Write a Python function that returns the product of two integers.",
    "sampling_params": {
      "temperature": 0,
      "max_new_tokens": 32
    }
  }' >"$response_file"

python - "$response_file" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as f:
    response = json.load(f)

text = response.get("text")
if not isinstance(text, str) or not text.strip():
    raise SystemExit(f"PD response has no generated text: {response}")
print("MULTINODE_PD_RESPONSE_OK")
print(text)
PY

python - "$LOG_DIR" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
required = ("installTransport, type=rdma", "Found 1 HCAs", "RDMA device: mlx5_0")
for name in ("prefill.log", "decode.log"):
    text = (root / name).read_text(errors="replace")
    missing = [item for item in required if item not in text]
    if missing:
        raise SystemExit(f"{name} is missing RDMA evidence: {missing}")
    fatal = [
        item
        for item in ("KVTransferError", "Segmentation fault", "OutOfMemoryError")
        if item in text
    ]
    if fatal:
        raise SystemExit(f"{name} contains fatal patterns: {fatal}")
print("MULTINODE_RDMA_LOGS_OK")
PY

if ! kill -0 "$prefill_step_pid" 2>/dev/null; then
  echo "Prefill/router step exited unexpectedly" >&2
  exit 1
fi
if ! kill -0 "$decode_step_pid" 2>/dev/null; then
  echo "Decode step exited unexpectedly" >&2
  exit 1
fi

echo "MULTINODE_PD_SMOKE_COMPLETE"
