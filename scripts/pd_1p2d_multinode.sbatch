#!/usr/bin/env bash
#SBATCH --job-name=qwen32b-pd-1p2d
#SBATCH --partition=GAIA
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/lustre/users/sezhou/dual-path/logs/pd-1p2d/slurm-%j.out

set -euo pipefail

CODE_HOME=${CODE_HOME:-/labhome/sezhou/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/mnt/lustre/users/sezhou/dual-path}
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3-mooncake0.3.9.sqsh}
MODEL_DIR=${MODEL_DIR:-$RUNTIME_HOME/models/Qwen2.5-32B}
LOG_DIR=${LOG_DIR:-$RUNTIME_HOME/logs/pd-1p2d/$SLURM_JOB_ID}
IB_DEVICE=${IB_DEVICE:-mlx5_0}
PREFILL_PORT=${PREFILL_PORT:-30000}
DECODE_PORT=${DECODE_PORT:-30000}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
ROUTER_PORT=${ROUTER_PORT:-8000}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-1200}
SMOKE_REQUESTS=${SMOKE_REQUESTS:-6}

mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ ${#hosts[@]} -ne 3 ]]; then
  echo "Expected exactly three nodes, got: ${hosts[*]}" >&2
  exit 1
fi

export PREFILL_HOST=${hosts[0]}
export DECODE0_HOST=${hosts[1]}
export DECODE1_HOST=${hosts[2]}
export DECODE_HOSTS="$DECODE0_HOST $DECODE1_HOST"
export MODEL_DIR LOG_DIR IB_DEVICE PREFILL_PORT DECODE_PORT
export BOOTSTRAP_PORT ROUTER_PORT SERVER_TIMEOUT

mkdir -p "$LOG_DIR"
printf '%s\n' "$PREFILL_HOST" >"$LOG_DIR/prefill_host"
printf '%s\n' "$DECODE0_HOST" >"$LOG_DIR/decode0_host"
printf '%s\n' "$DECODE1_HOST" >"$LOG_DIR/decode1_host"

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

echo "Job $SLURM_JOB_ID: prefill=$PREFILL_HOST decode0=$DECODE0_HOST decode1=$DECODE1_HOST"

srun --nodes=1 --ntasks=1 --nodelist="$PREFILL_HOST" \
  --gres=gpu:8 --exclusive --exact --mpi=none \
  --container-image="$IMAGE" \
  --container-mounts="/mnt/lustre:/mnt/lustre,$CODE_HOME:/workspace/dual-path" \
  --container-workdir=/workspace/dual-path \
  --no-container-mount-home \
  bash scripts/pd_prefill_router.sh &
step_pids+=("$!")
prefill_step_pid=$!

srun --nodes=1 --ntasks=1 --nodelist="$DECODE0_HOST" \
  --gres=gpu:8 --exclusive --exact --mpi=none \
  --container-image="$IMAGE" \
  --container-mounts="/mnt/lustre:/mnt/lustre,$CODE_HOME:/workspace/dual-path" \
  --container-workdir=/workspace/dual-path \
  --no-container-mount-home \
  env DECODE_ID=0 DECODE_HOST="$DECODE0_HOST" bash scripts/pd_decode.sh &
step_pids+=("$!")
decode0_step_pid=$!

srun --nodes=1 --ntasks=1 --nodelist="$DECODE1_HOST" \
  --gres=gpu:8 --exclusive --exact --mpi=none \
  --container-image="$IMAGE" \
  --container-mounts="/mnt/lustre:/mnt/lustre,$CODE_HOME:/workspace/dual-path" \
  --container-workdir=/workspace/dual-path \
  --no-container-mount-home \
  env DECODE_ID=1 DECODE_HOST="$DECODE1_HOST" bash scripts/pd_decode.sh &
step_pids+=("$!")
decode1_step_pid=$!

wait_for_url decode0 "http://$DECODE0_HOST:$DECODE_PORT"
wait_for_url decode1 "http://$DECODE1_HOST:$DECODE_PORT"
wait_for_url router "http://$PREFILL_HOST:$ROUTER_PORT"

for request_id in $(seq 0 $((SMOKE_REQUESTS - 1))); do
  response_file="$LOG_DIR/response-$request_id.json"
  curl --fail --silent --show-error --max-time 300 \
    -X POST "http://$PREFILL_HOST:$ROUTER_PORT/generate" \
    -H "Content-Type: application/json" \
    -d "{
      \"text\": \"def multiply_${request_id}(a, b):\\n    \\\"\\\"\\\"Return the product of two integers.\\\"\\\"\\\"\\n\",
      \"sampling_params\": {
        \"temperature\": 0,
        \"max_new_tokens\": 24
      }
    }" >"$response_file"
done

python - "$LOG_DIR" "$SMOKE_REQUESTS" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
request_count = int(sys.argv[2])
for request_id in range(request_count):
    path = root / f"response-{request_id}.json"
    with path.open() as f:
        response = json.load(f)
    text = response.get("text")
    if not isinstance(text, str) or not text.strip():
        raise SystemExit(f"PD response {request_id} has no generated text: {response}")
print(f"MULTINODE_1P2D_RESPONSES_OK count={request_count}")
PY

python - "$LOG_DIR" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
required = ("installTransport, type=rdma", "Found 1 HCAs", "RDMA device: mlx5_0")
for name in ("prefill.log", "decode-0.log", "decode-1.log"):
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

for name in ("decode-0.log", "decode-1.log"):
    text = (root / name).read_text(errors="replace")
    if "Decode batch" not in text:
        raise SystemExit(f"{name} did not process any decode batch")

print("MULTINODE_1P2D_BOTH_DECODES_USED")
print("MULTINODE_1P2D_RDMA_LOGS_OK")
PY

for pid in "$prefill_step_pid" "$decode0_step_pid" "$decode1_step_pid"; do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "A server step exited unexpectedly: $pid" >&2
    exit 1
  fi
done

echo "MULTINODE_1P2D_SMOKE_COMPLETE"
