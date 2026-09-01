#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/mnt/lustre/users/sezhou/dual-path/models/Qwen2.5-32B}
LOG_DIR=${LOG_DIR:-/mnt/lustre/users/sezhou/dual-path/logs/pd-smoke}
IB_DEVICE=${IB_DEVICE:-mlx5_0}
PREFILL_PORT=${PREFILL_PORT:-30000}
DECODE_PORT=${DECODE_PORT:-30001}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
ROUTER_PORT=${ROUTER_PORT:-8000}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-1200}

mkdir -p "$LOG_DIR"

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_health() {
  local name=$1
  local url=$2
  local pid=$3
  local deadline=$((SECONDS + SERVER_TIMEOUT))

  until curl --fail --silent --max-time 2 "$url/health" >/dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited before becoming healthy" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for $name at $url" >&2
      return 1
    fi
    sleep 5
  done
  echo "$name is healthy at $url"
}

echo "Starting Qwen2.5-32B prefill server on GPUs 0-3"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --served-model-name qwen2.5-32b \
  --host 0.0.0.0 \
  --port "$PREFILL_PORT" \
  --tp-size 4 \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --context-length 4096 \
  --mem-fraction-static 0.75 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
  --disaggregation-ib-device "$IB_DEVICE" \
  >"$LOG_DIR/prefill.log" 2>&1 &
pids+=("$!")
prefill_pid=$!

echo "Starting Qwen2.5-32B decode server on GPUs 4-7"
CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --served-model-name qwen2.5-32b \
  --host 0.0.0.0 \
  --port "$DECODE_PORT" \
  --tp-size 4 \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --context-length 4096 \
  --mem-fraction-static 0.75 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "$IB_DEVICE" \
  >"$LOG_DIR/decode.log" 2>&1 &
pids+=("$!")
decode_pid=$!

wait_for_health prefill "http://127.0.0.1:$PREFILL_PORT" "$prefill_pid"
wait_for_health decode "http://127.0.0.1:$DECODE_PORT" "$decode_pid"

echo "Starting PD router"
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://127.0.0.1:$PREFILL_PORT" "$BOOTSTRAP_PORT" \
  --decode "http://127.0.0.1:$DECODE_PORT" \
  --policy round_robin \
  --host 0.0.0.0 \
  --port "$ROUTER_PORT" \
  >"$LOG_DIR/router.log" 2>&1 &
pids+=("$!")
router_pid=$!

wait_for_health router "http://127.0.0.1:$ROUTER_PORT" "$router_pid"

response_file="$LOG_DIR/response.json"
curl --fail --silent --show-error --max-time 300 \
  -X POST "http://127.0.0.1:$ROUTER_PORT/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Write a Python function that returns the sum of two integers.",
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
print("PD_RESPONSE_OK")
print(text)
PY

echo "PD_SMOKE_COMPLETE"
