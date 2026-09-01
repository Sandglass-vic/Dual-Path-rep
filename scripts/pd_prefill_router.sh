#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR is required}"
: "${LOG_DIR:?LOG_DIR is required}"
: "${PREFILL_HOST:?PREFILL_HOST is required}"

if [[ -z ${PREFILL_HOSTS:-} ]]; then
  PREFILL_HOSTS=$PREFILL_HOST
fi
read -r -a prefill_hosts <<<"$PREFILL_HOSTS"
if [[ ${#prefill_hosts[@]} -eq 0 ]]; then
  echo "At least one prefill host is required" >&2
  exit 1
fi

if [[ -z ${DECODE_HOSTS:-} ]]; then
  : "${DECODE_HOST:?DECODE_HOST or DECODE_HOSTS is required}"
  DECODE_HOSTS=$DECODE_HOST
fi
read -r -a decode_hosts <<<"$DECODE_HOSTS"
if [[ ${#decode_hosts[@]} -eq 0 ]]; then
  echo "At least one decode host is required" >&2
  exit 1
fi

IB_DEVICE=${IB_DEVICE:-mlx5_0}
PREFILL_PORT=${PREFILL_PORT:-30000}
DECODE_PORT=${DECODE_PORT:-30000}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
ROUTER_PORT=${ROUTER_PORT:-8000}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-1200}
if [[ -n ${PREFILL_ID:-} ]]; then
  PREFILL_LOG="$LOG_DIR/prefill-$PREFILL_ID.log"
else
  PREFILL_LOG="$LOG_DIR/prefill.log"
fi

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

echo "Starting TP8 prefill on $PREFILL_HOST"
python -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --served-model-name qwen2.5-32b \
  --host 0.0.0.0 \
  --port "$PREFILL_PORT" \
  --tp-size 8 \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --context-length 4096 \
  --mem-fraction-static 0.75 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
  --disaggregation-ib-device "$IB_DEVICE" \
  >"$PREFILL_LOG" 2>&1 &
pids+=("$!")
prefill_pid=$!

wait_for_health prefill "http://127.0.0.1:$PREFILL_PORT" "$prefill_pid"

decode_args=()
for decode_host in "${decode_hosts[@]}"; do
  decode_args+=(--decode "http://$decode_host:$DECODE_PORT")
done

prefill_args=()
for prefill_host in "${prefill_hosts[@]}"; do
  prefill_args+=(--prefill "http://$prefill_host:$PREFILL_PORT" "$BOOTSTRAP_PORT")
done

echo "Starting router on $PREFILL_HOST for prefill hosts: ${prefill_hosts[*]}; decode hosts: ${decode_hosts[*]}"
python -m sglang_router.launch_router \
  --pd-disaggregation \
  "${prefill_args[@]}" \
  "${decode_args[@]}" \
  --prefill-policy round_robin \
  --decode-policy round_robin \
  --host 0.0.0.0 \
  --port "$ROUTER_PORT" \
  >"$LOG_DIR/router.log" 2>&1 &
pids+=("$!")
router_pid=$!

wait_for_health router "http://127.0.0.1:$ROUTER_PORT" "$router_pid"
echo "PREFILL_ROUTER_READY"

wait "$prefill_pid" "$router_pid"
