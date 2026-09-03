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
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-4096}
PD_TRANSFER_BACKEND=${PD_TRANSFER_BACKEND:-mooncake}
HICACHE_STORAGE_BACKEND=${HICACHE_STORAGE_BACKEND:-file}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-}
DISABLE_OVERLAP_SCHEDULE=${DISABLE_OVERLAP_SCHEDULE:-0}
if [[ -n ${PREFILL_ID:-} ]]; then
  PREFILL_LOG="$LOG_DIR/prefill-$PREFILL_ID.log"
else
  PREFILL_LOG="$LOG_DIR/prefill.log"
fi
ROUTER_LOG="$LOG_DIR/router.log"
if [[ -n ${RUN_ID:-} ]]; then
  PREFILL_LOG="${PREFILL_LOG%.log}-$RUN_ID.log"
  ROUTER_LOG="${ROUTER_LOG%.log}-$RUN_ID.log"
fi

hicache_args=()
if [[ ${ENABLE_HICACHE_L3:-0} == 1 ]]; then
  : "${CACHE_DIR:?CACHE_DIR is required when ENABLE_HICACHE_L3=1}"
  if [[ "$HICACHE_STORAGE_BACKEND" == file ]]; then
    export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=$CACHE_DIR
    hicache_extra_config='{"prefetch_threshold":64,"prefetch_timeout_base":10,"prefetch_timeout_per_ki_token":2}'
  elif [[ "$HICACHE_STORAGE_BACKEND" == mooncake ]]; then
    : "${MOONCAKE_MASTER:?MOONCAKE_MASTER is required for Mooncake HiCache}"
    : "${MOONCAKE_TE_META_DATA_SERVER:?MOONCAKE_TE_META_DATA_SERVER is required}"
    : "${MOONCAKE_OFFLOAD_FILE_STORAGE_PATH:?Mooncake FileStorage path is required}"
    mkdir -p "$MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"
    hicache_extra_config=$(python - <<'PY'
import json
import os

print(json.dumps({
    "master_server_address": os.environ["MOONCAKE_MASTER"],
    "metadata_server": os.environ["MOONCAKE_TE_META_DATA_SERVER"],
    "local_hostname": os.environ["PREFILL_HOST"],
    "global_segment_size": os.environ.get(
        "MOONCAKE_GLOBAL_SEGMENT_SIZE", "32gb"
    ),
    "protocol": os.environ.get("MOONCAKE_PROTOCOL", "rdma"),
    "device_name": os.environ.get("MOONCAKE_DEVICE", ""),
    "check_server": False,
    "replica_num": 1,
    "dfs_replica_num": 1,
    "enable_ssd_offload": True,
    "ssd_offload_path": os.environ["MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"],
}))
PY
)
  else
    echo "Unsupported HICACHE_STORAGE_BACKEND=$HICACHE_STORAGE_BACKEND" >&2
    exit 2
  fi
  hicache_args=(
    --page-size 64
    --enable-metrics
    --enable-hierarchical-cache
    --hicache-ratio "${HICACHE_RATIO:-1.2}"
    --hicache-size 0
    --hicache-mem-layout page_first_direct
    --hicache-io-backend direct
    --hicache-write-policy write_through
    --hicache-storage-backend "$HICACHE_STORAGE_BACKEND"
    --hicache-storage-prefetch-policy wait_complete
    --hicache-storage-backend-extra-config
    "$hicache_extra_config"
  )
fi

timing_args=()
if [[ ${ENABLE_BENCHMARK_TIMING:-0} == 1 ]]; then
  timing_args+=(--enable-request-time-stats-logging)
fi

schedule_args=()
if [[ -n "$CHUNKED_PREFILL_SIZE" ]]; then
  schedule_args+=(--chunked-prefill-size "$CHUNKED_PREFILL_SIZE")
fi
if [[ "$DISABLE_OVERLAP_SCHEDULE" == 1 ]]; then
  schedule_args+=(--disable-overlap-schedule)
elif [[ "$DISABLE_OVERLAP_SCHEDULE" != 0 ]]; then
  echo "DISABLE_OVERLAP_SCHEDULE must be 0 or 1" >&2
  exit 2
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

if [[ ${START_MOONCAKE_MASTER:-0} == 1 ]]; then
  : "${MOONCAKE_DFS_ROOT_DIR:?MOONCAKE_DFS_ROOT_DIR is required}"
  mkdir -p "$MOONCAKE_DFS_ROOT_DIR"
  echo "Starting Mooncake master on $PREFILL_HOST"
  mooncake_master \
    --rpc_address=0.0.0.0 \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port="${MOONCAKE_METADATA_PORT:-8080}" \
    >"$LOG_DIR/mooncake-master${RUN_ID:+-$RUN_ID}.log" 2>&1 &
  pids+=("$!")
  master_pid=$!
  python - "$PREFILL_HOST" "${MOONCAKE_MASTER_PORT:-50051}" "$master_pid" <<'PY'
import os
import socket
import sys
import time

host, port, pid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1):
            print(f"Mooncake master is reachable at {host}:{port}")
            break
    except OSError:
        try:
            os.kill(pid, 0)
        except OSError as error:
            raise SystemExit("Mooncake master exited during startup") from error
        time.sleep(1)
else:
    raise SystemExit("Timed out waiting for Mooncake master")
PY
elif [[ ${START_MOONCAKE_MASTER:-0} != 0 ]]; then
  echo "START_MOONCAKE_MASTER must be 0 or 1" >&2
  exit 2
fi

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
  --context-length "$CONTEXT_LENGTH" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend "$PD_TRANSFER_BACKEND" \
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
  --disaggregation-ib-device "$IB_DEVICE" \
  "${timing_args[@]}" \
  "${schedule_args[@]}" \
  "${hicache_args[@]}" \
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
  >"$ROUTER_LOG" 2>&1 &
pids+=("$!")
router_pid=$!

wait_for_health router "http://127.0.0.1:$ROUTER_PORT" "$router_pid"
echo "PREFILL_ROUTER_READY"

wait "$prefill_pid" "$router_pid"
