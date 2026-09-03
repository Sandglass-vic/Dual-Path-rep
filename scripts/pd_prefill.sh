#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR is required}"
: "${LOG_DIR:?LOG_DIR is required}"
: "${PREFILL_HOST:?PREFILL_HOST is required}"

IB_DEVICE=${IB_DEVICE:-mlx5_0}
PREFILL_PORT=${PREFILL_PORT:-30000}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
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
if [[ -n ${RUN_ID:-} ]]; then
  PREFILL_LOG="${PREFILL_LOG%.log}-$RUN_ID.log"
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

echo "Starting TP8 prefill on $PREFILL_HOST"
exec python -m sglang.launch_server \
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
  >"$PREFILL_LOG" 2>&1
