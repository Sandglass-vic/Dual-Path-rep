#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR is required}"
: "${LOG_DIR:?LOG_DIR is required}"
: "${DECODE_HOST:?DECODE_HOST is required}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/tests/runtime/pd_network_config.sh
source "$SCRIPT_DIR/pd_network_config.sh"
unset DUALPATH_HICACHE_READ_ONLY
unset DUALPATH_HICACHE_EVICT_UNLOCKED
export DUALPATH_DECODE_PD_HOST_STAGING=${DUALPATH_DECODE_PD_HOST_STAGING:-1}

DECODE_PORT=${DECODE_PORT:-30000}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-4096}
TP_SIZE=${TP_SIZE:-4}
PD_TRANSFER_BACKEND=${PD_TRANSFER_BACKEND:-mooncake}
HICACHE_STORAGE_BACKEND=${HICACHE_STORAGE_BACKEND:-file}
DECODE_OFFLOAD_KVCACHE=${DECODE_OFFLOAD_KVCACHE:-${ENABLE_HICACHE_L3:-0}}
if [[ "$PD_TRANSFER_BACKEND" == mooncake || "$HICACHE_STORAGE_BACKEND" == mooncake ]]; then
  dualpath_validate_ib_devices
fi
if [[ -n ${DECODE_ID:-} ]]; then
  DECODE_LOG="$LOG_DIR/decode-$DECODE_ID.log"
else
  DECODE_LOG="$LOG_DIR/decode.log"
fi
if [[ -n ${RUN_ID:-} ]]; then
  DECODE_LOG="${DECODE_LOG%.log}-$RUN_ID.log"
fi

hicache_args=()
if [[ ${ENABLE_HICACHE_L3:-0} == 1 ]]; then
  : "${CACHE_DIR:?CACHE_DIR is required when ENABLE_HICACHE_L3=1}"
  if [[ "$HICACHE_STORAGE_BACKEND" == file ]]; then
    export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=$CACHE_DIR
    hicache_extra_config=$(python - "$DUAL_PATH_O_DIRECT_READ" <<'PY'
import json
import sys

print(json.dumps({
    "prefetch_threshold": 64,
    "prefetch_timeout_base": 10,
    "prefetch_timeout_per_ki_token": 2,
    "dual_path_o_direct_read": bool(int(sys.argv[1])),
    "dual_path_o_direct_alignment": 4096,
}))
PY
)
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
    "local_hostname": os.environ["DECODE_HOST"],
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
    --hicache-ratio "${DECODE_HICACHE_RATIO:-1.05}"
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
if [[ "$DECODE_OFFLOAD_KVCACHE" == 1 ]]; then
  if [[ ${ENABLE_HICACHE_L3:-0} != 1 ]]; then
    echo "DECODE_OFFLOAD_KVCACHE=1 requires ENABLE_HICACHE_L3=1" >&2
    exit 2
  fi
  hicache_args+=(--disaggregation-decode-enable-offload-kvcache)
elif [[ "$DECODE_OFFLOAD_KVCACHE" != 0 ]]; then
  echo "DECODE_OFFLOAD_KVCACHE must be 0 or 1" >&2
  exit 2
fi

timing_args=()
if [[ ${ENABLE_BENCHMARK_TIMING:-0} == 1 ]]; then
  timing_args+=(--enable-request-time-stats-logging)
fi

mkdir -p "$LOG_DIR"

echo "Starting TP${TP_SIZE} decode on $DECODE_HOST"
echo "Streaming Decode output to console and $DECODE_LOG"
set +e
PYTHONUNBUFFERED=1 python -u -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --served-model-name qwen2.5-32b \
  --host 0.0.0.0 \
  --port "$DECODE_PORT" \
  --tp-size "$TP_SIZE" \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --context-length "$CONTEXT_LENGTH" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend "$PD_TRANSFER_BACKEND" \
  --disaggregation-ib-device "$IB_DEVICE" \
  "${timing_args[@]}" \
  "${hicache_args[@]}" \
  2>&1 | tee -a "$DECODE_LOG"
server_status=${PIPESTATUS[0]}
set -e
exit "$server_status"
