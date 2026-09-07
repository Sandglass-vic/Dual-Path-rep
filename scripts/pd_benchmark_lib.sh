#!/usr/bin/env bash
# Shared helpers for formal 1P1D / 1P2D / 2P2D Mooncake PD benchmarks.
# shellcheck disable=SC2034

dualpath_export_common_env() {
  export ENABLE_BENCHMARK_TIMING=1
  export HICACHE_STORAGE_BACKEND=file
  export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.55}
  export HICACHE_RATIO=${HICACHE_RATIO:-1.2}
  export DECODE_HICACHE_RATIO=${DECODE_HICACHE_RATIO:-1.05}
  export CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
  export MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-rdma}
  export MOONCAKE_DEVICE=${MOONCAKE_DEVICE:-$IB_DEVICE}
  # Mooncake defaults synchronous transfers to 30s. Long-context concurrent
  # PD batches can exceed that and permanently invalidate the RDMA session.
  export MC_TRANSFER_TIMEOUT=${MC_TRANSFER_TIMEOUT:-120}
  export SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES=${SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES:-128}
  export MOONCAKE_MASTER_PORT=${MOONCAKE_MASTER_PORT:-50051}
  export MOONCAKE_METADATA_PORT=${MOONCAKE_METADATA_PORT:-8080}
  export MOONCAKE_MASTER="$PREFILL_HOST:$MOONCAKE_MASTER_PORT"
  export MOONCAKE_TE_META_DATA_SERVER="http://$PREFILL_HOST:$MOONCAKE_METADATA_PORT/metadata"
  # Release unlocked Prefill L1/L2 pages after the per-request PD ACK. This
  # preserves active shared prefixes without a stop-the-world /flush_cache.
  export DUALPATH_PREFILL_RELEASE_LOCAL_CACHE=${DUALPATH_PREFILL_RELEASE_LOCAL_CACHE:-1}
  export PREFILL_HICACHE_READ_ONLY=${PREFILL_HICACHE_READ_ONLY:-1}
  # A page can still be locked by HiCache when the PD ACK arrives. Revisit the
  # cache after HiCache progress so pages unlocked after the immediate release
  # do not become cross-turn L1/L2 hits.
  export PREFILL_HICACHE_EVICT_UNLOCKED=${PREFILL_HICACHE_EVICT_UNLOCKED:-1}
  # Retained in run metadata for compatibility; Prefill performs no backup and
  # therefore never waits on this timeout.
  export DUALPATH_PREFILL_BACKUP_TIMEOUT=${DUALPATH_PREFILL_BACKUP_TIMEOUT:-0}
  export PREFILL_HICACHE_WRITE_POLICY=${PREFILL_HICACHE_WRITE_POLICY:-write_through_selective}
  # Prefill RDMA writes into a pinned Decode Host pool. Decode copies the
  # completed request to HBM before exposing transfer success to scheduling.
  export DUALPATH_DECODE_PD_HOST_STAGING=${DUALPATH_DECODE_PD_HOST_STAGING:-1}
  export DUALPATH_DECODE_PD_HOST_STAGING_GB=${DUALPATH_DECODE_PD_HOST_STAGING_GB:-40}
  # Retained only as a historical/control switch; disabled for formal runs.
  export FLUSH_LOCAL_CACHE_BETWEEN_TURNS=${FLUSH_LOCAL_CACHE_BETWEEN_TURNS:-0}
  export STORAGE_SETTLE_SECONDS=${STORAGE_SETTLE_SECONDS:-0}
  export DUALPATH_DECODE_L3_SYNC_RANKS=${DUALPATH_DECODE_L3_SYNC_RANKS:-8}
  export DUALPATH_DECODE_L3_SYNC_TIMEOUT=${DUALPATH_DECODE_L3_SYNC_TIMEOUT:-600}
  # Recursive Lustre usage scans are diagnostic-only. Running one after every
  # turn dominates long-context benchmark wall time under concurrent agents.
  export COLLECT_L3_STORAGE_SNAPSHOT=${COLLECT_L3_STORAGE_SNAPSHOT:-0}
  export MODEL_DIR LOG_DIR CACHE_DIR IB_DEVICE PREFILL_PORT DECODE_PORT
  export BOOTSTRAP_PORT ROUTER_PORT SERVER_TIMEOUT
  export MC_TRANSFER_TIMEOUT SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES
  export DUALPATH_PREFILL_RELEASE_LOCAL_CACHE PREFILL_HICACHE_READ_ONLY
  export PREFILL_HICACHE_EVICT_UNLOCKED
  export DUALPATH_PREFILL_BACKUP_TIMEOUT
  export PREFILL_HICACHE_WRITE_POLICY
  export DUALPATH_DECODE_PD_HOST_STAGING
  export DUALPATH_DECODE_PD_HOST_STAGING_GB DECODE_HICACHE_RATIO
  export FLUSH_LOCAL_CACHE_BETWEEN_TURNS STORAGE_SETTLE_SECONDS
  export DUALPATH_DECODE_L3_SYNC_RANKS DUALPATH_DECODE_L3_SYNC_TIMEOUT
  export COLLECT_L3_STORAGE_SNAPSHOT
  export MOONCAKE_DEVICE
}

dualpath_wait_for_url() {
  local name=$1
  local url=$2
  local step_pid=$3
  local deadline=$((SECONDS + SERVER_TIMEOUT))
  until curl --fail --silent --max-time 2 "$url/health" >/dev/null; do
    if ! kill -0 "$step_pid" 2>/dev/null; then
      echo "$name exited before becoming healthy" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for $name at $url" >&2
      return 1
    fi
    sleep 5
  done
}

dualpath_wait_for_url_down() {
  local name=$1
  local url=$2
  local deadline=$((SECONDS + ${SERVER_SHUTDOWN_TIMEOUT:-120}))
  while curl --fail --silent --max-time 2 "$url/health" >/dev/null; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for $name to stop at $url" >&2
      return 1
    fi
    sleep 1
  done
}

dualpath_stop_steps() {
  local pid
  local host
  for pid in "${step_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${step_pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
  step_pids=()

  # srun can exit before the remote task has released its listening socket.
  # Do not delete this run's cache or accept the next run as healthy until all
  # prior services are unreachable.
  if [[ -n ${PREFILL_HOSTS:-} ]]; then
    for host in $PREFILL_HOSTS; do
      dualpath_wait_for_url_down \
        "Prefill on $host" "http://$host:$PREFILL_PORT"
    done
  elif [[ -n ${PREFILL_HOST:-} ]]; then
    dualpath_wait_for_url_down \
      "Prefill on $PREFILL_HOST" "http://$PREFILL_HOST:$PREFILL_PORT"
  fi
  if [[ -n ${PREFILL_HOST:-} ]]; then
    dualpath_wait_for_url_down \
      "router on $PREFILL_HOST" "http://$PREFILL_HOST:$ROUTER_PORT"
  fi
  for host in ${DECODE_HOSTS:-}; do
    dualpath_wait_for_url_down \
      "Decode on $host" "http://$host:$DECODE_PORT"
  done
}

dualpath_container_srun() {
  local host=$1
  shift
  exec srun --nodes=1 --ntasks=1 --nodelist="$host" \
    --gres=gpu:8 --exclusive --exact --mpi=none \
    --container-image="$IMAGE" \
    --container-mounts="/mnt/lustre:/mnt/lustre,$CODE_HOME:/workspace/dual-path" \
    --container-workdir=/workspace/dual-path \
    --no-container-mount-home \
    "$@"
}

dualpath_write_run_json() {
  local path=$1
  local architecture=$2
  local max_len=$3
  local agents=$4
  local repetition=$5
  local dataset=${6:-${max_len}K}
  local flush_flag=0
  local snapshot_flag=0
  local settle="${STORAGE_SETTLE_SECONDS:-0}"
  if [[ "${FLUSH_LOCAL_CACHE_BETWEEN_TURNS:-1}" == 1 ]]; then
    flush_flag=1
  fi
  if [[ "${COLLECT_L3_STORAGE_SNAPSHOT:-0}" == 1 ]]; then
    snapshot_flag=1
  fi
  python - "$path" "$architecture" "$max_len" "$agents" "$repetition" "$dataset" \
    "$MAX_TURNS" "$ABLATION_PROFILE" "$MEASURED_PREFILL_L3" \
    "$MEASURED_DECODE_OFFLOAD" "$CHUNKED_PREFILL_SIZE" \
    "$DISABLE_OVERLAP_SCHEDULE" "$PAPER_EXPAND_AGENTS" \
    "$PAPER_SYNTHETIC_APPEND_TOKENS" "$flush_flag" "$settle" \
    "$snapshot_flag" "$MC_TRANSFER_TIMEOUT" \
    "$SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES" \
    "$DUALPATH_PREFILL_RELEASE_LOCAL_CACHE" \
    "$PREFILL_HICACHE_WRITE_POLICY" "$IB_DEVICE" \
    "$PREFILL_HICACHE_READ_ONLY" "$PREFILL_HICACHE_EVICT_UNLOCKED" \
    "$DUALPATH_PREFILL_BACKUP_TIMEOUT" \
    "$DUALPATH_DECODE_PD_HOST_STAGING" \
    "$DUALPATH_DECODE_PD_HOST_STAGING_GB" \
    "$DECODE_HICACHE_RATIO" <<'PY'
from pathlib import Path
import json
import os
import sys

try:
    pd_ib_device_mapping = json.loads(sys.argv[22])
except json.JSONDecodeError:
    pd_ib_device_mapping = sys.argv[22]

Path(sys.argv[1]).write_text(json.dumps({
    "architecture": sys.argv[2],
    "max_len_k": int(sys.argv[3]),
    "agents": int(sys.argv[4]),
    "repetition": int(sys.argv[5]),
    "dataset": sys.argv[6],
    "turns": int(sys.argv[7]),
    "ablation_profile": sys.argv[8],
    "prefill_l3": bool(int(sys.argv[9])),
    "decode_offload": bool(int(sys.argv[10])),
    "chunked_prefill_size": int(sys.argv[11]),
    "disable_overlap_schedule": bool(int(sys.argv[12])),
    "paper_expand_agents": bool(int(sys.argv[13])),
    "paper_synthetic_append_tokens": int(sys.argv[14]),
    "cache_policy": "paper-aligned-l3-reload",
    "cache_initial_state": "empty",
    "storage_backend": "sglang-file-posix-lustre",
    "artificial_inter_turn_delay": False,
    "per_turn_cache_flush": bool(int(sys.argv[15])),
    "storage_settle_seconds": float(sys.argv[16]),
    "storage_quiescence_wait": bool(float(sys.argv[16]) > 0),
    "collect_l3_storage_snapshot": bool(int(sys.argv[17])),
    "mooncake_transfer_timeout_seconds": int(sys.argv[18]),
    "mooncake_max_transfer_batch_indices": int(sys.argv[19]),
    "prefill_release_local_cache_after_pd_ack": bool(int(sys.argv[20])),
    "prefill_hicache_write_policy": sys.argv[21],
    "pd_ib_device_mapping": pd_ib_device_mapping,
    "prefill_hicache_read_only": bool(int(sys.argv[23])),
    "prefill_hicache_evict_unlocked": bool(int(sys.argv[24])),
    "prefill_backup_timeout_seconds": float(sys.argv[25]),
    "decode_pd_host_staging": bool(int(sys.argv[26])),
    "decode_pd_host_staging_pool": "dedicated_pinned_host_pool",
    "decode_pd_host_staging_gb_per_rank": float(sys.argv[27]),
    "decode_hicache_ratio": float(sys.argv[28]),
    "decode_l3_completion_sync": bool(
        os.environ.get("DUALPATH_DECODE_L3_SYNC_DIR")
    ),
    "decode_l3_completion_sync_ranks": int(
        os.environ.get("DUALPATH_DECODE_L3_SYNC_RANKS", "8")
    ),
    "decode_l3_completion_sync_timeout_seconds": float(
        os.environ.get("DUALPATH_DECODE_L3_SYNC_TIMEOUT", "600")
    ),
    "l3_storage_snapshot_kind": (
        "recursive apparent bytes and st_blocks allocated bytes"
        if bool(int(sys.argv[17]))
        else "disabled"
    ),
    "kv_buffer_semantics": (
        "DualPath paper: staging/GPU buffers released after each request; "
        "cross-turn reuse via L3 only. SGLang approximation: after successful "
        "PD ACK, release the request lock and evict unlocked Prefill L1/L2 "
        "radix pages without Prefill offload or backup wait. P-to-D RDMA targets "
        "a bounded pinned Decode Host staging pool; Decode copies those pages to "
        "HBM, then publishes newly computed KV after the cached_tokens boundary "
        "through a separate HiCache Host pool to Lustre file storage."
    ),
}, indent=2) + "\n")
PY
}

dualpath_resolve_dataset() {
  local spec=$1
  local value
  if [[ "$spec" == *=* ]]; then
    DATASET_LABEL=${spec%%=*}
    DATASET_PATH=${spec#*=}
  else
    value=${spec%[Kk]}
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
      echo "Dataset '$spec' must be N, NK, or LABEL=/path/file.jsonl" >&2
      return 2
    fi
    DATASET_LABEL="${value}K"
    DATASET_PATH="$RUNTIME_HOME/test/dataset/dualpath_fake_${value}k.jsonl"
  fi
  value=${DATASET_LABEL%[Kk]}
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "Dataset label '$DATASET_LABEL' must use the form NK" >&2
    return 2
  fi
  DATASET_MAX_LEN=$value
  DATASET_LABEL="${value}K"
  if [[ ! -f "$DATASET_PATH" ]]; then
    echo "Dataset file does not exist: $DATASET_PATH" >&2
    return 2
  fi
  export DATASET_LABEL DATASET_PATH DATASET_MAX_LEN
}

dualpath_repetition_ids() {
  if [[ -n ${REP_LIST:-} ]]; then
    printf '%s\n' ${REP_LIST//,/ }
  elif [[ "$REPETITIONS" == *" "* || "$REPETITIONS" == *","* ]]; then
    printf '%s\n' ${REPETITIONS//,/ }
  elif [[ "$REPETITIONS" =~ ^[0-9]+$ ]] && (( REPETITIONS > 0 )); then
    seq 1 "$REPETITIONS"
  else
    echo "REPETITIONS must be a positive count or REP_LIST must list IDs" >&2
    return 2
  fi
}

dualpath_run_replay() {
  local dataset=$1
  local agents=$2
  local output_dir=$3
  local prefix=$4
  local paper_args=()
  local turn_args=()
  local flush_args=()
  local storage_args=()
  local sync_args=()
  local host
  if (( MAX_TURNS > 0 )); then
    turn_args+=(--max-turns "$MAX_TURNS")
  fi
  if [[ "$PAPER_EXPAND_AGENTS" == 1 ]]; then
    paper_args+=(
      --paper-expand-agents
      --paper-synthetic-append-tokens "$PAPER_SYNTHETIC_APPEND_TOKENS"
    )
  elif [[ "$PAPER_EXPAND_AGENTS" != 0 ]]; then
    echo "PAPER_EXPAND_AGENTS must be 0 or 1" >&2
    return 2
  fi
  if [[ "${FLUSH_LOCAL_CACHE_BETWEEN_TURNS:-1}" == 1 ]]; then
    flush_args+=(--flush-local-cache-between-turns)
    # Flush every Prefill instance (1P* uses PREFILL_HOST; 2P2D also lists PREFILL_HOSTS).
    if [[ -n ${PREFILL_HOSTS:-} ]]; then
      for host in $PREFILL_HOSTS; do
        flush_args+=(--flush-cache-url "http://$host:$PREFILL_PORT/flush_cache")
      done
    else
      flush_args+=(--flush-cache-url "http://$PREFILL_HOST:$PREFILL_PORT/flush_cache")
    fi
  elif [[ "${FLUSH_LOCAL_CACHE_BETWEEN_TURNS:-1}" != 0 ]]; then
    echo "FLUSH_LOCAL_CACHE_BETWEEN_TURNS must be 0 or 1" >&2
    return 2
  fi
  flush_args+=(--storage-settle-seconds "${STORAGE_SETTLE_SECONDS:-0}")
  if [[ -n ${DUALPATH_DECODE_L3_SYNC_DIR:-} ]]; then
    sync_args+=(
      --decode-l3-sync-dir "$DUALPATH_DECODE_L3_SYNC_DIR"
      --decode-l3-sync-ranks "${DUALPATH_DECODE_L3_SYNC_RANKS:-8}"
      --decode-l3-sync-timeout "${DUALPATH_DECODE_L3_SYNC_TIMEOUT:-600}"
    )
  fi
  if [[ "${COLLECT_L3_STORAGE_SNAPSHOT:-0}" == 1 ]]; then
    : "${L3_STORAGE_DIR:?L3_STORAGE_DIR is required for L3 snapshots}"
    storage_args+=(--storage-dir "$L3_STORAGE_DIR")
  elif [[ "${COLLECT_L3_STORAGE_SNAPSHOT:-0}" != 0 ]]; then
    echo "COLLECT_L3_STORAGE_SNAPSHOT must be 0 or 1" >&2
    return 2
  fi
  python scripts/benchmark/replay_fake_agent_traces.py \
    --dataset "$dataset" \
    --base-url "http://$PREFILL_HOST:$ROUTER_PORT" \
    --output-dir "$output_dir" \
    --server-selection-timeout "${SERVER_SELECTION_TIMEOUT:-120}" \
    --trajectory-offset "$TRAJECTORY_OFFSET" \
    --num-trajectories "$agents" \
    --max-workers "$agents" \
    "${turn_args[@]}" \
    --max-context-tokens "$CONTEXT_LENGTH" \
    --request-id-prefix "$prefix" \
    "${paper_args[@]}" \
    "${flush_args[@]}" \
    "${sync_args[@]}" \
    "${storage_args[@]}" \
    --stream
}
