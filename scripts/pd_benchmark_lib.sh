#!/usr/bin/env bash
# Shared helpers for formal 1P1D / 1P2D / 2P2D Mooncake Store benchmarks.
# shellcheck disable=SC2034

dualpath_export_common_env() {
  export ENABLE_BENCHMARK_TIMING=1
  export HICACHE_STORAGE_BACKEND=mooncake
  export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.55}
  export HICACHE_RATIO=${HICACHE_RATIO:-1.2}
  export CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
  export MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-rdma}
  export MOONCAKE_DEVICE=${MOONCAKE_DEVICE:-$IB_DEVICE}
  export MOONCAKE_GLOBAL_SEGMENT_SIZE=${MOONCAKE_GLOBAL_SEGMENT_SIZE:-1536gb}
  export MOONCAKE_MASTER_PORT=${MOONCAKE_MASTER_PORT:-50051}
  export MOONCAKE_METADATA_PORT=${MOONCAKE_METADATA_PORT:-8080}
  export MOONCAKE_MASTER="$PREFILL_HOST:$MOONCAKE_MASTER_PORT"
  export MOONCAKE_TE_META_DATA_SERVER="http://$PREFILL_HOST:$MOONCAKE_METADATA_PORT/metadata"
  export MOONCAKE_OFFLOAD_ENABLED=true
  export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=distributed_storage_backend
  export MOONCAKE_DFS_FS_ADAPTER=posix
  export MOONCAKE_DFS_SHARD_COUNT=${MOONCAKE_DFS_SHARD_COUNT:-64}
  export MOONCAKE_DFS_SHARD_CAPACITY=${MOONCAKE_DFS_SHARD_CAPACITY:-4294967296}
  export MOONCAKE_DFS_ALIGNMENT=${MOONCAKE_DFS_ALIGNMENT:-4096}
  export MOONCAKE_DFS_SINGLE_TENANT=true
  export MOONCAKE_ENABLE_DFS=1
  export MODEL_DIR LOG_DIR CACHE_DIR IB_DEVICE PREFILL_PORT DECODE_PORT
  export BOOTSTRAP_PORT ROUTER_PORT SERVER_TIMEOUT
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

dualpath_stop_steps() {
  local pid
  for pid in "${step_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${step_pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
  step_pids=()
}

dualpath_container_srun() {
  local host=$1
  shift
  srun --nodes=1 --ntasks=1 --nodelist="$host" \
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
  python - "$path" "$architecture" "$max_len" "$agents" "$repetition" \
    "$MAX_TURNS" "$ABLATION_PROFILE" "$MEASURED_PREFILL_L3" \
    "$MEASURED_DECODE_OFFLOAD" "$CHUNKED_PREFILL_SIZE" \
    "$DISABLE_OVERLAP_SCHEDULE" "$PAPER_EXPAND_AGENTS" \
    "$PAPER_SYNTHETIC_APPEND_TOKENS" <<'PY'
from pathlib import Path
import json
import sys

Path(sys.argv[1]).write_text(json.dumps({
    "architecture": sys.argv[2],
    "max_len_k": int(sys.argv[3]),
    "agents": int(sys.argv[4]),
    "repetition": int(sys.argv[5]),
    "turns": int(sys.argv[6]),
    "ablation_profile": sys.argv[7],
    "prefill_l3": bool(int(sys.argv[8])),
    "decode_offload": bool(int(sys.argv[9])),
    "chunked_prefill_size": int(sys.argv[10]),
    "disable_overlap_schedule": bool(int(sys.argv[11])),
    "paper_expand_agents": bool(int(sys.argv[12])),
    "paper_synthetic_append_tokens": int(sys.argv[13]),
    "cache_policy": "dynamic-full-rollout",
    "cache_initial_state": "empty",
    "storage_backend": "mooncake-dfs-posix-lustre",
    "artificial_inter_turn_delay": False,
    "per_turn_cache_flush": False,
    "storage_quiescence_wait": False,
}, indent=2) + "\n")
PY
}

dualpath_run_replay() {
  local dataset=$1
  local agents=$2
  local output_dir=$3
  local prefix=$4
  local paper_args=()
  local turn_args=()
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
  python scripts/replay_fake_agent_traces.py \
    --dataset "$dataset" \
    --base-url "http://$PREFILL_HOST:$ROUTER_PORT" \
    --output-dir "$output_dir" \
    --trajectory-offset "$TRAJECTORY_OFFSET" \
    --num-trajectories "$agents" \
    --max-workers "$agents" \
    "${turn_args[@]}" \
    --max-context-tokens "$CONTEXT_LENGTH" \
    --request-id-prefix "$prefix" \
    "${paper_args[@]}" \
    --stream
}
