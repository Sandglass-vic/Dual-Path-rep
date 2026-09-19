#!/usr/bin/env bash
# Runs inside the benchmark container on exactly one allocated node.
set -euo pipefail

role=${1:-}
role_id=${2:-0}
if [[ $role != prefill && $role != decode && $role != client ]]; then
  echo "Usage: scripts/tests/runtime/pd_runtime.sh prefill|decode|client [engine-id]" >&2
  exit 2
fi
[[ $role_id =~ ^[0-9]+$ ]] || { echo "Invalid engine ID: $role_id" >&2; exit 2; }

CODE_HOME=${CODE_HOME:-/workspace/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/data/home/yongxiao/mydata}
: "${PREFILL_HOSTS:?PREFILL_HOSTS is required}"
: "${DECODE_HOSTS:?DECODE_HOSTS is required}"
read -r -a prefill_hosts <<<"$PREFILL_HOSTS"
read -r -a decode_hosts <<<"$DECODE_HOSTS"
export PREFILL_NODES=${#prefill_hosts[@]} DECODE_NODES=${#decode_hosts[@]}
export ARCHITECTURE="${PREFILL_NODES}P${DECODE_NODES}D"
export PREFILL_HOST=${prefill_hosts[0]}
export ROUTER_HOST=${ROUTER_HOST:-$(hostname -s)}
export MODEL_DIR=${MODEL_DIR:-$RUNTIME_HOME/models/Qwen2.5-32B}
export PREFILL_PORT=${PREFILL_PORT:-30000} DECODE_PORT=${DECODE_PORT:-30000}
export BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998} ROUTER_PORT=${ROUTER_PORT:-8000}
export SERVER_TIMEOUT=${SERVER_TIMEOUT:-1200}
export TP_SIZE=${TP_SIZE:-4}
export JOB_RUNTIME_DIR=${JOB_RUNTIME_DIR:-$RUNTIME_HOME/test/$ALLOC_JOB_ID}
export SERVER_CACHE_DIR=${SERVER_CACHE_DIR:-$JOB_RUNTIME_DIR/pd-servers/kv-cache/file}
: "${DUALPATH_L3_READ_PATH:?Set DUALPATH_L3_READ_PATH=prefill or decode for every role}"
[[ $DUALPATH_L3_READ_PATH == prefill || $DUALPATH_L3_READ_PATH == decode ]] || {
  echo "DUALPATH_L3_READ_PATH must be prefill or decode" >&2
  exit 2
}

source "$CODE_HOME/scripts/tests/runtime/pd_network_config.sh"
source "$CODE_HOME/scripts/tests/runtime/ablation_profiles.sh"
source "$CODE_HOME/scripts/tests/runtime/pd_benchmark_lib.sh"
apply_ablation_profile "${ABLATION_PROFILE:-full}"
dualpath_export_common_env
unset MOONCAKE_MASTER MOONCAKE_TE_META_DATA_SERVER

probe_o_direct_read() {
  local directory=$1
  python - "$directory" <<'PY'
import ctypes
import mmap
import os
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
alignment = 4096
path = directory / f".dualpath-o-direct-probe-{os.getpid()}"
view = None
scratch = None
try:
    with path.open("wb", buffering=0) as file:
        file.write(b"\0" * alignment)
        os.fsync(file.fileno())

    scratch = mmap.mmap(-1, alignment * 2)
    address = ctypes.addressof(ctypes.c_char.from_buffer(scratch))
    offset = (-address) % alignment
    view = memoryview(scratch)[offset : offset + alignment]
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        count = os.readv(fd, [view])
    finally:
        os.close(fd)
    if count != alignment:
        raise OSError(f"short O_DIRECT probe read: {count}/{alignment}")
except Exception as exc:
    print(
        f"O_DIRECT read probe failed for {directory}: "
        f"{type(exc).__name__}: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(2)
finally:
    if view is not None:
        view.release()
    if scratch is not None:
        scratch.close()
    path.unlink(missing_ok=True)
PY
}

if [[ $role == prefill || $role == decode ]]; then
  TASK_NAME=${TASK_NAME:-allocation-$ALLOC_JOB_ID}
  export TASK_NAME
  export LOG_DIR=${SERVER_LOG_DIR:-$JOB_RUNTIME_DIR/pd-servers}
  export CACHE_DIR=$SERVER_CACHE_DIR
else
  DATASET=${DATASET:-32K}
  AGENTS=${AGENTS:-1}
  REPETITION=${REPETITION:-1}
  MAX_TURNS=${MAX_TURNS:-0}
  TRAJECTORY_OFFSET=${TRAJECTORY_OFFSET:-0}
  PAPER_EXPAND_AGENTS=${PAPER_EXPAND_AGENTS:-0}
  PAPER_SYNTHETIC_APPEND_TOKENS=${PAPER_SYNTHETIC_APPEND_TOKENS:-1}
  TASK_NAME=${TASK_NAME:-${ARCHITECTURE,,}-${DATASET,,}-a${AGENTS}-r${REPETITION}-$(date +%Y%m%d-%H%M%S)}
  TASK_RUNTIME_DIR=${TASK_RUNTIME_DIR:-$JOB_RUNTIME_DIR/$TASK_NAME}
  LOG_DIR=${LOG_DIR:-$TASK_RUNTIME_DIR/logs}
  RESULT_DIR=${RESULT_DIR:-$TASK_RUNTIME_DIR/logs}
  CACHE_DIR=$SERVER_CACHE_DIR
  export TASK_NAME DATASET AGENTS REPETITION MAX_TURNS TRAJECTORY_OFFSET
  export PAPER_EXPAND_AGENTS PAPER_SYNTHETIC_APPEND_TOKENS LOG_DIR CACHE_DIR
  for name in AGENTS REPETITION; do
    [[ ${!name} =~ ^[1-9][0-9]*$ ]] || { echo "$name must be positive" >&2; exit 2; }
  done
  for name in MAX_TURNS TRAJECTORY_OFFSET; do
    [[ ${!name} =~ ^(0|[1-9][0-9]*)$ ]] || { echo "$name must be nonnegative" >&2; exit 2; }
  done
fi

if [[ $role == client ]]; then
  requested_router_port=${ROUTER_PORT:-8000}
  if dualpath_port_in_use "$requested_router_port"; then
    echo "Requested Router port $requested_router_port is occupied; checking for stale client state." >&2
    dualpath_report_router_port "$requested_router_port" || true
  fi
  if ! dualpath_report_client_residue; then
    if [[ ${PD_CLIENT_ALLOW_RESIDUAL:-0} != 1 ]]; then
      echo "Refusing to start client. Clean residual processes first, or set PD_CLIENT_ALLOW_RESIDUAL=1." >&2
      exit 2
    fi
    echo "PD_CLIENT_ALLOW_RESIDUAL=1: continuing despite residual client processes." >&2
  fi
  ROUTER_PORT=$(dualpath_choose_router_port)
  if ! dualpath_report_router_port "$ROUTER_PORT"; then
    echo "Selected Router port $ROUTER_PORT became occupied; retry the client run." >&2
    exit 2
  fi
  export ROUTER_PORT
  printf '  router_port=%s (requested=%s, auto=%s)\n' \
    "$ROUTER_PORT" "$requested_router_port" "${ROUTER_PORT_AUTO:-1}"
fi

export DUALPATH_DECODE_L3_SYNC_DIR=
if [[ $MEASURED_DECODE_OFFLOAD == 1 ]]; then
  export DUALPATH_DECODE_L3_SYNC_DIR="${CACHE_DIR%/file}/l3-sync"
fi

if [[ $role == client ]]; then
  printf 'Client run configuration:\n'
  printf '  task=%s dataset=%s agents=%s repetition=%s l3_read=%s\n' \
    "$TASK_NAME" "$DATASET" "$AGENTS" "$REPETITION" \
    "$DUALPATH_L3_READ_PATH"
  printf '  dual_path_o_direct_read=%s\n' "$DUAL_PATH_O_DIRECT_READ"
  printf '  ttft_timing_v3=%s detail=%s\n' \
    "$DUALPATH_TTFT_TIMING" "$DUALPATH_TTFT_TIMING_DETAIL"
  printf '  log_dir=%s\n  result_dir=%s\n  server_kv_cache_dir=%s\n' \
    "$LOG_DIR" "$RESULT_DIR" "$CACHE_DIR"
fi

if [[ $role == prefill || $role == decode ]]; then
  mkdir -p "$LOG_DIR" "$CACHE_DIR"
  storage_fstype=$(stat -f -c %T "$CACHE_DIR")
  export STORAGE_FSTYPE=${STORAGE_FSTYPE:-$storage_fstype}
  if [[ $DUAL_PATH_O_DIRECT_READ == 1 ]]; then
    probe_o_direct_read "$CACHE_DIR"
  fi
  printf 'Server configuration: role=%s l3_read=%s log_dir=%s cache_dir=%s\n' \
    "$role" "$DUALPATH_L3_READ_PATH" "$LOG_DIR" "$CACHE_DIR"
  printf '  storage_fstype=%s storage_topology=%s\n' \
    "$STORAGE_FSTYPE" "${STORAGE_TOPOLOGY:-g5-xfs-export-g6-nfs}"
  printf '  dual_path_o_direct_read=%s\n' "$DUAL_PATH_O_DIRECT_READ"
  printf '  ttft_timing_v3=%s detail=%s\n' \
    "$DUALPATH_TTFT_TIMING" "$DUALPATH_TTFT_TIMING_DETAIL"
fi

if [[ $role == prefill || $role == decode ]]; then
  # Engine processes are long-lived; they do not own a client run directory.
  if [[ $role == prefill ]]; then
    (( role_id >= 0 && role_id < ${#prefill_hosts[@]} )) || {
      echo "Invalid Prefill engine ID: $role_id" >&2
      exit 2
    }
    export PREFILL_ID=$role_id PREFILL_HOST=${prefill_hosts[role_id]}
    export ENABLE_HICACHE_L3=$MEASURED_PREFILL_L3
    exec bash "$CODE_HOME/scripts/tests/runtime/pd_prefill.sh"
  fi
  (( role_id >= 0 && role_id < ${#decode_hosts[@]} )) || {
    echo "Invalid Decode engine ID: $role_id" >&2
    exit 2
  }
  export DECODE_ID=$role_id DECODE_HOST=${decode_hosts[role_id]}
  export ENABLE_HICACHE_L3=$MEASURED_DECODE_OFFLOAD
  export DECODE_OFFLOAD_KVCACHE=$MEASURED_DECODE_OFFLOAD
  exec bash "$CODE_HOME/scripts/tests/runtime/pd_decode.sh"
fi

dualpath_resolve_dataset "$DATASET"
export DATASET_PATH
export RUN_ID="${DATASET_LABEL,,}-a${AGENTS}-r${REPETITION}"
export RUN_DIR="$RESULT_DIR/$DATASET_LABEL/agents-$AGENTS/rep-$REPETITION"
if [[ -e $RUN_DIR/summary.json || -e $RUN_DIR/rounds.jsonl ]]; then
  echo "Result files already exist under $RUN_DIR; choose a new TASK_NAME" >&2
  exit 2
fi
mkdir -p "$LOG_DIR" "$RUN_DIR" "$CACHE_DIR"
storage_fstype=$(stat -f -c %T "$CACHE_DIR")
export STORAGE_FSTYPE=${STORAGE_FSTYPE:-$storage_fstype}
if [[ -n $DUALPATH_DECODE_L3_SYNC_DIR ]]; then
  mkdir -p "$DUALPATH_DECODE_L3_SYNC_DIR"
fi

for host in "${prefill_hosts[@]}"; do
  dualpath_wait_for_url "Prefill $host" "http://$host:$PREFILL_PORT"
done
for host in "${decode_hosts[@]}"; do
  dualpath_wait_for_url "Decode $host" "http://$host:$DECODE_PORT"
done

printf '%s\n' "${prefill_hosts[@]}" "${decode_hosts[@]}" >"$RESULT_DIR/nodes.txt"
printf 'client=%s\n' "$ROUTER_HOST" >>"$RESULT_DIR/nodes.txt"
print_ablation_profile >"$RESULT_DIR/ablation-profile.env"
printf 'ARCHITECTURE=%s\n' "$ARCHITECTURE" >>"$RESULT_DIR/ablation-profile.env"
dualpath_write_run_json "$RUN_DIR/run.json" "$ARCHITECTURE" "$DATASET_MAX_LEN" \
  "$AGENTS" "$REPETITION" "$DATASET_LABEL"

router_log="$LOG_DIR/router-$RUN_ID.log"
router_cmd=(python -m sglang_router.launch_router --pd-disaggregation
  --host 0.0.0.0 --port "$ROUTER_PORT"
  --prefill-policy round_robin --decode-policy round_robin)
for host in "${prefill_hosts[@]}"; do
  router_cmd+=(--prefill "http://$host:$PREFILL_PORT" "$BOOTSTRAP_PORT")
done
for host in "${decode_hosts[@]}"; do
  router_cmd+=(--decode "http://$host:$DECODE_PORT")
done

setsid "${router_cmd[@]}" >"$router_log" 2>&1 &
router_pid=$!
replay_pid=
client_outcome=failed
print_client_cleanup_hint() {
  printf '\nCold-cache cleanup (only after stopping Prefill/Decode and releasing the allocation):\n'
  printf '  rm -rf %q\n' "$JOB_RUNTIME_DIR/pd-servers/kv-cache"
}
print_client_success_hints() {
  print_client_cleanup_hint
  printf '\nPlot command:\n'
  printf '  cd %q && ALLOC_JOB_ID=%q TASK_NAME=%q scripts/pd_plot.sh %q\n' \
    "$CODE_HOME" "$ALLOC_JOB_ID" "$TASK_NAME" "$TASK_NAME"
}
cleanup() {
  trap - EXIT
  if [[ -n ${replay_pid:-} ]] && kill -0 "$replay_pid" 2>/dev/null; then
    kill -TERM -- "-$replay_pid" 2>/dev/null || kill -TERM "$replay_pid" 2>/dev/null || true
    for _ in {1..100}; do
      kill -0 "$replay_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$replay_pid" 2>/dev/null; then
      kill -KILL -- "-$replay_pid" 2>/dev/null || kill -KILL "$replay_pid" 2>/dev/null || true
    fi
    wait "$replay_pid" 2>/dev/null || true
  fi
  if kill -0 "$router_pid" 2>/dev/null; then
    kill -TERM -- "-$router_pid" 2>/dev/null || true
    for _ in {1..100}; do
      kill -0 "$router_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$router_pid" 2>/dev/null; then
      kill -KILL -- "-$router_pid" 2>/dev/null || true
    fi
    wait "$router_pid" 2>/dev/null || true
  fi
  if [[ $client_outcome == success ]]; then
    print_client_success_hints
  else
    print_client_cleanup_hint
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
dualpath_wait_for_url Router "http://$ROUTER_HOST:$ROUTER_PORT" "$router_pid"
dualpath_run_replay "$DATASET_PATH" "$AGENTS" "$RUN_DIR" \
  "bench-${TASK_NAME}-${ARCHITECTURE,,}-${RUN_ID}-" &
replay_pid=$!
set +e
wait "$replay_pid"
replay_status=$?
set -e
replay_pid=
if (( replay_status != 0 )); then
  echo "Replay exited with status $replay_status" >&2
  exit "$replay_status"
fi
client_outcome=success
echo "REPLAY_COMPLETE task=$TASK_NAME result_dir=$RESULT_DIR"
echo "Prefill/Decode servers remain running; this client run only stopped Router."
