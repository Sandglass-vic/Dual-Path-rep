#!/usr/bin/env bash
# Launch one manual salloc session role from the node where this script runs.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/pd_benchmark.sh [prefill|decode|client|server]

Normally scripts/pd_alloc.sh writes the allocation environment and this
launcher loads it automatically. The role argument is optional. Prefer the
explicit prefill/decode/client arguments printed by pd_alloc.sh; server remains
as a compatibility alias for either engine role.

Common client workload settings:
  DATASET=32K AGENTS=1 REPETITION=1

DUALPATH_L3_READ_PATH is a required per-role launch parameter. The same value
must be supplied on every Prefill, Decode, and client invocation:
  DUALPATH_L3_READ_PATH=prefill|decode scripts/pd_benchmark.sh prefill|decode|client

Workload settings are resolved per client invocation, so the same running
Prefill/Decode servers can be reused for multiple client runs.
EOF
}

requested_role=${1:-auto}
if [[ $requested_role != auto && $requested_role != server && \
      $requested_role != prefill && $requested_role != decode && \
      $requested_role != client ]]; then
  usage >&2
  exit 2
fi

CODE_HOME=${CODE_HOME:-/data/home/yongxiao/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/data/home/yongxiao/mydata}
current_host=${DUALPATH_HOSTNAME:-$(hostname -s)}
allocation_id=${ALLOC_JOB_ID:-${SLURM_JOB_ID:-}}
if [[ -z $allocation_id ]]; then
  mapfile -t allocations < <(
    squeue -h -u "$USER" -w "$current_host" -t RUNNING -o '%A' | sort -u
  )
  if (( ${#allocations[@]} != 1 )); then
    echo "Set ALLOC_JOB_ID; found ${#allocations[@]} running allocations on $current_host" >&2
    exit 2
  fi
  allocation_id=${allocations[0]}
fi
[[ $allocation_id =~ ^[0-9]+$ ]] || { echo "Invalid ALLOC_JOB_ID: $allocation_id" >&2; exit 2; }

allocation_env=${PD_ALLOCATION_ENV:-$RUNTIME_HOME/config/allocations/$allocation_id.env}
runtime_override_names=(
  ABLATION_PROFILE MODEL_DIR CONTEXT_LENGTH MEM_FRACTION_STATIC
  HICACHE_SIZE_GB HICACHE_RATIO DECODE_HICACHE_RATIO PREFILL_HICACHE_MEM_LAYOUT
  PREFILL_HICACHE_READ_ONLY PREFILL_HICACHE_EVICT_UNLOCKED
  PREFILL_HICACHE_WRITE_POLICY DUALPATH_PREFILL_RELEASE_LOCAL_CACHE
  DUALPATH_PREFILL_BACKUP_TIMEOUT DUALPATH_DECODE_PD_HOST_STAGING
  DUALPATH_DECODE_PD_HOST_STAGING_GB FLUSH_LOCAL_CACHE_BETWEEN_TURNS
  STORAGE_SETTLE_SECONDS DUALPATH_DECODE_L3_SYNC_RANKS
  DUALPATH_DECODE_L3_SYNC_TIMEOUT MC_TRANSFER_TIMEOUT
  SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES SERVER_TIMEOUT
  PREFILL_PORT DECODE_PORT BOOTSTRAP_PORT ROUTER_PORT IB_DEVICE
  DUALPATH_TTFT_TIMING DUALPATH_TTFT_TIMING_DETAIL
  DUALPATH_L3_READ_PATH DUAL_PATH_O_DIRECT_READ STORAGE_TOPOLOGY
  DUALPATH_RUNTIME_CACHE_ROOT DUALPATH_DATASET_ROOT
  ENGINE_THREADS_PER_RANK
  TASK_NAME DATASET AGENTS REPETITION MAX_TURNS TRAJECTORY_OFFSET
  PAPER_EXPAND_AGENTS PAPER_SYNTHETIC_APPEND_TOKENS LOG_DIR RESULT_DIR CACHE_DIR
)
declare -A runtime_overrides=()
for name in "${runtime_override_names[@]}"; do
  if [[ -v $name ]]; then
    runtime_overrides[$name]=${!name}
  fi
done
if [[ -f $allocation_env ]]; then
  # shellcheck disable=SC1090
  source "$allocation_env"
elif [[ -z ${TASK_NAME:-} || -z ${PREFILL_HOSTS:-} || -z ${DECODE_HOSTS:-} ]]; then
  echo "Allocation environment not found: $allocation_env" >&2
  echo "Run scripts/pd_alloc.sh, or export TASK_NAME/PREFILL_HOSTS/DECODE_HOSTS manually." >&2
  exit 2
fi
for name in "${runtime_override_names[@]}"; do
  if [[ -v runtime_overrides[$name] ]]; then
    printf -v "$name" '%s' "${runtime_overrides[$name]}"
  else
    unset "$name"
  fi
done
: "${DUALPATH_L3_READ_PATH:?Set DUALPATH_L3_READ_PATH=prefill or decode for every role}"
[[ $DUALPATH_L3_READ_PATH == prefill || $DUALPATH_L3_READ_PATH == decode ]] || {
  echo "DUALPATH_L3_READ_PATH must be prefill or decode" >&2
  exit 2
}
[[ ${ALLOC_JOB_ID:-$allocation_id} == "$allocation_id" ]] || {
  echo "Allocation environment belongs to Job $ALLOC_JOB_ID, expected $allocation_id" >&2
  exit 2
}
export ALLOC_JOB_ID=$allocation_id
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched.sqsh}
ENGINE_GPUS=${ENGINE_GPUS:-4}
TP_SIZE=${TP_SIZE:-4}
: "${PREFILL_HOSTS:?PREFILL_HOSTS is required}"
: "${DECODE_HOSTS:?DECODE_HOSTS is required}"
if [[ -n ${TASK_NAME:-} && ! $TASK_NAME =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
  echo "Invalid TASK_NAME: $TASK_NAME" >&2
  exit 2
fi

read -r -a prefill_hosts <<<"$PREFILL_HOSTS"
read -r -a decode_hosts <<<"$DECODE_HOSTS"
(( ${#prefill_hosts[@]} > 0 && ${#decode_hosts[@]} > 0 )) || {
  echo "PREFILL_HOSTS and DECODE_HOSTS must both be nonempty" >&2
  exit 2
}
declare -A seen_hosts=()
for host in "${prefill_hosts[@]}" "${decode_hosts[@]}"; do
  if [[ -n ${seen_hosts[$host]:-} ]]; then
    echo "Engine host appears more than once: $host" >&2
    exit 2
  fi
  seen_hosts[$host]=1
done

engine_role=
engine_id=
for index in "${!prefill_hosts[@]}"; do
  if [[ ${prefill_hosts[index]} == "$current_host" ]]; then
    engine_role=prefill
    engine_id=$index
  fi
done
for index in "${!decode_hosts[@]}"; do
  if [[ ${decode_hosts[index]} == "$current_host" ]]; then
    engine_role=decode
    engine_id=$index
  fi
done
client_host=${CLIENT_HOST:-${ROUTER_HOST:-}}
is_client_host=0
if [[ -z $client_host || $client_host == "$current_host" ]]; then
  is_client_host=1
fi
if [[ -z $engine_role && $is_client_host != 1 ]]; then
  echo "$current_host has no role in allocation $allocation_id" >&2
  exit 2
fi
case $requested_role in
  auto)
    if [[ -n $engine_role ]]; then
      role=server
    else
      role=client
    fi
    ;;
  server)
    if [[ -z $engine_role ]]; then
      echo "$current_host is configured as client, not server" >&2
      exit 2
    fi
    role=server
    ;;
  client)
    if [[ $is_client_host != 1 ]]; then
      echo "$current_host is not configured as the client host" >&2
      exit 2
    fi
    role=client
    ;;
  prefill|decode)
    if [[ $engine_role != "$requested_role" ]]; then
      actual_role=${engine_role:-client}
      echo "$current_host is configured as $actual_role, not $requested_role" >&2
      exit 2
    fi
    role=server
    ;;
esac

if [[ ${DUALPATH_SKIP_ALLOCATION_CHECK:-0} != 1 ]]; then
  mapfile -t allocated_hosts < <(
    scontrol show hostnames "$(squeue -h -j "$allocation_id" -o '%N')"
  )
  declare -A allocated=()
  for host in "${allocated_hosts[@]}"; do
    allocated[$host]=1
  done
  for host in "$current_host" "${prefill_hosts[@]}" "${decode_hosts[@]}" \
    "${CLIENT_HOST:-${ROUTER_HOST:-}}"; do
    [[ -n $host ]] || continue
    [[ -n ${allocated[$host]:-} ]] || {
      echo "$host is not part of allocation $allocation_id" >&2
      exit 2
    }
  done
fi

export CODE_HOME RUNTIME_HOME IMAGE TASK_NAME PREFILL_HOSTS DECODE_HOSTS
export ENGINE_GPUS TP_SIZE
export ROUTER_HOST=${ROUTER_HOST:-${CLIENT_HOST:-$current_host}}
if [[ $role == client ]]; then
  runtime_role=client
  runtime_id=0
  cpus=${CLIENT_CPUS:-8}
  gres=none
else
  runtime_role=$engine_role
  runtime_id=$engine_id
  cpus=${ENGINE_CPUS:-192}
  gres="gpu:$ENGINE_GPUS"
fi
[[ $cpus =~ ^[1-9][0-9]*$ ]] || { echo "Invalid CPU count: $cpus" >&2; exit 2; }

runtime_cache_root=${DUALPATH_RUNTIME_CACHE_ROOT:-$RUNTIME_HOME/runtime-cache}
runtime_home=$runtime_cache_root/home
xdg_cache_home=$runtime_cache_root/xdg
flashinfer_workspace=$runtime_cache_root/flashinfer
triton_cache_dir=$runtime_cache_root/triton
torch_extensions_dir=$runtime_cache_root/torch-extensions
torchinductor_cache_dir=$runtime_cache_root/torchinductor
mkdir -p "$runtime_home" "$xdg_cache_home" "$flashinfer_workspace" \
  "$triton_cache_dir" "$torch_extensions_dir" "$torchinductor_cache_dir"

if [[ $role == client ]]; then
  threads_per_process=${OMP_NUM_THREADS:-$cpus}
else
  if (( cpus % TP_SIZE != 0 )); then
    echo "ENGINE_CPUS ($cpus) must be divisible by TP_SIZE ($TP_SIZE)" >&2
    exit 2
  fi
  threads_per_process=${ENGINE_THREADS_PER_RANK:-$((cpus / TP_SIZE))}
fi
[[ $threads_per_process =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid threads per process: $threads_per_process" >&2
  exit 2
}

step_sharing=--exclusive
if [[ $role == client ]]; then
  step_sharing=--overlap
fi
command=(
  srun --jobid="$allocation_id" "$step_sharing" --quit-on-interrupt --unbuffered
  --nodes=1 --ntasks=1 --nodelist="$current_host"
  --cpus-per-task="$cpus" --gres="$gres" --mpi=none
  --container-image="$IMAGE"
  --container-mounts="$RUNTIME_HOME:$RUNTIME_HOME,$CODE_HOME:/workspace/dual-path"
  --container-workdir=/workspace/dual-path --no-container-mount-home
  env CODE_HOME=/workspace/dual-path
  HOME="$runtime_home"
  XDG_CACHE_HOME="$xdg_cache_home"
  FLASHINFER_WORKSPACE_BASE="$flashinfer_workspace"
  TRITON_CACHE_DIR="$triton_cache_dir"
  TORCH_EXTENSIONS_DIR="$torch_extensions_dir"
  TORCHINDUCTOR_CACHE_DIR="$torchinductor_cache_dir"
  OMP_NUM_THREADS="$threads_per_process"
  MKL_NUM_THREADS="$threads_per_process"
  bash scripts/tests/runtime/pd_runtime.sh "$runtime_role" "$runtime_id"
)

printf 'allocation=%s host=%s role=%s id=%s task=%s tp=%s l3_read=%s o_direct=%s\n' \
  "$allocation_id" "$current_host" "$runtime_role" "$runtime_id" \
  "${TASK_NAME:-<client-default>}" "$TP_SIZE" "$DUALPATH_L3_READ_PATH" \
  "${DUAL_PATH_O_DIRECT_READ:-0}"
printf '  cpus=%s threads_per_process=%s runtime_cache=%s\n' \
  "$cpus" "$threads_per_process" "$runtime_cache_root"
if [[ ${PD_DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
