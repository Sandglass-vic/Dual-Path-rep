#!/usr/bin/env bash
# Formal 1P1D Mooncake Store dynamic full-rollout latency matrix.
#SBATCH --job-name=qwen32b-1p1d-bench
#SBATCH --partition=GAIA
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=06:00:00
# Known platform-faulty nodes; keep in sync with docs/platform.md.
#SBATCH --exclude=dgx-gaia-18,dgx-gaia-30,dgx-gaia-46,dgx-gaia-53
#SBATCH --output=/mnt/lustre/users/sezhou/dual-path/logs/slurm-1p1d-benchmark-%j.out

set -euo pipefail

CODE_HOME=${CODE_HOME:-/labhome/sezhou/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/mnt/lustre/users/sezhou/dual-path}
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3-mooncake0.3.13.post1-timing-v2.sqsh}
MODEL_DIR=${MODEL_DIR:-$RUNTIME_HOME/models/Qwen2.5-32B}
TASK_NAME=${TASK_NAME:-${SLURM_JOB_NAME}-${SLURM_JOB_ID}}
if [[ "$TASK_NAME" == */* || -z "$TASK_NAME" ]]; then
  echo "TASK_NAME must be a non-empty directory name without /" >&2
  exit 2
fi
LOG_DIR=${LOG_DIR:-$RUNTIME_HOME/logs/latency-benchmark/$TASK_NAME}
RESULT_DIR=${RESULT_DIR:-$RUNTIME_HOME/results/latency-benchmark/$TASK_NAME}
CACHE_DIR=${CACHE_DIR:-$RUNTIME_HOME/kv-cache/latency-benchmark/$TASK_NAME}
REPO_RESULT_DIR=${REPO_RESULT_DIR:-$CODE_HOME/results/latency-benchmark/$TASK_NAME}
IB_DEVICE=${IB_DEVICE:-mlx5_0}
PREFILL_PORT=${PREFILL_PORT:-30000}
DECODE_PORT=${DECODE_PORT:-30000}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
ROUTER_PORT=${ROUTER_PORT:-8000}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-1200}
REPETITIONS=${REPETITIONS:-3}
REP_LIST=${REP_LIST:-}
MAX_TURNS=${MAX_TURNS:-0}
TRAJECTORY_OFFSET=${TRAJECTORY_OFFSET:-0}
# DATASETS entries: N/NK uses the standard fake dataset; NK=/path selects a file.
DATASETS=${DATASETS:-${MAX_LENGTHS:-"32"}}
AGENT_COUNTS=${AGENT_COUNTS:-"32 64 128"}
ABLATION_PROFILE=${ABLATION_PROFILE:-full}
PAPER_EXPAND_AGENTS=${PAPER_EXPAND_AGENTS:-1}
PAPER_SYNTHETIC_APPEND_TOKENS=${PAPER_SYNTHETIC_APPEND_TOKENS:-1}
ARCHITECTURE=1P1D

# shellcheck source=scripts/ablation_profiles.sh
source "$CODE_HOME/scripts/ablation_profiles.sh"
# shellcheck source=scripts/pd_benchmark_lib.sh
source "$CODE_HOME/scripts/pd_benchmark_lib.sh"
apply_ablation_profile "$ABLATION_PROFILE"

mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ ${#hosts[@]} -ne 2 ]]; then
  echo "Expected exactly two nodes, got: ${hosts[*]}" >&2
  exit 1
fi

export PREFILL_HOST=${hosts[0]}
export DECODE_HOST=${hosts[1]}
export DECODE_HOSTS=$DECODE_HOST
dualpath_export_common_env

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$CACHE_DIR" "$REPO_RESULT_DIR"
cache_marker="$CACHE_DIR/.dual-path-test-cache-$SLURM_JOB_ID"
touch "$cache_marker"
printf '%s\n' "${hosts[@]}" >"$RESULT_DIR/nodes.txt"
print_ablation_profile | tee "$RESULT_DIR/ablation-profile.env"
printf 'ARCHITECTURE=%s\n' "$ARCHITECTURE" | tee -a "$RESULT_DIR/ablation-profile.env"

step_pids=()
cleanup() {
  dualpath_stop_steps
  if [[ -f "$cache_marker" ]]; then
    rm -rf -- "$CACHE_DIR"
    echo "Removed test KV cache: $CACHE_DIR"
  fi
}
trap cleanup EXIT INT TERM

launch_cluster() {
  local run_id=$1
  local run_cache_dir=$2
  local start_mooncake_master=1
  if [[ "$MEASURED_PREFILL_L3" == 0 && "$MEASURED_DECODE_OFFLOAD" == 0 ]]; then
    start_mooncake_master=0
  fi
  export MOONCAKE_DFS_ROOT_DIR="$run_cache_dir/dfs"
  export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="$run_cache_dir/client-init/prefill"
  mkdir -p "$MOONCAKE_DFS_ROOT_DIR" "$MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"
  touch "$run_cache_dir/.dual-path-run-cache-$SLURM_JOB_ID"
  export RUN_ID=$run_id

  dualpath_container_srun "$PREFILL_HOST" \
    env ENABLE_HICACHE_L3="$MEASURED_PREFILL_L3" \
      START_MOONCAKE_MASTER="$start_mooncake_master" \
      MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="$run_cache_dir/client-init/prefill" \
      bash scripts/pd_prefill_router.sh &
  step_pids+=("$!")
  prefill_step_pid=$!

  dualpath_container_srun "$DECODE_HOST" \
    env ENABLE_HICACHE_L3="$MEASURED_DECODE_OFFLOAD" \
      DECODE_OFFLOAD_KVCACHE="$MEASURED_DECODE_OFFLOAD" \
      MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="$run_cache_dir/client-init/decode" \
      DECODE_HOST="$DECODE_HOST" bash scripts/pd_decode.sh &
  step_pids+=("$!")
  decode_step_pid=$!

  dualpath_wait_for_url decode "http://$DECODE_HOST:$DECODE_PORT" "$decode_step_pid"
  dualpath_wait_for_url router "http://$PREFILL_HOST:$ROUTER_PORT" "$prefill_step_pid"
  echo "$run_id cluster is healthy"
}

remove_run_cache() {
  local run_cache_dir=$1
  if [[ ! -f "$run_cache_dir/.dual-path-run-cache-$SLURM_JOB_ID" ]]; then
    echo "Refusing to remove unmarked run cache: $run_cache_dir" >&2
    return 1
  fi
  rm -rf -- "$run_cache_dir"
  echo "Removed completed run KV cache: $run_cache_dir"
}

mapfile -t repetition_ids < <(dualpath_repetition_ids)
if (( ${#repetition_ids[@]} == 0 )); then
  echo "No repetition IDs configured" >&2
  exit 2
fi
for dataset_spec in $DATASETS; do
  dualpath_resolve_dataset "$dataset_spec"
  dataset_slug=${DATASET_LABEL,,}
  for repetition in "${repetition_ids[@]}"; do
    for agents in $AGENT_COUNTS; do
      run_dir="$RESULT_DIR/$DATASET_LABEL/agents-${agents}/rep-${repetition}"
      run_id="${dataset_slug}-a${agents}-r${repetition}"
      run_cache_dir="$CACHE_DIR/$run_id"
      mkdir -p "$run_dir"
      launch_cluster "$run_id" "$run_cache_dir"
      dualpath_write_run_json \
        "$run_dir/run.json" "$ARCHITECTURE" "$DATASET_MAX_LEN" \
        "$agents" "$repetition" "$DATASET_LABEL"
      dualpath_run_replay \
        "$DATASET_PATH" "$agents" "$run_dir" \
        "bench-${ARCHITECTURE,,}-${dataset_slug}-a${agents}-r${repetition}-"
      dualpath_stop_steps
      remove_run_cache "$run_cache_dir"
    done
  done
done

python "$CODE_HOME/scripts/benchmark/collect_benchmark_metrics.py" \
  --result-dir "$RESULT_DIR" \
  --log-dir "$LOG_DIR" \
  --repo-result-dir "$REPO_RESULT_DIR" \
  --code-home "$CODE_HOME"

echo "BENCHMARK_COMPLETE architecture=$ARCHITECTURE result_dir=$RESULT_DIR"
echo "BENCHMARK_REPO_RESULTS=$REPO_RESULT_DIR"
