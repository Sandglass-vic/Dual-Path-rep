#!/usr/bin/env bash
# Compare one-node and two-node Decode-style Lustre page reads inside an salloc.
set -euo pipefail

CODE_HOME=${CODE_HOME:-/labhome/sezhou/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/mnt/lustre/users/sezhou/dual-path}
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang-file-l3-dual-staging-67876.sqsh}
TP_RANKS=${TP_RANKS:-8}
PAGES_PER_RANK=${PAGES_PER_RANK:-512}
PAGE_BYTES=${PAGE_BYTES:-2097152}
PASSES=${PASSES:-1}
REPETITIONS=${REPETITIONS:-3}
BUFFER_TYPE=${BUFFER_TYPE:-pinned}
CPUS_PER_NODE=${CPUS_PER_NODE:-112}
DROP_CACHE=${DROP_CACHE:-1}
COPY_TO_STAGING=${COPY_TO_STAGING:-1}
COPY_IMPLEMENTATION=${COPY_IMPLEMENTATION:-torch}
TORCH_THREADS=${TORCH_THREADS:-1}
KEEP_DATA=${KEEP_DATA:-0}
LAYERS=${LAYERS:-64}
PAGE_TOKENS=${PAGE_TOKENS:-64}
HEAD_DIM=${HEAD_DIM:-128}
SYNC_PORT=${SYNC_PORT:-29577}

: "${SLURM_JOB_ID:?Run this script inside an allocation containing at least two nodes}"
mapfile -t allocated_hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
if (( ${#allocated_hosts[@]} < 2 )); then
  echo "At least two allocated nodes are required" >&2
  exit 2
fi
if [[ -n ${READER_HOSTS:-} ]]; then
  read -r -a reader_hosts <<<"$READER_HOSTS"
else
  reader_hosts=("${allocated_hosts[0]}" "${allocated_hosts[1]}")
fi
if (( ${#reader_hosts[@]} != 2 )); then
  echo "READER_HOSTS must contain exactly two hosts" >&2
  exit 2
fi
for value in TP_RANKS PAGES_PER_RANK PAGE_BYTES PASSES REPETITIONS CPUS_PER_NODE TORCH_THREADS LAYERS PAGE_TOKENS HEAD_DIM SYNC_PORT; do
  [[ ${!value} =~ ^[1-9][0-9]*$ ]] || { echo "$value must be a positive integer" >&2; exit 2; }
done
[[ $BUFFER_TYPE == pinned || $BUFFER_TYPE == pageable ]] || {
  echo "BUFFER_TYPE must be pinned or pageable" >&2
  exit 2
}
[[ $COPY_IMPLEMENTATION == torch || $COPY_IMPLEMENTATION == python ]] || {
  echo "COPY_IMPLEMENTATION must be torch or python" >&2
  exit 2
}

RUN_ROOT=${RUN_ROOT:-$RUNTIME_HOME/io-test/$SLURM_JOB_ID}
DATA_DIR=${DATA_DIR:-$RUN_ROOT/kv-pages}
LOCAL_RESULT_DIR=${LOCAL_RESULT_DIR:-$CODE_HOME/io-test/results/$SLURM_JOB_ID}
RESULT_LOG=${RESULT_LOG:-$LOCAL_RESULT_DIR/read-results.log}
SUMMARY_JSON=${SUMMARY_JSON:-$LOCAL_RESULT_DIR/summary.json}
mkdir -p "$LOCAL_RESULT_DIR"
: >"$RESULT_LOG"

container_args=(
  --container-image="$IMAGE"
  --container-mounts="/mnt/lustre:/mnt/lustre,$CODE_HOME:/workspace/dual-path"
  --container-workdir=/workspace/dual-path
  --no-container-mount-home
)

run_python() {
  local nodes=$1
  local hosts=$2
  shift 2
  srun --jobid="$SLURM_JOB_ID" --overlap --unbuffered --label \
    --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 \
    --nodelist="$hosts" --cpus-per-task="$CPUS_PER_NODE" \
    --gres=gpu:1 --mpi=none "${container_args[@]}" \
    python3 /workspace/dual-path/io-test/kv_read_bench.py "$@"
}

echo "DualPath Decode-style Lustre read mini bench"
echo "  allocation=$SLURM_JOB_ID"
echo "  reader-0=${reader_hosts[0]} reader-1=${reader_hosts[1]}"
echo "  data_dir=$DATA_DIR"
echo "  local_result_dir=$LOCAL_RESULT_DIR"
echo "  tp_ranks=$TP_RANKS pages_per_rank=$PAGES_PER_RANK page_bytes=$PAGE_BYTES"
echo "  bytes_per_node=$((TP_RANKS * PAGES_PER_RANK * PAGE_BYTES)) repetitions=$REPETITIONS"
echo "  buffer_type=$BUFFER_TYPE drop_local_cache=$DROP_CACHE"
echo "  copy_to_layer_first_staging=$COPY_TO_STAGING implementation=$COPY_IMPLEMENTATION"
echo "  torch_threads_per_rank=$TORCH_THREADS layers=$LAYERS page_tokens=$PAGE_TOKENS head_dim=$HEAD_DIM"

echo "Preparing synthetic KV page files on ${reader_hosts[0]}..."
run_python 1 "${reader_hosts[0]}" prepare \
  --data-dir "$DATA_DIR" --tp-ranks "$TP_RANKS" \
  --pages-per-rank "$PAGES_PER_RANK" --page-bytes "$PAGE_BYTES"

drop_args=()
if [[ $DROP_CACHE == 1 ]]; then
  drop_args+=(--drop-cache)
fi
copy_args=()
if [[ $COPY_TO_STAGING == 1 ]]; then
  copy_args+=(--copy-to-staging)
fi
host_csv="${reader_hosts[0]},${reader_hosts[1]}"

run_read_phase() {
  local phase=$1
  local repetition=$2
  local nodes hosts
  if [[ $phase == single ]]; then
    nodes=1
    hosts=${reader_hosts[0]}
  else
    nodes=2
    hosts=$host_csv
  fi
  echo "Running phase=$phase repetition=$repetition hosts=$hosts"
  run_python "$nodes" "$hosts" read \
    --data-dir "$DATA_DIR" --tp-ranks "$TP_RANKS" \
    --pages-per-rank "$PAGES_PER_RANK" --page-bytes "$PAGE_BYTES" \
    --passes "$PASSES" --buffer-type "$BUFFER_TYPE" "${drop_args[@]}" \
    "${copy_args[@]}" --copy-implementation "$COPY_IMPLEMENTATION" \
    --torch-threads "$TORCH_THREADS" \
    --layers "$LAYERS" --page-tokens "$PAGE_TOKENS" --head-dim "$HEAD_DIM" \
    --sync-host "${reader_hosts[0]}" --sync-port "$SYNC_PORT" \
    --phase "$phase" --repetition "$repetition" --nodes "$nodes" \
    | tee -a "$RESULT_LOG"
}

# Alternate phase order to reduce systematic warm-cache and temporal-load bias.
for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
  if (( repetition % 2 == 1 )); then
    run_read_phase single "$repetition"
    run_read_phase dual "$repetition"
  else
    run_read_phase dual "$repetition"
    run_read_phase single "$repetition"
  fi
done

python3 "$CODE_HOME/io-test/summarize.py" "$RESULT_LOG" --output "$SUMMARY_JSON"
if [[ $KEEP_DATA == 0 ]]; then
  echo "Removing synthetic Lustre page files..."
  run_python 1 "${reader_hosts[0]}" cleanup --data-dir "$DATA_DIR"
fi
echo "KVIO_BENCH_COMPLETE result_log=$RESULT_LOG summary=$SUMMARY_JSON"
if [[ $KEEP_DATA == 1 ]]; then
  echo "Lustre cleanup: rm -rf $RUN_ROOT"
fi
