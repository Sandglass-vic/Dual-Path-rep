#!/usr/bin/env bash
# Compare Decode-style buffered and O_DIRECT Lustre reads on one compute node.
set -euo pipefail

CODE_HOME=${CODE_HOME:-/labhome/sezhou/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/mnt/lustre/users/sezhou/dual-path}
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang-file-l3-dual-staging-69054.sqsh}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-/mnt/lustre:/mnt/lustre,$CODE_HOME:/workspace/dual-path}
TP_RANKS=${TP_RANKS:-8}
PAGES_PER_RANK=${PAGES_PER_RANK:-512}
PAGE_BYTES=${PAGE_BYTES:-2097152}
PASSES=${PASSES:-1}
REPETITIONS=${REPETITIONS:-3}
BUFFER_TYPE=${BUFFER_TYPE:-pinned}
CPUS_PER_NODE=${CPUS_PER_NODE:-112}
COPY_TO_STAGING=${COPY_TO_STAGING:-1}
TORCH_THREADS=${TORCH_THREADS:-1}
DIRECT_ALIGNMENT=${DIRECT_ALIGNMENT:-4096}
KEEP_DATA=${KEEP_DATA:-0}
LAYERS=${LAYERS:-64}
PAGE_TOKENS=${PAGE_TOKENS:-64}
HEAD_DIM=${HEAD_DIM:-128}

: "${SLURM_JOB_ID:?Run this script inside a GPU-node allocation}"
reader_host=${READER_HOST:-$(scontrol show hostnames "${SLURM_JOB_NODELIST:?}" | head -n 1)}
RUN_ROOT=${RUN_ROOT:-$RUNTIME_HOME/io-test/$SLURM_JOB_ID-direct}
DATA_DIR=${DATA_DIR:-$RUN_ROOT/kv-pages}
LOCAL_RESULT_DIR=${LOCAL_RESULT_DIR:-$CODE_HOME/io-test/results/$SLURM_JOB_ID-direct}
RESULT_LOG=${RESULT_LOG:-$LOCAL_RESULT_DIR/read-results.log}
SUMMARY_JSON=${SUMMARY_JSON:-$LOCAL_RESULT_DIR/summary.json}
mkdir -p "$LOCAL_RESULT_DIR"
: >"$RESULT_LOG"

container_args=(
  --container-image="$IMAGE"
  --container-mounts="$CONTAINER_MOUNTS"
  --container-workdir=/workspace/dual-path
  --no-container-mount-home
)

run_python() {
  srun --jobid="$SLURM_JOB_ID" --overlap --unbuffered --label \
    --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="$reader_host" \
    --cpus-per-task="$CPUS_PER_NODE" --gres=gpu:1 --mpi=none \
    "${container_args[@]}" python3 /workspace/dual-path/io-test/kv_read_bench.py "$@"
}

echo "DualPath buffered versus O_DIRECT KV read mini bench"
echo "  allocation=$SLURM_JOB_ID reader=$reader_host data_dir=$DATA_DIR"
echo "  tp_ranks=$TP_RANKS pages_per_rank=$PAGES_PER_RANK page_bytes=$PAGE_BYTES"
echo "  bytes_per_mode=$((TP_RANKS * PAGES_PER_RANK * PAGE_BYTES)) repetitions=$REPETITIONS"
echo "  buffer_type=$BUFFER_TYPE direct_alignment=$DIRECT_ALIGNMENT"

run_python prepare --data-dir "$DATA_DIR" --tp-ranks "$TP_RANKS" \
  --pages-per-rank "$PAGES_PER_RANK" --page-bytes "$PAGE_BYTES"

run_python cache-probe --data-dir "$DATA_DIR" --tp-ranks "$TP_RANKS" \
  --pages-per-rank "$PAGES_PER_RANK" --page-bytes "$PAGE_BYTES" \
  --buffer-type "$BUFFER_TYPE" --direct-alignment "$DIRECT_ALIGNMENT" \
  | tee -a "$RESULT_LOG"

copy_args=()
if [[ $COPY_TO_STAGING == 1 ]]; then
  copy_args+=(--copy-to-staging)
fi

run_mode() {
  local mode=$1 repetition=$2
  echo "Running io_mode=$mode repetition=$repetition"
  run_python read --data-dir "$DATA_DIR" --tp-ranks "$TP_RANKS" \
    --pages-per-rank "$PAGES_PER_RANK" --page-bytes "$PAGE_BYTES" \
    --passes "$PASSES" --buffer-type "$BUFFER_TYPE" --drop-cache \
    "${copy_args[@]}" --copy-implementation torch --torch-threads "$TORCH_THREADS" \
    --layers "$LAYERS" --page-tokens "$PAGE_TOKENS" --head-dim "$HEAD_DIM" \
    --phase "$mode" --repetition "$repetition" --nodes 1 --io-mode "$mode" \
    --direct-alignment "$DIRECT_ALIGNMENT" | tee -a "$RESULT_LOG"
}

for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
  if (( repetition % 2 == 1 )); then
    run_mode buffered "$repetition"
    run_mode direct "$repetition"
  else
    run_mode direct "$repetition"
    run_mode buffered "$repetition"
  fi
done

python3 "$CODE_HOME/io-test/summarize_direct.py" "$RESULT_LOG" --output "$SUMMARY_JSON"
if [[ $KEEP_DATA == 0 ]]; then
  run_python cleanup --data-dir "$DATA_DIR"
fi
echo "KVIO_DIRECT_BENCH_COMPLETE result_log=$RESULT_LOG summary=$SUMMARY_JSON"
