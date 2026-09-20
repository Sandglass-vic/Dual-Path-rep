#!/usr/bin/env bash
# Measure the cache/O_DIRECT matrix for Decode-style KV page reads on one node.
#
# Four quadrants over one identical 8 GiB page set:
#
#   cold-buffered  POSIX_FADV_DONTNEED, then buffered read
#   hot-buffered   no eviction, buffered read over the cache just populated
#   hot-direct     no eviction, O_DIRECT read over that same hot cache
#   cold-direct    POSIX_FADV_DONTNEED, then O_DIRECT read
#
# Every quadrant runs the same open/readinto/close loop and the same
# page_first -> layer_first staging copy, and reports the file I/O and the
# staging scatter as separate timings.
set -euo pipefail

CODE_HOME=${CODE_HOME:-/labhome/sezhou/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/mnt/lustre/users/sezhou/dual-path}
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang-file-l3-dual-staging-69054.sqsh}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-/mnt/lustre:/mnt/lustre,$CODE_HOME:/workspace/dual-path}
TP_RANKS=${TP_RANKS:-8}
PAGES_PER_RANK=${PAGES_PER_RANK:-512}
PAGE_BYTES=${PAGE_BYTES:-2097152}
REPETITIONS=${REPETITIONS:-3}
CPUS_PER_NODE=${CPUS_PER_NODE:-112}
DIRECT_ALIGNMENT=${DIRECT_ALIGNMENT:-4096}
BUFFER_TYPE=${BUFFER_TYPE:-pinned}
COPY_TO_STAGING=${COPY_TO_STAGING:-1}
COPY_IMPLEMENTATION=${COPY_IMPLEMENTATION:-torch}
TORCH_THREADS=${TORCH_THREADS:-1}
LAYERS=${LAYERS:-64}
PAGE_TOKENS=${PAGE_TOKENS:-64}
HEAD_DIM=${HEAD_DIM:-128}
GRES=${GRES:-gpu:1}
KEEP_DATA=${KEEP_DATA:-0}
# The reader only evicts its own page cache. On g6 the files also live in g5's
# NFS server cache, so a g6 cold phase measures the storage network path unless
# the peer evicts them too.
PEER_HOST=${PEER_HOST:-}
DROP_PEER_CACHE=${DROP_PEER_CACHE:-0}

: "${SLURM_JOB_ID:?Run this script inside a GPU-node allocation}"
reader_host=${READER_HOST:-$(scontrol show hostnames "${SLURM_JOB_NODELIST:?}" | head -n 1)}
# The reader host is part of the default paths so that running the matrix for
# g5 and then g6 inside one allocation does not overwrite the first result.
run_root=${RUN_ROOT:-$RUNTIME_HOME/io-test/$SLURM_JOB_ID-cache-matrix-$reader_host}
data_dir=${DATA_DIR:-$run_root/kv-pages}
result_dir=${LOCAL_RESULT_DIR:-$CODE_HOME/io-test/results/$SLURM_JOB_ID-cache-matrix-$reader_host}
result_log=$result_dir/read-results.log
records_dir=$result_dir/records
summary_json=$result_dir/summary.json
mkdir -p "$records_dir"
: >"$result_log"

for value in TP_RANKS PAGES_PER_RANK PAGE_BYTES REPETITIONS CPUS_PER_NODE \
  DIRECT_ALIGNMENT TORCH_THREADS LAYERS PAGE_TOKENS HEAD_DIM; do
  [[ ${!value} =~ ^[1-9][0-9]*$ ]] || { echo "$value must be a positive integer" >&2; exit 2; }
done
[[ $BUFFER_TYPE == pinned || $BUFFER_TYPE == pageable ]] || {
  echo "BUFFER_TYPE must be pinned or pageable" >&2
  exit 2
}
if [[ $DROP_PEER_CACHE == 1 && -z $PEER_HOST ]]; then
  echo "DROP_PEER_CACHE=1 requires PEER_HOST" >&2
  exit 2
fi

container_args=(
  --container-image="$IMAGE"
  --container-mounts="$CONTAINER_MOUNTS"
  --container-workdir=/workspace/dual-path
  --no-container-mount-home
)

run_python() {
  local host=$1
  shift
  # No --label: every step here is a single task, and srun's labelled output
  # path splits lines longer than 1000 characters mid-record.
  srun --jobid="$SLURM_JOB_ID" --overlap --unbuffered \
    --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="$host" \
    --cpus-per-task="$CPUS_PER_NODE" --gres="$GRES" --mpi=none \
    "${container_args[@]}" python3 /workspace/dual-path/io-test/kv_read_bench.py "$@"
}

layout=(
  --data-dir "$data_dir" --tp-ranks "$TP_RANKS"
  --pages-per-rank "$PAGES_PER_RANK" --page-bytes "$PAGE_BYTES"
)

echo "DualPath cache/O_DIRECT matrix for Decode KV page reads"
echo "  allocation=$SLURM_JOB_ID reader=$reader_host peer=${PEER_HOST:-none}"
echo "  data_dir=$data_dir result_dir=$result_dir"
echo "  workload=$((TP_RANKS * PAGES_PER_RANK * PAGE_BYTES / 2**30)) GiB per quadrant"
echo "  tp_ranks=$TP_RANKS pages_per_rank=$PAGES_PER_RANK page_bytes=$PAGE_BYTES"
echo "  buffer_type=$BUFFER_TYPE staging=$COPY_TO_STAGING/$COPY_IMPLEMENTATION"
echo "  drop_peer_cache=$DROP_PEER_CACHE repetitions=$REPETITIONS"

run_python "$reader_host" prepare "${layout[@]}" 2>&1 | tee -a "$result_log"
# The probe verifies with mincore that O_DIRECT really bypasses the page cache
# on this filesystem before any bandwidth number is attributed to it.
run_python "$reader_host" cache-probe "${layout[@]}" \
  --buffer-type "$BUFFER_TYPE" --direct-alignment "$DIRECT_ALIGNMENT" \
  2>&1 | tee -a "$result_log"

copy_args=()
if [[ $COPY_TO_STAGING == 1 ]]; then
  copy_args+=(--copy-to-staging)
fi

run_quadrant() {
  local phase=$1 repetition=$2
  local mode drop_args=()
  case $phase in
    cold-buffered) mode=buffered; drop_args+=(--drop-cache) ;;
    hot-buffered) mode=buffered ;;
    hot-direct) mode=direct ;;
    cold-direct) mode=direct; drop_args+=(--drop-cache) ;;
    *) echo "unknown quadrant: $phase" >&2; return 2 ;;
  esac
  if [[ ${#drop_args[@]} -gt 0 && $DROP_PEER_CACHE == 1 ]]; then
    run_python "$PEER_HOST" drop-cache "${layout[@]}" 2>&1 | tee -a "$result_log"
  fi
  echo "Running quadrant=$phase io_mode=$mode repetition=$repetition"
  run_python "$reader_host" read "${layout[@]}" \
    --passes 1 --buffer-type "$BUFFER_TYPE" "${drop_args[@]}" \
    "${copy_args[@]}" --copy-implementation "$COPY_IMPLEMENTATION" \
    --torch-threads "$TORCH_THREADS" \
    --layers "$LAYERS" --page-tokens "$PAGE_TOKENS" --head-dim "$HEAD_DIM" \
    --nodes 1 --phase "$phase" --repetition "$repetition" --io-mode "$mode" \
    --direct-alignment "$DIRECT_ALIGNMENT" \
    --output "$records_dir/$phase.r$repetition.json" 2>&1 | tee -a "$result_log"
}

# hot-* must follow a buffered read that leaves the cache populated. The two
# orderings keep that invariant while alternating which quadrant sees the
# filesystem first, so a warm-up trend cannot be mistaken for a mode effect.
for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
  if (( repetition % 2 == 1 )); then
    run_quadrant cold-buffered "$repetition"
    run_quadrant hot-buffered "$repetition"
    run_quadrant hot-direct "$repetition"
    run_quadrant cold-direct "$repetition"
  else
    run_quadrant cold-direct "$repetition"
    run_quadrant cold-buffered "$repetition"
    run_quadrant hot-buffered "$repetition"
    run_quadrant hot-direct "$repetition"
  fi
done

python3 "$CODE_HOME/io-test/summarize_hot_cache.py" "$result_log" \
  --records-dir "$records_dir" --output "$summary_json" --host "$reader_host"
if [[ $KEEP_DATA == 0 ]]; then
  run_python "$reader_host" cleanup --data-dir "$data_dir" 2>&1 | tee -a "$result_log"
fi
echo "KVIO_CACHE_MATRIX_COMPLETE result_log=$result_log summary=$summary_json"
