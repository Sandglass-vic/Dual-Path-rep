#!/usr/bin/env bash
# Measure buffered hot-cache reads and O_DIRECT reads over the same hot files.
set -euo pipefail

CODE_HOME=${CODE_HOME:-/labhome/sezhou/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/mnt/lustre/users/sezhou/dual-path}
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang-file-l3-dual-staging-69054.sqsh}
TP_RANKS=${TP_RANKS:-8}
PAGES_PER_RANK=${PAGES_PER_RANK:-512}
PAGE_BYTES=${PAGE_BYTES:-2097152}
REPETITIONS=${REPETITIONS:-3}
CPUS_PER_NODE=${CPUS_PER_NODE:-112}
DIRECT_ALIGNMENT=${DIRECT_ALIGNMENT:-4096}
BUFFER_TYPE=${BUFFER_TYPE:-pinned}

: "${SLURM_JOB_ID:?Run this script inside a GPU-node allocation}"
reader_host=${READER_HOST:-$(scontrol show hostnames "${SLURM_JOB_NODELIST:?}" | head -n 1)}
run_root=${RUN_ROOT:-$RUNTIME_HOME/io-test/$SLURM_JOB_ID-hot-cache}
data_dir=${DATA_DIR:-$run_root/kv-pages}
result_dir=${LOCAL_RESULT_DIR:-$CODE_HOME/io-test/results/$SLURM_JOB_ID-hot-cache}
result_log=$result_dir/read-results.log
summary_json=$result_dir/summary.json
mkdir -p "$result_dir"
: >"$result_log"

container_args=(
  --container-image="$IMAGE"
  --container-mounts="/mnt/lustre:/mnt/lustre,$CODE_HOME:/workspace/dual-path"
  --container-workdir=/workspace/dual-path
  --no-container-mount-home
)

run_python() {
  srun --jobid="$SLURM_JOB_ID" --overlap --unbuffered --label \
    --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="$reader_host" \
    --cpus-per-task="$CPUS_PER_NODE" --gres=gpu:1 --mpi=none \
    "${container_args[@]}" python3 /workspace/dual-path/io-test/kv_read_bench.py "$@"
}

echo "DualPath hot page-cache versus O_DIRECT mini bench"
echo "  allocation=$SLURM_JOB_ID reader=$reader_host"
echo "  workload=$((TP_RANKS * PAGES_PER_RANK * PAGE_BYTES / 2**30)) GiB per phase"

run_python prepare --data-dir "$data_dir" --tp-ranks "$TP_RANKS" \
  --pages-per-rank "$PAGES_PER_RANK" --page-bytes "$PAGE_BYTES"
# cache-probe calls sync(2), ensuring the subsequent DONTNEED can evict prepared data.
run_python cache-probe --data-dir "$data_dir" --tp-ranks "$TP_RANKS" \
  --pages-per-rank "$PAGES_PER_RANK" --page-bytes "$PAGE_BYTES" \
  --buffer-type "$BUFFER_TYPE" --direct-alignment "$DIRECT_ALIGNMENT" \
  | tee -a "$result_log"

run_phase() {
  local phase=$1 mode=$2 repetition=$3 drop=$4
  local drop_args=()
  if [[ $drop == 1 ]]; then
    drop_args+=(--drop-cache)
  fi
  echo "Running phase=$phase io_mode=$mode repetition=$repetition drop_cache=$drop"
  run_python read --data-dir "$data_dir" --tp-ranks "$TP_RANKS" \
    --pages-per-rank "$PAGES_PER_RANK" --page-bytes "$PAGE_BYTES" \
    --passes 1 --buffer-type "$BUFFER_TYPE" "${drop_args[@]}" \
    --copy-to-staging --copy-implementation torch --torch-threads 1 \
    --layers 64 --page-tokens 64 --head-dim 128 --nodes 1 \
    --phase "$phase" --repetition "$repetition" --io-mode "$mode" \
    --direct-alignment "$DIRECT_ALIGNMENT" | tee -a "$result_log"
}

for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
  run_phase buffered-populate buffered "$repetition" 1
  run_phase buffered-hot buffered "$repetition" 0
  run_phase direct-over-hot-cache direct "$repetition" 0
  run_phase buffered-hot-after-direct buffered "$repetition" 0
done

python3 "$CODE_HOME/io-test/summarize_hot_cache.py" "$result_log" --output "$summary_json"
run_python cleanup --data-dir "$data_dir"
echo "KVIO_HOT_CACHE_BENCH_COMPLETE result_log=$result_log summary=$summary_json"
