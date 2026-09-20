#!/usr/bin/env bash
# End-to-end pza100 I/O characterization for g5 local /data and g6 NFS access.
set -Eeuo pipefail

CODE_HOME=${CODE_HOME:-/data/home/yongxiao/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/data/home/yongxiao/mydata}
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched.sqsh}
PARTITION=${PARTITION:-build}
HOSTS=${HOSTS:-g5,g6}
CPUS_PER_NODE=${CPUS_PER_NODE:-32}
TIME_LIMIT=${TIME_LIMIT:-04:00:00}
RUN_ID=${RUN_ID:-pza100-io-$(date +%Y%m%d-%H%M%S)}
BENCH_ROOT=${BENCH_ROOT:-$RUNTIME_HOME/dual-test/io-bench}
RUN_ROOT=${RUN_ROOT:-$BENCH_ROOT/$RUN_ID}
DATA_ROOT=$RUN_ROOT/data
RESULT_ROOT=$RUN_ROOT/results
MARKER=$RUN_ROOT/.dualpath-io-bench

REPETITIONS=${REPETITIONS:-3}
RAW_TOTAL_GIB=${RAW_TOTAL_GIB:-8}
# Each raw case costs 7 phases x modes x repetitions x 2 telemetry steps, so the
# default is one representative point. Widen it explicitly when there is time.
RAW_JOBS_LIST=${RAW_JOBS_LIST:-"4"}
RAW_BLOCK_SIZES=${RAW_BLOCK_SIZES:-"4M"}
RAW_MODES=${RAW_MODES:-"direct buffered-cold buffered-hot"}
# Restrict the raw matrix to a subset of phases, e.g. RAW_PHASES="g5-read g6-read"
# for a concurrency sweep that looks for the knee of the read path.
RAW_PHASES=${RAW_PHASES:-"g5-read g6-read dual-read g5-write g6-write dual-write mixed"}
RAW_BACKEND=${RAW_BACKEND:-auto}
TP_RANKS=${TP_RANKS:-4}
PAGES_PER_RANK=${PAGES_PER_RANK:-512}
PAGE_BYTES=${PAGE_BYTES:-4194304}
KV_MODES=${KV_MODES:-"io-only staging"}
WRITE_SYNC_MODES=${WRITE_SYNC_MODES:-"none batch-fdatasync"}
MATRIX_BUFFER_TYPE=${MATRIX_BUFFER_TYPE:-pinned}
MATRIX_COPY_IMPLEMENTATION=${MATRIX_COPY_IMPLEMENTATION:-torch}
# A cold g6 read only leaves g6's page cache; the files stay in g5's NFS server
# cache unless g5 evicts them too. 0 therefore measures the storage network
# path, 1 measures network plus NVMe.
MATRIX_DROP_PEER_CACHE=${MATRIX_DROP_PEER_CACHE:-0}
# Same idea for the raw phases, which have their own reader and never went
# through run_hot_cache.sh. Evicts the source from g5's page cache before each
# read phase so g6 and dual-read measure the NVMe instead of g5's DRAM.
# buffered-hot is skipped: warm cache is the point of that mode.
RAW_DROP_SERVER_CACHE=${RAW_DROP_SERVER_CACHE:-0}

RUN_MATRIX=${RUN_MATRIX:-1}
RUN_RAW=${RUN_RAW:-1}
RUN_KV=${RUN_KV:-1}
RUN_WRITE=${RUN_WRITE:-1}
KEEP_DATA=${KEEP_DATA:-0}
PZA100_IO_SMOKE=${PZA100_IO_SMOKE:-0}

if [[ $PZA100_IO_SMOKE == 1 ]]; then
  REPETITIONS=1
  RAW_TOTAL_GIB=1
  RAW_JOBS_LIST=1
  RAW_BLOCK_SIZES=4M
  RAW_MODES=direct
  PAGES_PER_RANK=8
  KV_MODES=io-only
  WRITE_SYNC_MODES=none
fi

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "Requesting an exclusive two-node pza100 allocation"
  exec salloc \
    --partition="$PARTITION" \
    --nodes=2 \
    --nodelist="$HOSTS" \
    --ntasks=2 \
    --ntasks-per-node=1 \
    --cpus-per-task="$CPUS_PER_NODE" \
    --gres=gpu:1 \
    --mem=0 \
    --exclusive \
    --time="$TIME_LIMIT" \
    bash "$0"
fi

mapfile -t allocated_hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
if (( ${#allocated_hosts[@]} != 2 )); then
  echo "ERROR: expected exactly two allocated nodes, got: ${allocated_hosts[*]}" >&2
  exit 2
fi
if [[ " ${allocated_hosts[*]} " != *" g5 "* || " ${allocated_hosts[*]} " != *" g6 "* ]]; then
  echo "ERROR: pza100 test requires g5 and g6, got: ${allocated_hosts[*]}" >&2
  exit 2
fi
for value in REPETITIONS RAW_TOTAL_GIB TP_RANKS PAGES_PER_RANK PAGE_BYTES CPUS_PER_NODE; do
  [[ ${!value} =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: $value must be a positive integer" >&2
    exit 2
  }
done
for path in \
  "$IMAGE" \
  "$CODE_HOME/io-test/kv_read_bench.py" \
  "$CODE_HOME/io-test/kv_write_load.py" \
  "$CODE_HOME/io-test/cluster_barrier.py" \
  "$CODE_HOME/io-test/posix_bandwidth_bench.py" \
  "$CODE_HOME/io-test/run_hot_cache.sh" \
  "$CODE_HOME/io-test/summarize_hot_cache.py" \
  "$CODE_HOME/io-test/marker_log.py" \
  "$CODE_HOME/io-test/capture_io_state.py"; do
  [[ -r $path ]] || { echo "ERROR: required path is not readable: $path" >&2; exit 2; }
done

mkdir -p "$BENCH_ROOT"
bench_root_real=$(realpath -e "$BENCH_ROOT")
run_parent_real=$(realpath -e "$(dirname "$RUN_ROOT")")
if [[ $run_parent_real != "$bench_root_real" || -e $RUN_ROOT || -L $RUN_ROOT ]]; then
  echo "ERROR: RUN_ROOT must be a new direct child of $bench_root_real: $RUN_ROOT" >&2
  exit 2
fi
mkdir "$RUN_ROOT"
mkdir "$DATA_ROOT" "$RESULT_ROOT"
printf '%s\n' "$RUN_ID" >"$MARKER"
git_commit=$(git -C "$CODE_HOME" rev-parse HEAD 2>/dev/null || printf unknown)
git_dirty=0
[[ -n $(git -C "$CODE_HOME" status --porcelain 2>/dev/null || true) ]] && git_dirty=1
image_sha256=${IMAGE_SHA256:-$(sha256sum "$IMAGE" | awk '{print $1}')}
python3 - "$RESULT_ROOT/config.json" \
  "$RUN_ID" "$SLURM_JOB_ID" "$IMAGE" "$image_sha256" "$git_commit" "$git_dirty" \
  "$REPETITIONS" "$RAW_TOTAL_GIB" "$RAW_JOBS_LIST" "$RAW_BLOCK_SIZES" \
  "$RAW_MODES" "$TP_RANKS" "$PAGES_PER_RANK" "$PAGE_BYTES" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    run_id,
    job_id,
    image,
    image_sha256,
    git_commit,
    git_dirty,
    repetitions,
    raw_total_gib,
    raw_jobs,
    raw_blocks,
    raw_modes,
    tp_ranks,
    pages_per_rank,
    page_bytes,
) = sys.argv[1:]
config = {
    "schema_version": 1,
    "run_id": run_id,
    "slurm_job_id": job_id,
    "image": image,
    "image_sha256": image_sha256,
    "git_commit": git_commit,
    "git_dirty": bool(int(git_dirty)),
    "repetitions": int(repetitions),
    "raw_total_gib": int(raw_total_gib),
    "raw_jobs": raw_jobs.split(),
    "raw_block_sizes": raw_blocks.split(),
    "raw_modes": raw_modes.split(),
    "tp_ranks": int(tp_ranks),
    "pages_per_rank": int(pages_per_rank),
    "page_bytes": int(page_bytes),
}
Path(output).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY
run_failed=1
partial_failure=0

cleanup() {
  local marker_value=
  if [[ -r $MARKER ]]; then
    marker_value=$(<"$MARKER")
  fi
  if [[ $run_failed == 0 && $KEEP_DATA == 0 && $marker_value == "$RUN_ID" ]]; then
    rm -rf -- "$DATA_ROOT"
    echo "Removed benchmark data: $DATA_ROOT"
  elif [[ -d $DATA_ROOT ]]; then
    echo "Preserved benchmark data: $DATA_ROOT"
  fi
}
trap cleanup EXIT

remove_phase_data() {
  local path=$1 marker_value=
  [[ -r $MARKER ]] && marker_value=$(<"$MARKER")
  if [[ $marker_value != "$RUN_ID" || $path != "$DATA_ROOT/"* ]]; then
    echo "ERROR: refusing phase cleanup outside marked data root: $path" >&2
    return 1
  fi
  rm -rf -- "$path"
}

container_mounts="/data:/data,$CODE_HOME:/workspace/dual-path:ro"
container_args=(
  --container-image="$IMAGE"
  --container-mounts="$container_mounts"
  --container-workdir=/workspace/dual-path
  --no-container-mount-home
)

run_step() {
  srun --jobid="$SLURM_JOB_ID" --overlap --unbuffered --label --mpi=none "$@"
}

capture_state() {
  local label=$1
  run_step \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 --nodelist=g5,g6 \
    --cpus-per-task=1 --gres=none \
    --export=ALL,CODE_HOME="$CODE_HOME",DATA_ROOT="$DATA_ROOT",RESULT_ROOT="$RESULT_ROOT",LABEL="$label" \
    bash -lc '
      host=$(hostname -s)
      python3 "$CODE_HOME/io-test/capture_io_state.py" \
        --path "$DATA_ROOT" --label "$LABEL" --interface ibs3 \
        --hca mlx5_0 --hca mlx5_4 \
        --output "$RESULT_ROOT/telemetry/$LABEL.$host.json"
    ' \
    | tee -a "$RESULT_ROOT/telemetry.log"
}

echo "DualPath pza100 cluster I/O benchmark"
echo "  allocation=$SLURM_JOB_ID hosts=${allocated_hosts[*]}"
echo "  run_root=$RUN_ROOT"
echo "  image=$IMAGE"
echo "  raw_total_gib=$RAW_TOTAL_GIB repetitions=$REPETITIONS"
echo "  tp_ranks=$TP_RANKS pages_per_rank=$PAGES_PER_RANK page_bytes=$PAGE_BYTES"

echo "[preflight] Capturing topology and tool availability"
run_step \
  --nodes=2 --ntasks=2 --ntasks-per-node=1 --nodelist=g5,g6 \
  --cpus-per-task=1 --gres=none \
  --export=ALL,DATA_ROOT="$DATA_ROOT",RESULT_ROOT="$RESULT_ROOT" \
  bash -lc '
    set -e
    host=$(hostname -s)
    {
      echo "HOST=$host"
      date --iso-8601=seconds
      findmnt -T "$DATA_ROOT" -o TARGET,SOURCE,FSTYPE,OPTIONS
      df -hT "$DATA_ROOT"
      if command -v fio; then
        fio --version
      else
        echo "fio unavailable"
      fi
      ip -s link show ibs3 || true
      nfsstat -m 2>/dev/null || true
    } >"$RESULT_ROOT/preflight.$host.txt"
  '
capture_state preflight
if [[ $RAW_BACKEND == auto ]]; then
  fio_probe=$(run_step \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 --nodelist=g5,g6 \
    --cpus-per-task=1 --gres=none \
    bash -lc 'if command -v fio >/dev/null; then echo FIO_OK; else echo FIO_MISSING; fi')
  if [[ $fio_probe == *FIO_MISSING* ]]; then
    RAW_BACKEND=python-posix
  else
    RAW_BACKEND=fio
  fi
fi
[[ $RAW_BACKEND == fio || $RAW_BACKEND == python-posix ]] || {
  echo "ERROR: RAW_BACKEND must be auto, fio, or python-posix" >&2
  exit 2
}
python3 - "$RESULT_ROOT/config.json" "$RAW_BACKEND" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["raw_backend"] = sys.argv[2]
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
echo "  raw_backend=$RAW_BACKEND"

prepare_raw_read_source() {
  local jobs=$1 block_size=$2 case_id=$3
  local size_mib=$((RAW_TOTAL_GIB * 1024 / jobs))
  local size_bytes=$((size_mib * 1024 * 1024))
  local block_bytes
  block_bytes=$(numfmt --from=iec "$block_size")
  local source_dir=$DATA_ROOT/raw/$case_id/read-source
  mkdir -p "$source_dir"
  if [[ $RAW_BACKEND == fio ]]; then
    run_step \
      --nodes=1 --ntasks=1 --nodelist=g5 --cpus-per-task="$CPUS_PER_NODE" --gres=none \
      fio --name=prepare --directory="$source_dir" --filename_format='source.$jobnum' \
        --rw=write --bs="$block_size" --ioengine=libaio --direct=1 --iodepth=16 \
        --numjobs="$jobs" --size="${size_mib}M" --group_reporting=1 --end_fsync=1 \
        --output-format=json --output="$RESULT_ROOT/raw/$case_id.prepare.json"
  else
    run_step \
      --nodes=1 --ntasks=1 --nodelist=g5 --cpus-per-task="$CPUS_PER_NODE" --gres=none \
      python3 "$CODE_HOME/io-test/posix_bandwidth_bench.py" prepare \
        --directory "$source_dir" --filename-prefix source \
        --jobs "$jobs" --size-bytes "$size_bytes" --block-bytes "$block_bytes"
  fi
}

drop_raw_source_server_cache() {
  # g6 reads the source through g5's NFS server, so only evicting the g6
  # client cache still measures the storage network. Evict it on g5 as well
  # to force the read down to the NVMe.
  local case_id=$1 jobs=$2
  run_step \
    --nodes=1 --ntasks=1 --nodelist=g5 --cpus-per-task=1 --gres=none \
    python3 "$CODE_HOME/io-test/posix_bandwidth_bench.py" drop \
      --directory "$DATA_ROOT/raw/$case_id/read-source" \
      --filename-prefix source --jobs "$jobs"
}

run_raw_phase() {
  local case_id=$1 phase=$2 hosts=$3 nodes=$4 rw=$5 mode=$6 jobs=$7 block_size=$8 repetition=$9
  local size_mib=$((RAW_TOTAL_GIB * 1024 / jobs))
  local size_bytes=$((size_mib * 1024 * 1024))
  local block_bytes
  block_bytes=$(numfmt --from=iec "$block_size")
  if [[ $rw == read && $RAW_DROP_SERVER_CACHE == 1 && $mode != buffered-hot ]]; then
    drop_raw_source_server_cache "$case_id" "$jobs"
  fi
  local sync_port=$((31000 + RANDOM % 20000))
  local direct=0 invalidate=1
  if [[ $mode == direct ]]; then
    direct=1
  elif [[ $mode == buffered-hot ]]; then
    invalidate=0
  fi
  local label="raw.$case_id.$phase.$mode.r$repetition"
  capture_state "$label.before"
  if ! run_step \
    --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 --nodelist="$hosts" \
    --cpus-per-task="$CPUS_PER_NODE" --gres=none \
    --export=ALL,CODE_HOME="$CODE_HOME",RAW_BACKEND="$RAW_BACKEND",RW="$rw",SOURCE_DIR="$DATA_ROOT/raw/$case_id/read-source",WRITE_ROOT="$DATA_ROOT/raw/$case_id/write-$phase-$mode-r$repetition",RESULT_DIR="$RESULT_ROOT/raw/$case_id",PHASE="$phase",MODE="$mode",BLOCK_SIZE="$block_size",BLOCK_BYTES="$block_bytes",DIRECT="$direct",INVALIDATE="$invalidate",JOBS="$jobs",SIZE_MIB="$size_mib",SIZE_BYTES="$size_bytes",REPETITION="$repetition",NODES="$nodes",SYNC_PORT="$sync_port" \
    bash -lc '
      set -e
      host=$(hostname -s)
      if [[ $RW == read ]]; then
        directory="$SOURCE_DIR"
        format="source.\$jobnum"
      else
        directory="$WRITE_ROOT/$host"
        format="write.\$jobnum"
        mkdir -p "$directory"
      fi
      output="$RESULT_DIR/$PHASE.$MODE.r$REPETITION.$host.json"
      if [[ $RAW_BACKEND == fio ]]; then
        python3 "$CODE_HOME/io-test/cluster_barrier.py" \
          --nodes "$NODES" --host g5 --port "$SYNC_PORT"
        fio --name="$PHASE" --directory="$directory" --filename_format="$format" \
          --rw="$RW" --bs="$BLOCK_SIZE" --ioengine=libaio \
          --direct="$DIRECT" --invalidate="$INVALIDATE" --iodepth=16 \
          --numjobs="$JOBS" --size="${SIZE_MIB}M" --group_reporting=1 \
          --end_fsync=1 --output-format=json --output="$output"
      else
        durable=()
        [[ $RW == write ]] && durable+=(--durable)
        python3 "$CODE_HOME/io-test/posix_bandwidth_bench.py" io \
          --directory "$directory" --filename-prefix "${format%.\$jobnum}" \
          --jobs "$JOBS" --size-bytes "$SIZE_BYTES" --block-bytes "$BLOCK_BYTES" \
          --rw "$RW" --mode "$MODE" "${durable[@]}" \
          --nodes "$NODES" --sync-host g5 --sync-port "$SYNC_PORT" \
          --output "$output"
      fi
    '; then
    printf '{"phase":"%s","mode":"%s","repetition":%d,"status":"unsupported-or-failed"}\n' \
      "$phase" "$mode" "$repetition" \
      >"$RESULT_ROOT/raw/$case_id/$phase.$mode.r$repetition.status.json"
    partial_failure=1
  fi
  capture_state "$label.after"
  if [[ $rw == write && ! -e $RESULT_ROOT/raw/$case_id/$phase.$mode.r$repetition.status.json ]]; then
    remove_phase_data "$DATA_ROOT/raw/$case_id/write-$phase-$mode-r$repetition"
  fi
}

run_raw_mixed_phase() {
  local case_id=$1 mode=$2 jobs=$3 block_size=$4 repetition=$5
  local size_mib=$((RAW_TOTAL_GIB * 1024 / jobs))
  local size_bytes=$((size_mib * 1024 * 1024))
  local block_bytes
  block_bytes=$(numfmt --from=iec "$block_size")
  local sync_port=$((31000 + RANDOM % 20000))
  local direct=0 invalidate=1
  [[ $mode == direct ]] && direct=1
  [[ $mode == buffered-hot ]] && invalidate=0
  local label="raw.$case_id.mixed-g5-read-g6-write.$mode.r$repetition"
  capture_state "$label.before"
  if ! run_step \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 --nodelist=g5,g6 \
    --cpus-per-task="$CPUS_PER_NODE" --gres=none \
    --export=ALL,CODE_HOME="$CODE_HOME",RAW_BACKEND="$RAW_BACKEND",SOURCE_DIR="$DATA_ROOT/raw/$case_id/read-source",WRITE_ROOT="$DATA_ROOT/raw/$case_id/write-mixed-$mode-r$repetition",RESULT_DIR="$RESULT_ROOT/raw/$case_id",MODE="$mode",BLOCK_SIZE="$block_size",BLOCK_BYTES="$block_bytes",DIRECT="$direct",INVALIDATE="$invalidate",JOBS="$jobs",SIZE_MIB="$size_mib",SIZE_BYTES="$size_bytes",REPETITION="$repetition",SYNC_PORT="$sync_port" \
    bash -lc '
      set -e
      host=$(hostname -s)
      if [[ $host == g5 ]]; then
        rw=read
        directory="$SOURCE_DIR"
        format="source.\$jobnum"
      else
        rw=write
        directory="$WRITE_ROOT/$host"
        format="write.\$jobnum"
        mkdir -p "$directory"
      fi
      output="$RESULT_DIR/mixed.$MODE.r$REPETITION.$host.json"
      if [[ $RAW_BACKEND == fio ]]; then
        python3 "$CODE_HOME/io-test/cluster_barrier.py" \
          --nodes 2 --host g5 --port "$SYNC_PORT"
        fio --name="mixed-$rw" --directory="$directory" --filename_format="$format" \
          --rw="$rw" --bs="$BLOCK_SIZE" --ioengine=libaio \
          --direct="$DIRECT" --invalidate="$INVALIDATE" --iodepth=16 \
          --numjobs="$JOBS" --size="${SIZE_MIB}M" --group_reporting=1 \
          --end_fsync=1 --output-format=json --output="$output"
      else
        durable=()
        [[ $rw == write ]] && durable+=(--durable)
        python3 "$CODE_HOME/io-test/posix_bandwidth_bench.py" io \
          --directory "$directory" --filename-prefix "${format%.\$jobnum}" \
          --jobs "$JOBS" --size-bytes "$SIZE_BYTES" --block-bytes "$BLOCK_BYTES" \
          --rw "$rw" --mode "$MODE" "${durable[@]}" \
          --nodes 2 --sync-host g5 --sync-port "$SYNC_PORT" \
          --output "$output"
      fi
    '; then
    printf '{"phase":"mixed","mode":"%s","repetition":%d,"status":"unsupported-or-failed"}\n' \
      "$mode" "$repetition" \
      >"$RESULT_ROOT/raw/$case_id/mixed.$mode.r$repetition.status.json"
    partial_failure=1
  fi
  capture_state "$label.after"
  if [[ ! -e $RESULT_ROOT/raw/$case_id/mixed.$mode.r$repetition.status.json ]]; then
    remove_phase_data "$DATA_ROOT/raw/$case_id/write-mixed-$mode-r$repetition"
  fi
}

phase_enabled() {
  [[ " $RAW_PHASES " == *" $1 "* ]]
}

raw_phase_if_enabled() {
  local phase=$2
  phase_enabled "$phase" && run_raw_phase "$@"
  return 0
}

raw_mixed_if_enabled() {
  phase_enabled mixed && run_raw_mixed_phase "$@"
  return 0
}

run_raw_matrix() {
  echo "[raw] Running $RAW_BACKEND matrix over phases: $RAW_PHASES"
  for jobs in $RAW_JOBS_LIST; do
    (( RAW_TOTAL_GIB * 1024 % jobs == 0 )) || {
      echo "ERROR: RAW_TOTAL_GIB*1024 must divide evenly by jobs=$jobs" >&2
      exit 2
    }
    for block_size in $RAW_BLOCK_SIZES; do
      case_id="jobs-$jobs-bs-$block_size"
      mkdir -p "$RESULT_ROOT/raw/$case_id"
      prepare_raw_read_source "$jobs" "$block_size" "$case_id"
      for mode in $RAW_MODES; do
        for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
          if (( repetition % 2 == 1 )); then
            raw_phase_if_enabled "$case_id" g5-read g5 1 read "$mode" "$jobs" "$block_size" "$repetition"
            raw_phase_if_enabled "$case_id" g6-read g6 1 read "$mode" "$jobs" "$block_size" "$repetition"
            raw_phase_if_enabled "$case_id" dual-read g5,g6 2 read "$mode" "$jobs" "$block_size" "$repetition"
            raw_phase_if_enabled "$case_id" g5-write g5 1 write "$mode" "$jobs" "$block_size" "$repetition"
            raw_phase_if_enabled "$case_id" g6-write g6 1 write "$mode" "$jobs" "$block_size" "$repetition"
            raw_phase_if_enabled "$case_id" dual-write g5,g6 2 write "$mode" "$jobs" "$block_size" "$repetition"
            raw_mixed_if_enabled "$case_id" "$mode" "$jobs" "$block_size" "$repetition"
          else
            raw_mixed_if_enabled "$case_id" "$mode" "$jobs" "$block_size" "$repetition"
            raw_phase_if_enabled "$case_id" dual-write g5,g6 2 write "$mode" "$jobs" "$block_size" "$repetition"
            raw_phase_if_enabled "$case_id" g6-write g6 1 write "$mode" "$jobs" "$block_size" "$repetition"
            raw_phase_if_enabled "$case_id" g5-write g5 1 write "$mode" "$jobs" "$block_size" "$repetition"
            raw_phase_if_enabled "$case_id" dual-read g5,g6 2 read "$mode" "$jobs" "$block_size" "$repetition"
            raw_phase_if_enabled "$case_id" g6-read g6 1 read "$mode" "$jobs" "$block_size" "$repetition"
            raw_phase_if_enabled "$case_id" g5-read g5 1 read "$mode" "$jobs" "$block_size" "$repetition"
          fi
        done
      done
    done
  done
}

run_kv_locality() {
  local writer=$1 peer=$2 mode=$3
  local copy=0 buffer=pageable
  if [[ $mode == staging ]]; then
    copy=1
    buffer=pinned
  fi
  local name="kv-$mode-writer-$writer"
  echo "[kv] $name"
  capture_state "$name.before"
  CODE_HOME="$CODE_HOME" RUNTIME_HOME="$RUNTIME_HOME" IMAGE="$IMAGE" \
    CONTAINER_MOUNTS="$container_mounts" \
    READER_HOSTS="$writer $peer" TP_RANKS="$TP_RANKS" \
    PAGES_PER_RANK="$PAGES_PER_RANK" PAGE_BYTES="$PAGE_BYTES" \
    REPETITIONS="$REPETITIONS" BUFFER_TYPE="$buffer" COPY_TO_STAGING="$copy" \
    COPY_IMPLEMENTATION=torch INCLUDE_PEER_SINGLE=1 CPUS_PER_NODE="$CPUS_PER_NODE" \
    RUN_ROOT="$DATA_ROOT/$name" LOCAL_RESULT_DIR="$RESULT_ROOT/$name" \
    SYNC_PORT="$((29577 + RANDOM % 1000))" \
    "$CODE_HOME/io-test/run.sh" \
    >>"$RESULT_ROOT/$name.log" 2>&1 || {
    printf '{"phase":"%s","status":"unsupported-or-failed","log":"%s"}\n' \
      "$name" "$name.log" >"$RESULT_ROOT/$name.status.json"
    echo "  FAILED; see $RESULT_ROOT/$name.log" >&2
    tail -n 20 "$RESULT_ROOT/$name.log" >&2 || true
    partial_failure=1
  }
  capture_state "$name.after"
}

run_kv_matrix() {
  for mode in $KV_MODES; do
    run_kv_locality g5 g6 "$mode"
    run_kv_locality g6 g5 "$mode"
  done
}

run_cache_matrix() {
  local host=$1 peer=$2
  echo "[matrix] cold/hot x buffered/O_DIRECT on $host"
  capture_state "cache-matrix-$host.before"
  if ! env \
    CODE_HOME="$CODE_HOME" \
    RUNTIME_HOME="$RUNTIME_HOME" \
    IMAGE="$IMAGE" \
    CONTAINER_MOUNTS="$container_mounts" \
    READER_HOST="$host" \
    PEER_HOST="$peer" \
    DROP_PEER_CACHE="$MATRIX_DROP_PEER_CACHE" \
    TP_RANKS="$TP_RANKS" \
    PAGES_PER_RANK="$PAGES_PER_RANK" \
    PAGE_BYTES="$PAGE_BYTES" \
    REPETITIONS="$REPETITIONS" \
    CPUS_PER_NODE="$CPUS_PER_NODE" \
    BUFFER_TYPE="$MATRIX_BUFFER_TYPE" \
    COPY_TO_STAGING=1 \
    COPY_IMPLEMENTATION="$MATRIX_COPY_IMPLEMENTATION" \
    RUN_ROOT="$DATA_ROOT/cache-matrix-$host" \
    LOCAL_RESULT_DIR="$RESULT_ROOT/cache-matrix-$host" \
    "$CODE_HOME/io-test/run_hot_cache.sh" \
    >>"$RESULT_ROOT/cache-matrix-$host.log" 2>&1; then
    printf '{"host":"%s","status":"unsupported-or-failed","log":"%s"}\n' \
      "$host" "cache-matrix-$host.log" \
      >"$RESULT_ROOT/cache-matrix-$host.status.json"
    echo "  FAILED; see $RESULT_ROOT/cache-matrix-$host.log" >&2
    tail -n 20 "$RESULT_ROOT/cache-matrix-$host.log" >&2 || true
    partial_failure=1
  fi
  capture_state "cache-matrix-$host.after"
}

summarize_now() {
  python3 "$CODE_HOME/io-test/summarize_pza100.py" \
    --run-root "$RUN_ROOT" \
    --output "$RESULT_ROOT/summary.json" \
    --report "$RESULT_ROOT/REPORT.md" >/dev/null
}

run_kv_write_phase() {
  local phase=$1 hosts=$2 nodes=$3 sync_mode=$4 repetition=$5
  local output_root=$DATA_ROOT/kv-write/$phase-$sync_mode-r$repetition
  local sync_port=$((33000 + RANDOM % 20000))
  mkdir -p "$RESULT_ROOT/kv-write"
  local label="kv-write.$phase.$sync_mode.r$repetition"
  capture_state "$label.before"
  run_step \
    --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 --nodelist="$hosts" \
    --cpus-per-task="$CPUS_PER_NODE" --gres=gpu:1 \
    --export=ALL,OUTPUT_ROOT="$output_root",TP_RANKS="$TP_RANKS",PAGES_PER_RANK="$PAGES_PER_RANK",PAGE_BYTES="$PAGE_BYTES",SYNC_MODE="$sync_mode",PHASE="$phase",REPETITION="$repetition",NODES="$nodes",SYNC_PORT="$sync_port" \
    "${container_args[@]}" \
    bash -lc '
      host=$(hostname -s)
      python3 /workspace/dual-path/io-test/kv_write_load.py \
        --output-dir "$OUTPUT_ROOT/$host" \
        --tp-ranks "$TP_RANKS" \
        --pages-per-rank "$PAGES_PER_RANK" \
        --page-bytes "$PAGE_BYTES" \
        --buffer-type pinned \
        --sync-mode "$SYNC_MODE" \
        --phase "$PHASE" \
        --repetition "$REPETITION" \
        --nodes "$NODES" \
        --sync-host g5 \
        --sync-port "$SYNC_PORT"
    ' \
    | tee "$RESULT_ROOT/kv-write/$phase.$sync_mode.r$repetition.log"
  capture_state "$label.after"
  remove_phase_data "$output_root"
}

run_write_matrix() {
  echo "[write] Running Decode-style KV write matrix"
  for sync_mode in $WRITE_SYNC_MODES; do
    for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
      run_kv_write_phase g5-write g5 1 "$sync_mode" "$repetition"
      run_kv_write_phase g6-write g6 1 "$sync_mode" "$repetition"
      run_kv_write_phase dual-write g5,g6 2 "$sync_mode" "$repetition"
    done
  done
}

# The cache/O_DIRECT matrix is the primary result, so it runs before the long
# cross-node matrices, and the report is refreshed after every phase group. A
# run that has to be cut short still leaves the four quadrants summarized.
if [[ $RUN_MATRIX == 1 ]]; then
  run_cache_matrix g5 g6
  run_cache_matrix g6 g5
  summarize_now
fi
if [[ $RUN_WRITE == 1 ]]; then
  run_write_matrix
  summarize_now
fi
if [[ $RUN_RAW == 1 ]]; then
  run_raw_matrix
  summarize_now
fi
if [[ $RUN_KV == 1 ]]; then
  run_kv_matrix
  summarize_now
fi

capture_state completed
python3 "$CODE_HOME/io-test/summarize_pza100.py" \
  --run-root "$RUN_ROOT" \
  --output "$RESULT_ROOT/summary.json" \
  --report "$RESULT_ROOT/REPORT.md"

if (( partial_failure )); then
  echo "SUMMARY=$RESULT_ROOT/summary.json"
  echo "REPORT=$RESULT_ROOT/REPORT.md"
  echo "ERROR: one or more benchmark phases failed; data has been preserved" >&2
  exit 1
fi
run_failed=0
echo "PZA100_IO_BENCH_COMPLETE run_root=$RUN_ROOT"
echo "SUMMARY=$RESULT_ROOT/summary.json"
echo "REPORT=$RESULT_ROOT/REPORT.md"
