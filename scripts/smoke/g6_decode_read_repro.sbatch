#!/usr/bin/env bash
# Reproduce the Decode-path L3 read pattern on g6 with io-test/kv_read_bench.py:
# TP ranks read one 4 MiB page file each (open/O_DIRECT read/close) from the g5
# XFS export over NFS, then copy into a pinned staging buffer. Each repetition
# runs once with g5's NFS server cache hot and once after evicting it on g5.
#SBATCH --partition=run
#SBATCH --nodes=2
#SBATCH --nodelist=g5,g6
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=g6-decode-read-repro
#SBATCH --output=/data/home/yongxiao/dual-path/results/g6-decode-read-repro-%j.out

set -euo pipefail

CODE_HOME=${CODE_HOME:-/data/home/yongxiao/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/data/home/yongxiao/mydata}
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-l3bps-20260929-nich2d-async-kvtrans.sqsh}
TP_RANKS=${TP_RANKS:-4}
PAGES_PER_RANK=${PAGES_PER_RANK:-324}
PAGE_BYTES=${PAGE_BYTES:-4194304}
REPETITIONS=${REPETITIONS:-3}
IO_MODES=${IO_MODES:-direct}
SERVER_CACHE_STATES=${SERVER_CACHE_STATES:-"hot cold"}
# Concurrent g6 NFS writer ranks during each read (0 = no writer). The writer
# mimics HiCacheFile.set(): one buffered 4 MiB file per page, no fsync.
WRITE_LOADS=${WRITE_LOADS:-"0 1 4"}
WRITER_PAGES_PER_RANK=${WRITER_PAGES_PER_RANK:-256}
CPUS_PER_NODE=${CPUS_PER_NODE:-32}

DATA_DIR=${DATA_DIR:-$RUNTIME_HOME/io-test/$SLURM_JOB_ID-g6-decode-repro/kv-pages}
RESULT_ROOT=${RESULT_ROOT:-$CODE_HOME/io-test/results/g6-decode-read-repro-$SLURM_JOB_ID}
mkdir -p "$RESULT_ROOT"
layout=(--data-dir "$DATA_DIR" --tp-ranks "$TP_RANKS" \
  --pages-per-rank "$PAGES_PER_RANK" --page-bytes "$PAGE_BYTES")

bench_g6() {
  srun --jobid="$SLURM_JOB_ID" --overlap --unbuffered --label --mpi=none \
    --nodes=1 --ntasks=1 --nodelist=g6 --cpus-per-task="$CPUS_PER_NODE" --gres=gpu:1 \
    --container-image="$IMAGE" \
    --container-mounts="/data:/data,$CODE_HOME:/workspace/dual-path:ro" \
    --container-workdir=/workspace/dual-path --no-container-mount-home \
    python3 /workspace/dual-path/io-test/kv_read_bench.py "$@"
}

drop_server_cache() {
  srun --jobid="$SLURM_JOB_ID" --overlap --unbuffered --label --mpi=none \
    --nodes=1 --ntasks=1 --nodelist=g5 --cpus-per-task=1 --gres=none \
    python3 "$CODE_HOME/io-test/kv_read_bench.py" drop-cache "${layout[@]}"
}

WRITER_ROOT=$(dirname "$DATA_DIR")/writer
writer_pid=

start_writer() {
  local ranks=$1 dir=$2 log=$3
  # Rewrite the same page set in a loop so the writer outlives the reader's
  # container startup and read, while disk usage stays bounded.
  srun --jobid="$SLURM_JOB_ID" --overlap --unbuffered --mpi=none \
    --nodes=1 --ntasks=1 --nodelist=g6 --cpus-per-task=8 --gres=none \
    --export=ALL,CODE_HOME="$CODE_HOME",DIR="$dir",RANKS="$ranks",PAGES="$WRITER_PAGES_PER_RANK",PAGE_BYTES="$PAGE_BYTES" \
    bash -c 'while true; do
      python3 "$CODE_HOME/io-test/kv_write_load.py" --output-dir "$DIR" \
        --tp-ranks "$RANKS" --pages-per-rank "$PAGES" --page-bytes "$PAGE_BYTES" \
        --buffer-type pageable --sync-mode none --phase background-writer || exit 1
    done' >"$log" 2>&1 &
  writer_pid=$!
  local waited
  for ((waited = 0; waited < 600; waited++)); do
    [[ -n $(find "$dir" -type f -print -quit 2>/dev/null) ]] && return 0
    kill -0 "$writer_pid" 2>/dev/null || { echo "writer exited early; see $log" >&2; return 1; }
    sleep 0.1
  done
  echo "writer did not start; see $log" >&2
  return 1
}

stop_writer() {
  local dir=$1 log=$2 report=$3 alive=0
  kill -0 "$writer_pid" 2>/dev/null && alive=1
  kill -TERM "$writer_pid" 2>/dev/null || true
  wait "$writer_pid" 2>/dev/null || true
  writer_pid=
  python3 - "$log" "$alive" <<'PY' | tee "$report"
import json
import sys

marker = "KVWRITE_RESULT "
text = open(sys.argv[1], errors="replace").read().replace("\n", "")
decoder = json.JSONDecoder()
bytes_written = seconds = 0.0
rounds = 0
start = text.find(marker)
while start >= 0:
    result, _ = decoder.raw_decode(text, start + len(marker))
    bytes_written += result["bytes"]
    seconds += result["seconds"]
    rounds += 1
    start = text.find(marker, start + len(marker))
rate = bytes_written / seconds / 1e9 if seconds else 0.0
print(f"writer_rounds={rounds} writer_bytes={int(bytes_written)} "
      f"writer_seconds={seconds:.2f} writer_gb_s={rate:.3f} "
      f"writer_alive_through_read={sys.argv[2]}")
PY
  rm -rf -- "$dir"
}

cleanup() {
  [[ -n $writer_pid ]] && { kill -TERM "$writer_pid" 2>/dev/null || true; wait "$writer_pid" 2>/dev/null || true; }
  rm -rf -- "$WRITER_ROOT"
  bench_g6 cleanup --data-dir "$DATA_DIR" || true
}
trap cleanup EXIT

echo "G6_DECODE_READ_REPRO allocation=$SLURM_JOB_ID"
echo "  tp_ranks=$TP_RANKS pages_per_rank=$PAGES_PER_RANK page_bytes=$PAGE_BYTES"
echo "  bytes_per_read=$((TP_RANKS * PAGES_PER_RANK * PAGE_BYTES)) repetitions=$REPETITIONS"
echo "  io_modes=$IO_MODES server_cache_states=$SERVER_CACHE_STATES write_loads=$WRITE_LOADS"
srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 --nodelist=g6 \
  --cpus-per-task=1 --gres=none bash -c 'findmnt -T /data -o TARGET,SOURCE,FSTYPE,OPTIONS' \
  | tee "$RESULT_ROOT/g6-mount.txt"

# Written from g6 over NFS, like Decode offload writes its L3 pages.
bench_g6 prepare "${layout[@]}"

for ((rep = 1; rep <= REPETITIONS; rep++)); do
  for mode in $IO_MODES; do
    for state in $SERVER_CACHE_STATES; do
      for writers in $WRITE_LOADS; do
        [[ $state == cold ]] && drop_server_cache
        phase="$mode-server-$state-w$writers"
        echo "Running phase=$phase repetition=$rep"
        writer_dir=$WRITER_ROOT/$phase-r$rep
        if (( writers > 0 )); then
          start_writer "$writers" "$writer_dir" "$RESULT_ROOT/$phase.r$rep.writer.log"
        fi
        bench_g6 read "${layout[@]}" --passes 1 --buffer-type pinned --drop-cache \
          --copy-to-staging --copy-implementation torch --torch-threads 1 \
          --layers 64 --page-tokens 64 --head-dim 128 \
          --phase "$phase" --repetition "$rep" --nodes 1 --io-mode "$mode" \
          --direct-alignment 4096 --output "$RESULT_ROOT/$phase.r$rep.json" >/dev/null
        if (( writers > 0 )); then
          stop_writer "$writer_dir" "$RESULT_ROOT/$phase.r$rep.writer.log" \
            "$RESULT_ROOT/$phase.r$rep.writer.txt"
        fi
      done
    done
  done
done

python3 - "$RESULT_ROOT" <<'PY' | tee "$RESULT_ROOT/REPORT.md"
import json
import statistics
import sys
from pathlib import Path

records = {}
for path in sorted(Path(sys.argv[1]).glob("*.r*.json")):
    phase = path.name.rsplit(".", 2)[0]
    records.setdefault(phase, []).append(json.loads(path.read_text()))

def writer_rate(phase):
    rates, alive = [], True
    for path in sorted(Path(sys.argv[1]).glob(f"{phase}.r*.writer.txt")):
        fields = dict(item.split("=") for item in path.read_text().split())
        rates.append(float(fields["writer_gb_s"]))
        alive &= fields["writer_alive_through_read"] == "1"
    if not rates:
        return "-"
    return f"{statistics.median(rates):.3f}" + ("" if alive else " (ended early)")

print("| phase | end-to-end GB/s | file IO GB/s | per-rank GB/s | read p50 ms/page | open p50 ms | writer GB/s | runs GB/s |")
print("|---|---:|---:|---:|---:|---:|---:|---|")
for phase, runs in sorted(records.items()):
    e2e = [r["bytes"] / r["seconds"] / 1e9 for r in runs]
    fio = [r["file_io_gib_s"] * 2**30 / 1e9 for r in runs]
    rank = [r["rank_gib_s_median"] * 2**30 / 1e9 for r in runs]
    read = [r["read_ns"]["p50_ns"] / 1e6 for r in runs]
    opened = [r["open_ns"]["p50_ns"] / 1e6 for r in runs]
    print(f"| {phase} | {statistics.median(e2e):.3f} | {statistics.median(fio):.3f} | "
          f"{statistics.median(rank):.3f} | {statistics.median(read):.2f} | "
          f"{statistics.median(opened):.3f} | {writer_rate(phase)} | "
          + ", ".join(f"{v:.3f}" for v in e2e) + " |")
PY
echo "G6_DECODE_READ_REPRO_COMPLETE result_root=$RESULT_ROOT"
