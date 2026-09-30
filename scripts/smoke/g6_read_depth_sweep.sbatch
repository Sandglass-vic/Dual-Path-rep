#!/usr/bin/env bash
# Sweep outstanding-read depth for the Decode L3 read pattern on g6 (NFS):
# reader ranks x page-file size at a fixed total volume, plus a large-file
# sequential reference that isolates the per-file cost from per-stream limits.
#SBATCH --partition=run
#SBATCH --nodes=2
#SBATCH --nodelist=g5,g6
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --job-name=g6-read-depth-sweep
#SBATCH --output=/data/home/yongxiao/dual-path/results/g6-read-depth-sweep-%j.out

set -euo pipefail

CODE_HOME=${CODE_HOME:-/data/home/yongxiao/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/data/home/yongxiao/mydata}
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-l3bps-20260929-nich2d-async-kvtrans.sqsh}
RANKS_LIST=${RANKS_LIST:-"4 8 16"}
PAGE_MIB_LIST=${PAGE_MIB_LIST:-"4 8 16"}
TOTAL_MIB=${TOTAL_MIB:-5184}
REPETITIONS=${REPETITIONS:-2}
REF_JOBS_LIST=${REF_JOBS_LIST:-"4 16"}
CPUS_PER_NODE=${CPUS_PER_NODE:-32}

WORK_ROOT=${WORK_ROOT:-$RUNTIME_HOME/io-test/$SLURM_JOB_ID-g6-depth}
RESULT_ROOT=${RESULT_ROOT:-$CODE_HOME/io-test/results/g6-read-depth-sweep-$SLURM_JOB_ID}
mkdir -p "$WORK_ROOT" "$RESULT_ROOT"
BENCH=$CODE_HOME/io-test/posix_bandwidth_bench.py

bench_g6() {
  srun --jobid="$SLURM_JOB_ID" --overlap --unbuffered --label --mpi=none \
    --nodes=1 --ntasks=1 --nodelist=g6 --cpus-per-task="$CPUS_PER_NODE" --gres=gpu:1 \
    --container-image="$IMAGE" \
    --container-mounts="/data:/data,$CODE_HOME:/workspace/dual-path:ro" \
    --container-workdir=/workspace/dual-path --no-container-mount-home \
    python3 /workspace/dual-path/io-test/kv_read_bench.py "$@"
}

on_host() {
  local host=$1
  shift
  srun --jobid="$SLURM_JOB_ID" --overlap --unbuffered --label --mpi=none \
    --nodes=1 --ntasks=1 --nodelist="$host" --cpus-per-task="$CPUS_PER_NODE" --gres=none "$@"
}

trap 'rm -rf -- "$WORK_ROOT"' EXIT

echo "G6_READ_DEPTH_SWEEP allocation=$SLURM_JOB_ID total_mib=$TOTAL_MIB"
echo "  ranks=$RANKS_LIST page_mib=$PAGE_MIB_LIST ref_jobs=$REF_JOBS_LIST repetitions=$REPETITIONS"

for ranks in $RANKS_LIST; do
  for page_mib in $PAGE_MIB_LIST; do
    (( TOTAL_MIB % (ranks * page_mib) == 0 )) || { echo "skip ranks=$ranks page=${page_mib}M"; continue; }
    pages=$((TOTAL_MIB / (ranks * page_mib)))
    data_dir=$WORK_ROOT/kv-r$ranks-p$page_mib
    layout=(--data-dir "$data_dir" --tp-ranks "$ranks" --pages-per-rank "$pages" \
      --page-bytes "$((page_mib * 1024 * 1024))")
    bench_g6 prepare "${layout[@]}"
    for ((rep = 1; rep <= REPETITIONS; rep++)); do
      on_host g5 python3 "$CODE_HOME/io-test/kv_read_bench.py" drop-cache "${layout[@]}"
      phase="pages-r$ranks-p${page_mib}M"
      echo "Running $phase repetition=$rep pages_per_rank=$pages"
      bench_g6 read "${layout[@]}" --passes 1 --buffer-type pinned --drop-cache \
        --copy-to-staging --copy-implementation torch --torch-threads 1 \
        --layers 64 --page-tokens 64 --head-dim 128 \
        --phase "$phase" --repetition "$rep" --nodes 1 --io-mode direct \
        --direct-alignment 4096 --output "$RESULT_ROOT/$phase.r$rep.json" >/dev/null
    done
    bench_g6 cleanup --data-dir "$data_dir" >/dev/null
  done
done

for jobs in $REF_JOBS_LIST; do
  ref_dir=$WORK_ROOT/ref-j$jobs
  size=$((TOTAL_MIB / jobs * 1024 * 1024))
  on_host g5 python3 "$BENCH" prepare --directory "$ref_dir" --jobs "$jobs" \
    --size-bytes "$size" --block-bytes 4194304
  for ((rep = 1; rep <= REPETITIONS; rep++)); do
    on_host g5 python3 "$BENCH" drop --directory "$ref_dir" --jobs "$jobs"
    phase="bigfile-j$jobs-b4M"
    echo "Running $phase repetition=$rep"
    on_host g6 python3 "$BENCH" io --directory "$ref_dir" --jobs "$jobs" \
      --size-bytes "$size" --block-bytes 4194304 --rw read --mode direct \
      --output "$RESULT_ROOT/$phase.r$rep.json" >/dev/null
  done
  rm -rf -- "$ref_dir"
done

python3 - "$RESULT_ROOT" <<'PY' | tee "$RESULT_ROOT/REPORT.md"
import json
import statistics
import sys
from pathlib import Path

records = {}
for path in sorted(Path(sys.argv[1]).glob("*.r*.json")):
    records.setdefault(path.name.rsplit(".", 2)[0], []).append(json.loads(path.read_text()))

def key(phase):
    parts = phase.replace("M", "").split("-")
    return (parts[0], *[int(p[1:]) for p in parts[1:] if p[1:].isdigit()])

print("| phase | readers | read size | end-to-end GB/s | file IO GB/s | read p50 ms | runs GB/s |")
print("|---|---:|---:|---:|---:|---:|---|")
for phase in sorted(records, key=key):
    runs = records[phase]
    e2e = [r["bytes"] / r["seconds"] / 1e9 for r in runs]
    if phase.startswith("pages"):
        readers = runs[0]["tp_ranks"]
        size = runs[0]["page_bytes"] >> 20
        fio = statistics.median(r["file_io_gib_s"] * 2**30 / 1e9 for r in runs)
        p50 = statistics.median(r["read_ns"]["p50_ns"] / 1e6 for r in runs)
    else:
        readers = runs[0]["jobs"]
        size = runs[0]["block_bytes"] >> 20
        fio = statistics.median(r["io_gib_s"] * 2**30 / 1e9 for r in runs)
        p50 = statistics.median(r["latency_ns"]["p50"] / 1e6 for r in runs)
    print(f"| {phase} | {readers} | {size} MiB | {statistics.median(e2e):.3f} | {fio:.3f} | "
          f"{p50:.2f} | " + ", ".join(f"{v:.3f}" for v in e2e) + " |")
PY
echo "G6_READ_DEPTH_SWEEP_COMPLETE result_root=$RESULT_ROOT"
