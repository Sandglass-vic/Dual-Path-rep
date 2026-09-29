#!/usr/bin/env bash
# A/B g6 reads of the same g5 files over the TCP /data mount and an NFS/RDMA
# mount of the same export. Runs on the hosts (no container) so g6 steps see
# the RDMA mount. g5's page cache is evicted before every read.
#SBATCH --partition=run
#SBATCH --nodes=2
#SBATCH --nodelist=g5,g6
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --job-name=g6-nfs-rdma-ab
#SBATCH --output=/data/home/yongxiao/dual-path/results/g6-nfs-rdma-ab-%j.out

set -euo pipefail

CODE_HOME=${CODE_HOME:-/data/home/yongxiao/dual-path}
RDMA_MOUNT=${RDMA_MOUNT:-/mnt/nfsrdma}
EXPORT_ROOT=${EXPORT_ROOT:-/data}
MOUNTS=${MOUNTS:-"$EXPORT_ROOT $RDMA_MOUNT"}
TP_RANKS=${TP_RANKS:-4}
PAGES_PER_RANK=${PAGES_PER_RANK:-324}
PAGE_BYTES=${PAGE_BYTES:-4194304}
BIG_JOBS_LIST=${BIG_JOBS_LIST:-"4 8"}
BIG_TOTAL_BYTES=${BIG_TOTAL_BYTES:-6442450944}
REPETITIONS=${REPETITIONS:-3}

REL=home/yongxiao/mydata/io-test/$SLURM_JOB_ID-rdma-ab
DATA=$EXPORT_ROOT/$REL
RESULT_ROOT=${RESULT_ROOT:-$CODE_HOME/io-test/results/g6-nfs-rdma-ab-$SLURM_JOB_ID}
mkdir -p "$DATA" "$RESULT_ROOT"
KV=$CODE_HOME/io-test/kv_read_bench.py
BIG=$CODE_HOME/io-test/posix_bandwidth_bench.py
kv_layout=(--tp-ranks "$TP_RANKS" --pages-per-rank "$PAGES_PER_RANK" --page-bytes "$PAGE_BYTES")

on_host() {
  local host=$1
  shift
  srun --jobid="$SLURM_JOB_ID" --overlap --unbuffered --mpi=none \
    --nodes=1 --ntasks=1 --nodelist="$host" --cpus-per-task=32 "$@"
}

drop_all() {
  local n
  on_host g5 python3 "$KV" drop-cache --data-dir "$DATA/kv" "${kv_layout[@]}" >/dev/null
  for n in $BIG_JOBS_LIST; do
    on_host g5 python3 "$BIG" drop --directory "$DATA/big-j$n" --jobs "$n" >/dev/null
  done
}

trap 'rm -rf -- "$DATA"' EXIT

echo "G6_NFS_RDMA_AB allocation=$SLURM_JOB_ID mounts=$MOUNTS repetitions=$REPETITIONS"
on_host g6 bash -c "grep -E ' ($EXPORT_ROOT|$RDMA_MOUNT) ' /proc/mounts" | tee "$RESULT_ROOT/g6-mounts.txt"
grep -q "proto=rdma" "$RESULT_ROOT/g6-mounts.txt" || { echo "no RDMA mount on g6" >&2; exit 2; }

on_host g5 python3 "$KV" prepare --data-dir "$DATA/kv" "${kv_layout[@]}"
for jobs in $BIG_JOBS_LIST; do
  on_host g5 python3 "$BIG" prepare --directory "$DATA/big-j$jobs" --jobs "$jobs" \
    --size-bytes "$((BIG_TOTAL_BYTES / jobs))" --block-bytes 4194304
done

for ((rep = 1; rep <= REPETITIONS; rep++)); do
  for mount in $MOUNTS; do
    name=$([[ $mount == "$EXPORT_ROOT" ]] && echo tcp || echo rdma)
    base=$mount/$REL
    drop_all
    echo "Running kv-$name repetition=$rep"
    on_host g6 python3 "$KV" read --data-dir "$base/kv" "${kv_layout[@]}" \
      --buffer-type pageable --copy-implementation python --drop-cache \
      --phase "kv-$name" --repetition "$rep" --nodes 1 --io-mode direct \
      --direct-alignment 4096 --output "$RESULT_ROOT/kv-$name.r$rep.json" >/dev/null
    for jobs in $BIG_JOBS_LIST; do
      drop_all
      echo "Running big-j$jobs-$name repetition=$rep"
      on_host g6 python3 "$BIG" io --directory "$base/big-j$jobs" --jobs "$jobs" \
        --size-bytes "$((BIG_TOTAL_BYTES / jobs))" --block-bytes 4194304 \
        --rw read --mode direct --output "$RESULT_ROOT/big-j$jobs-$name.r$rep.json" >/dev/null
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
    records.setdefault(path.name.rsplit(".", 2)[0], []).append(json.loads(path.read_text()))

print("| case | transport | median GB/s | per-read p50 ms | runs GB/s |")
print("|---|---|---:|---:|---|")
for phase in sorted(records, key=lambda p: (p.rsplit("-", 1)[0], p.endswith("rdma"))):
    runs = records[phase]
    gbs = [r["bytes"] / r["seconds"] / 1e9 for r in runs]
    if phase.startswith("kv"):
        p50 = statistics.median(r["read_ns"]["p50_ns"] / 1e6 for r in runs)
    else:
        p50 = statistics.median(r["latency_ns"]["p50"] / 1e6 for r in runs)
    case, transport = phase.rsplit("-", 1)
    print(f"| {case} | {transport} | {statistics.median(gbs):.3f} | {p50:.2f} | "
          + ", ".join(f"{v:.3f}" for v in gbs) + " |")
PY
echo "G6_NFS_RDMA_AB_COMPLETE result_root=$RESULT_ROOT"
