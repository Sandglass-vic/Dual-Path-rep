#!/usr/bin/env bash
# Hot-cache L3 reads on pza100, engine read pattern (TP ranks x 4 MiB page
# files, pinned + torch staging): g5 local buffered reads from its page cache,
# g6 O_DIRECT reads over NFS/RDMA that nfsd serves from the same g5 cache.
# Phases: g5 alone, g6 alone, both at once on disjoint datasets.
#SBATCH --partition=run
#SBATCH --nodes=2
#SBATCH --nodelist=g5,g6
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --job-name=g5-g6-hot-read
#SBATCH --output=/data/home/yongxiao/dual-path/results/g5-g6-hot-read-%j.out

set -euo pipefail

CODE_HOME=${CODE_HOME:-/data/home/yongxiao/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/data/home/yongxiao/mydata}
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-l3bps-20260929-nich2d-async-kvtrans.sqsh}
RDMA_MOUNT=${RDMA_MOUNT:-/mnt/nfsrdma}
EXPORT_ROOT=${EXPORT_ROOT:-/data}
TP_RANKS=${TP_RANKS:-4}
PAGES_PER_RANK=${PAGES_PER_RANK:-324}
PAGE_BYTES=${PAGE_BYTES:-4194304}
REPETITIONS=${REPETITIONS:-3}
CPUS=${CPUS:-32}

DATA=$RUNTIME_HOME/io-test/$SLURM_JOB_ID-hot-read
RDMA_DATA=$RDMA_MOUNT/${DATA#"$EXPORT_ROOT"/}
RESULT_ROOT=${RESULT_ROOT:-$CODE_HOME/io-test/results/g5-g6-hot-read-$SLURM_JOB_ID}
mkdir -p "$DATA" "$RESULT_ROOT"
layout=(--tp-ranks "$TP_RANKS" --pages-per-rank "$PAGES_PER_RANK" --page-bytes "$PAGE_BYTES")
read_args=(--passes 1 --buffer-type pinned --copy-to-staging --copy-implementation torch
  --torch-threads 1 --layers 64 --page-tokens 64 --head-dim 128 --direct-alignment 4096)

bench() {
  local host=$1 mounts=$2
  shift 2
  srun --jobid="$SLURM_JOB_ID" --overlap --unbuffered --mpi=none \
    --nodes=1 --ntasks=1 --nodelist="$host" --cpus-per-task="$CPUS" --gres=gpu:1 \
    --container-image="$IMAGE" --container-mounts="$mounts" \
    --container-workdir=/workspace/dual-path --no-container-mount-home "$@"
}
g5_mounts="/data:/data,$CODE_HOME:/workspace/dual-path:ro"
g6_mounts="/data:/data,$RDMA_MOUNT:$RDMA_MOUNT,$CODE_HOME:/workspace/dual-path:ro"
KV=/workspace/dual-path/io-test/kv_read_bench.py

# g5: buffered, no eviction. g6: O_DIRECT through the RDMA view.
# $1=phase $2=rep $3=nodes $4=sync_port $5=procid for g6
read_g5() {
  bench g5 "$g5_mounts" python3 "$KV" read --data-dir "$DATA/g5" "${layout[@]}" "${read_args[@]}" \
    --io-mode buffered --phase "$1" --repetition "$2" --nodes "$3" \
    --sync-host g5 --sync-port "$4" --output "$RESULT_ROOT/$1.r$2.g5.json" >/dev/null
}
read_g6() {
  bench g6 "$g6_mounts" env SLURM_PROCID="$5" python3 "$KV" read --data-dir "$RDMA_DATA/g6" \
    "${layout[@]}" "${read_args[@]}" \
    --io-mode direct --phase "$1" --repetition "$2" --nodes "$3" \
    --sync-host g5 --sync-port "$4" --output "$RESULT_ROOT/$1.r$2.g6.json" >/dev/null
}

trap 'rm -rf -- "$DATA"' EXIT

echo "G5_G6_HOT_READ allocation=$SLURM_JOB_ID bytes_per_read=$((TP_RANKS * PAGES_PER_RANK * PAGE_BYTES))"
srun --jobid="$SLURM_JOB_ID" --overlap --nodes=1 --ntasks=1 --nodelist=g6 --cpus-per-task=1 \
  --gres=none "$CODE_HOME/scripts/pza100-nfsrdma.sh" status | tee "$RESULT_ROOT/g6-rdma-status.txt"

# Both datasets live on g5's XFS; warm both into g5's page cache.
for side in g5 g6; do
  bench g5 "$g5_mounts" python3 "$KV" prepare --data-dir "$DATA/$side" "${layout[@]}"
  bench g5 "$g5_mounts" python3 "$KV" read --data-dir "$DATA/$side" "${layout[@]}" "${read_args[@]}" \
    --io-mode buffered --phase warmup --repetition 0 --nodes 1 >/dev/null
done

for ((rep = 1; rep <= REPETITIONS; rep++)); do
  echo "Running repetition=$rep"
  read_g5 g5-hot-alone "$rep" 1 "$((31000 + RANDOM % 20000))"
  read_g6 g6-rdma-alone "$rep" 1 "$((31000 + RANDOM % 20000))" 0
  port=$((31000 + RANDOM % 20000))
  read_g5 dual "$rep" 2 "$port" &
  g5_pid=$!
  read_g6 dual "$rep" 2 "$port" 1
  wait "$g5_pid"
done

python3 - "$RESULT_ROOT" <<'PY' | tee "$RESULT_ROOT/REPORT.md"
import json
import statistics
import sys
from pathlib import Path

records = {}
for path in sorted(Path(sys.argv[1]).glob("*.r*.*.json")):
    phase, rep, host = path.name[:-5].rsplit(".", 2)
    records.setdefault((phase, host), []).append(json.loads(path.read_text()))

def gbs(r):
    return r["bytes"] / r["seconds"] / 1e9

print("| phase | host | end-to-end GB/s | file IO GB/s | read p50 ms/page | runs GB/s |")
print("|---|---|---:|---:|---:|---|")
for (phase, host), runs in sorted(records.items()):
    e2e = [gbs(r) for r in runs]
    fio = statistics.median(r["file_io_gib_s"] * 2**30 / 1e9 for r in runs)
    p50 = statistics.median(r["read_ns"]["p50_ns"] / 1e6 for r in runs)
    print(f"| {phase} | {host} | {statistics.median(e2e):.3f} | {fio:.3f} | {p50:.2f} | "
          + ", ".join(f"{v:.3f}" for v in e2e) + " |")

g5, g6 = records.get(("dual", "g5"), []), records.get(("dual", "g6"), [])
if g5 and g6:
    print("\n| dual rep | g5 GB/s | g6 GB/s | sum GB/s | overlap s | g5 s | g6 s |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for i, (a, b) in enumerate(zip(g5, g6), 1):
        overlap = (min(a["wall_end_ns_leader"], b["wall_end_ns_leader"])
                   - max(a["wall_start_ns_leader"], b["wall_start_ns_leader"])) / 1e9
        print(f"| {i} | {gbs(a):.3f} | {gbs(b):.3f} | {gbs(a) + gbs(b):.3f} | "
              f"{overlap:.2f} | {a['seconds']:.2f} | {b['seconds']:.2f} |")
PY
echo "G5_G6_HOT_READ_COMPLETE result_root=$RESULT_ROOT"
