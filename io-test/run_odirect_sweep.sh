#!/usr/bin/env bash
# Find the highest O_DIRECT read bandwidth one node can reach.
#
# The four-quadrant matrix deliberately reproduces the TP4 Decode access
# pattern: 4 processes, queue depth 1. That answers "what does Decode get", not
# "what can this storage path do". This sweep answers the second question by
# varying process count, block size and total I/O.
#
# One source set is prepared once with MAX_JOBS files of PER_JOB_MIB each, and
# every point reads the first N of them, so total I/O grows with concurrency
# while per-process work stays constant.
#
# Allocating both nodes lets the source set be written locally on the disk
# owner; with a single-node allocation the script prepares it from HOST itself.
#
#   salloc -p build -N2 -w g5,g6 --ntasks=2 --ntasks-per-node=1 \
#     --cpus-per-task=64 --mem=0 --time=02:00:00 \
#     bash -c 'cd /data/home/yongxiao/dual-path && HOST=g6 io-test/run_odirect_sweep.sh'
set -euo pipefail

CODE_HOME=${CODE_HOME:-/data/home/yongxiao/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/data/home/yongxiao/mydata}
HOST=${HOST:-g6}
# Writing the source set from the node that owns the disk is far faster than
# pushing 64 GiB back over NFS. Leave empty to prepare from HOST itself.
PREPARE_HOST=${PREPARE_HOST:-g5}
JOBS_LIST=${JOBS_LIST:-"1 2 4 8 16 32 64 128"}
BLOCK_SIZES=${BLOCK_SIZES:-"256K 1M 4M"}
MODES=${MODES:-"direct buffered-cold"}
PER_JOB_MIB=${PER_JOB_MIB:-512}
REPETITIONS=${REPETITIONS:-2}
CPUS_PER_NODE=${CPUS_PER_NODE:-64}
GRES=${GRES:-none}
KEEP_DATA=${KEEP_DATA:-0}

: "${SLURM_JOB_ID:?Run this script inside an allocation that contains $HOST}"
mapfile -t allocated_hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
if [[ " ${allocated_hosts[*]} " != *" $HOST "* ]]; then
  echo "ERROR: HOST=$HOST is not in this allocation: ${allocated_hosts[*]}" >&2
  exit 2
fi
if [[ -n $PREPARE_HOST && " ${allocated_hosts[*]} " != *" $PREPARE_HOST "* ]]; then
  echo "NOTE: PREPARE_HOST=$PREPARE_HOST is not in this allocation" \
    "(${allocated_hosts[*]}); preparing the source set from $HOST instead."
  echo "      Allocate both nodes to write it locally on the disk owner."
  PREPARE_HOST=
fi
run_root=${RUN_ROOT:-$RUNTIME_HOME/io-test/$SLURM_JOB_ID-odirect-sweep-$HOST}
source_dir=$run_root/source
result_dir=${LOCAL_RESULT_DIR:-$CODE_HOME/io-test/results/$SLURM_JOB_ID-odirect-sweep-$HOST}
records_dir=$result_dir/records
mkdir -p "$records_dir"

max_jobs=0
for jobs in $JOBS_LIST; do
  [[ $jobs =~ ^[1-9][0-9]*$ ]] || { echo "JOBS_LIST must be positive integers" >&2; exit 2; }
  (( jobs > max_jobs )) && max_jobs=$jobs
done
per_job_bytes=$((PER_JOB_MIB * 1024 * 1024))
for block_size in $BLOCK_SIZES; do
  block_bytes=$(numfmt --from=iec "$block_size")
  (( per_job_bytes % block_bytes == 0 )) || {
    echo "ERROR: PER_JOB_MIB=$PER_JOB_MIB is not divisible by block size $block_size" >&2
    exit 2
  }
done

# posix_bandwidth_bench.py only needs the standard library, so these steps skip
# the container and its per-step squashfs mount.
run_bench() {
  local host=$1
  shift
  srun --jobid="$SLURM_JOB_ID" --overlap --unbuffered \
    --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="$host" \
    --cpus-per-task="$CPUS_PER_NODE" --gres="$GRES" --mpi=none \
    python3 "$CODE_HOME/io-test/posix_bandwidth_bench.py" "$@"
}

echo "O_DIRECT read sweep on $HOST"
echo "  allocation=$SLURM_JOB_ID source=$source_dir results=$result_dir"
echo "  jobs=$JOBS_LIST block_sizes=$BLOCK_SIZES modes=$MODES"
echo "  per_job=${PER_JOB_MIB} MiB max_jobs=$max_jobs"
echo "  source_set=$((max_jobs * PER_JOB_MIB / 1024)) GiB repetitions=$REPETITIONS"

if [[ ! -e $source_dir/src.$((max_jobs - 1)) ]]; then
  echo "[prepare] writing $max_jobs files of ${PER_JOB_MIB} MiB from ${PREPARE_HOST:-$HOST}"
  run_bench "${PREPARE_HOST:-$HOST}" prepare \
    --directory "$source_dir" --filename-prefix src \
    --jobs "$max_jobs" --size-bytes "$per_job_bytes" --block-bytes 4194304
else
  echo "[prepare] reusing existing source set"
fi

for mode in $MODES; do
  for block_size in $BLOCK_SIZES; do
    block_bytes=$(numfmt --from=iec "$block_size")
    for jobs in $JOBS_LIST; do
      for ((repetition = 1; repetition <= REPETITIONS; repetition++)); do
        name="$mode.bs-$block_size.jobs-$jobs.r$repetition"
        echo "[run] $name"
        if ! run_bench "$HOST" io \
          --directory "$source_dir" --filename-prefix src \
          --jobs "$jobs" --size-bytes "$per_job_bytes" --block-bytes "$block_bytes" \
          --rw read --mode "$mode" \
          --output "$records_dir/$name.json" >/dev/null; then
          echo "  FAILED: $name" >&2
        fi
      done
    done
  done
done

python3 - "$records_dir" "$result_dir/REPORT.md" <<'PY'
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

records_dir, report_path = Path(sys.argv[1]), Path(sys.argv[2])
grouped = defaultdict(list)
for path in sorted(records_dir.glob("*.json")):
    payload = json.loads(path.read_text())
    key = (payload["mode"], payload["block_bytes"], payload["jobs"])
    grouped[key].append(payload)

lines = [
    "| mode | block | jobs | total GiB | GiB/s | Gb/s on the wire | per-block p50 | per-block p95 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
header = (
    f"{'mode':15s}{'block':>8s}{'jobs':>6s}{'totalGiB':>10s}"
    f"{'GiB/s':>9s}{'Gb/s':>8s}{'p50 ms':>9s}{'p95 ms':>9s}"
)
print()
print(header)
print("-" * len(header))
best = {}
for (mode, block_bytes, jobs), items in sorted(grouped.items()):
    rate = statistics.median(float(item["gib_s"]) for item in items)
    total = float(items[0]["bytes"]) / 2**30
    gbit = rate * 2**30 * 8 / 1e9
    p50 = statistics.median(float(item["latency_ns"]["p50"]) for item in items) / 1e6
    p95 = max(float(item["latency_ns"]["p95"]) for item in items) / 1e6
    block = f"{block_bytes // 1024}K" if block_bytes < 2**20 else f"{block_bytes // 2**20}M"
    print(
        f"{mode:15s}{block:>8s}{jobs:>6d}{total:10.1f}"
        f"{rate:9.3f}{gbit:8.1f}{p50:9.3f}{p95:9.3f}"
    )
    lines.append(
        f"| `{mode}` | {block} | {jobs} | {total:.1f} | {rate:.3f} | "
        f"{gbit:.1f} | {p50:.3f} ms | {p95:.3f} ms |"
    )
    if rate > best.get(mode, (0,))[0]:
        best[mode] = (rate, block, jobs, gbit)

# Captured from sysfs on g5/g6: IB 4X XDR port, HCA PCIe Gen5 x16, and the
# Solidigm namespace behind g5's /data on PCIe Gen4 x4. Only the first two
# apply to a g6 run; only the last one applies to a g5 run.
CEILINGS_GIB_S = {
    "IB 4X XDR port (800 Gb/s)": 93.13,
    "HCA PCIe Gen5 x16": 58.69,
    "g5 NVMe PCIe Gen4 x4": 7.34,
}

print()
summary = []
for mode, (rate, block, jobs, gbit) in sorted(best.items()):
    text = (
        f"best {mode}: {rate:.3f} GiB/s = {gbit:.1f} Gb/s "
        f"at {jobs} jobs x {block} blocks"
    )
    print(f"  {text}")
    summary.append(f"- {text}")
    for name, ceiling in CEILINGS_GIB_S.items():
        share = f"    {rate / ceiling:7.1%} of {name}"
        print(share)
        summary.append(f"  -{share.strip()}")

report_path.write_text(
    "# O_DIRECT read sweep\n\n" + "\n".join(summary + [""] + lines) + "\n"
)
print(f"\nREPORT={report_path}")
PY

if [[ $KEEP_DATA == 0 ]]; then
  echo "Removing source set: $source_dir"
  rm -rf -- "$source_dir"
fi
