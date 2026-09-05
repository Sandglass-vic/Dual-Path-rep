#!/usr/bin/env bash
#SBATCH --partition=GAIA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --time=00:10:00
#SBATCH --job-name=lustre-io-bandwidth
#SBATCH --output=/labhome/sezhou/dual-path/lustre-io-bandwidth-%j.out

set -euo pipefail

RUNTIME_HOME=${RUNTIME_HOME:-/mnt/lustre/users/sezhou/dual-path}
RUN_DIR="$RUNTIME_HOME/io-bandwidth/$SLURM_JOB_ID"
NUM_JOBS=${NUM_JOBS:-8}
SIZE_PER_JOB=${SIZE_PER_JOB:-16G}
BLOCK_SIZE=${BLOCK_SIZE:-4M}
IO_DEPTH=${IO_DEPTH:-16}
STRIPE_COUNT=${STRIPE_COUNT:-8}
STRIPE_SIZE=${STRIPE_SIZE:-16M}
DD_COUNT=${DD_COUNT:-4096}

cleanup() {
  rm -rf "$RUN_DIR"
}
trap cleanup EXIT

mkdir -p "$RUN_DIR"
if command -v lfs >/dev/null; then
  lfs setstripe -c "$STRIPE_COUNT" -S "$STRIPE_SIZE" "$RUN_DIR"
  lfs getstripe -d "$RUN_DIR"
fi

snapshot_hca() {
  local output=$1
  : >"$output"
  local device counter path value
  for device in mlx5_1 mlx5_7; do
    for counter in port_xmit_data port_rcv_data; do
      path="/sys/class/infiniband/$device/ports/1/counters/$counter"
      if [[ -r "$path" ]]; then
        value=$(<"$path")
        printf '%s %s %s\n' "$device" "$counter" "$value" >>"$output"
      fi
    done
  done
}

run_fio() {
  local mode=$1
  local output=$2
  fio \
    --name="lustre-$mode" \
    --directory="$RUN_DIR" \
    --filename_format='fio.$jobnum' \
    --rw="$mode" \
    --bs="$BLOCK_SIZE" \
    --ioengine=libaio \
    --direct=1 \
    --iodepth="$IO_DEPTH" \
    --numjobs="$NUM_JOBS" \
    --size="$SIZE_PER_JOB" \
    --group_reporting=1 \
    --end_fsync=1 \
    --output-format=json \
    --output="$output"
}

report_phase() {
  local phase=$1
  local fio_json=$2
  local before=$3
  local after=$4
  python3 - "$phase" "$fio_json" "$before" "$after" <<'PY'
import json
import sys
from pathlib import Path

phase, fio_path, before_path, after_path = sys.argv[1:]
job = json.loads(Path(fio_path).read_text())["jobs"][0]
stats = job["write" if phase == "write" else "read"]

def counters(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        device, name, value = line.split()
        result[(device, name)] = int(value)
    return result

before = counters(before_path)
after = counters(after_path)
print(
    f"RESULT phase={phase} "
    f"fio_gib_s={stats['bw_bytes'] / 2**30:.3f} "
    f"io_gib={stats['io_bytes'] / 2**30:.3f} "
    f"runtime_s={stats['runtime'] / 1000:.3f}"
)
for device in ("mlx5_1", "mlx5_7"):
    for name in ("port_xmit_data", "port_rcv_data"):
        key = (device, name)
        if key in before and key in after:
            # IB port_*_data counters count 32-bit words.
            delta_gib = (after[key] - before[key]) * 4 / 2**30
            print(
                f"HCA phase={phase} device={device} "
                f"counter={name} delta_gib={delta_gib:.3f}"
            )
PY
}

run_dd_phase() {
  local phase=$1
  local before=$2
  local after=$3
  local start end job
  local -a pids=()

  snapshot_hca "$before"
  start=$(date +%s.%N)
  for ((job = 0; job < NUM_JOBS; job++)); do
    if [[ "$phase" == write ]]; then
      dd if=/dev/zero of="$RUN_DIR/dd.$job" bs="$BLOCK_SIZE" \
        count="$DD_COUNT" oflag=direct conv=fdatasync status=none &
    else
      dd if="$RUN_DIR/dd.$job" of=/dev/null bs="$BLOCK_SIZE" \
        count="$DD_COUNT" iflag=direct status=none &
    fi
    pids+=("$!")
  done
  for job in "${pids[@]}"; do
    wait "$job"
  done
  end=$(date +%s.%N)
  snapshot_hca "$after"

  python3 - "$phase" "$start" "$end" "$NUM_JOBS" "$DD_COUNT" \
    "$BLOCK_SIZE" "$before" "$after" <<'PY'
import sys
from pathlib import Path

phase, start, end, jobs, count, block_size, before_path, after_path = sys.argv[1:]
units = {"K": 2**10, "M": 2**20, "G": 2**30}
block_bytes = int(block_size[:-1]) * units[block_size[-1].upper()]
io_bytes = int(jobs) * int(count) * block_bytes
elapsed = float(end) - float(start)

def counters(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        device, name, value = line.split()
        result[(device, name)] = int(value)
    return result

before = counters(before_path)
after = counters(after_path)
print(
    f"RESULT phase={phase} backend=dd-direct "
    f"gib_s={io_bytes / elapsed / 2**30:.3f} "
    f"io_gib={io_bytes / 2**30:.3f} runtime_s={elapsed:.3f}"
)
for device in ("mlx5_1", "mlx5_7"):
    for name in ("port_xmit_data", "port_rcv_data"):
        key = (device, name)
        if key in before and key in after:
            delta_gib = (after[key] - before[key]) * 4 / 2**30
            print(
                f"HCA phase={phase} device={device} "
                f"counter={name} delta_gib={delta_gib:.3f}"
            )
PY
}

echo "HOST $(hostname)"
if command -v fio >/dev/null; then
  echo "CONFIG backend=fio jobs=$NUM_JOBS size_per_job=$SIZE_PER_JOB bs=$BLOCK_SIZE depth=$IO_DEPTH"
  snapshot_hca /tmp/hca-before-write
  run_fio write /tmp/fio-write.json
  snapshot_hca /tmp/hca-after-write
  run_fio read /tmp/fio-read.json
  snapshot_hca /tmp/hca-after-read
  report_phase write /tmp/fio-write.json /tmp/hca-before-write /tmp/hca-after-write
  report_phase read /tmp/fio-read.json /tmp/hca-after-write /tmp/hca-after-read
else
  echo "CONFIG backend=dd-direct jobs=$NUM_JOBS count=$DD_COUNT bs=$BLOCK_SIZE"
  echo "fio is unavailable; using parallel dd with O_DIRECT"
  run_dd_phase write /tmp/hca-before-write /tmp/hca-after-write
  run_dd_phase read /tmp/hca-before-read /tmp/hca-after-read
fi
