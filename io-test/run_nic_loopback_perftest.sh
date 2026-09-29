#!/usr/bin/env bash
# Same-HCA RDMA loopback between host memory and GPU memory on one g5/g6 node.
#
# This is the raw-verbs ceiling for a NIC-driven H2D/D2H: perftest's client
# posts RDMA WRITEs from its buffer into the server's buffer, both on the HCA
# paired with the GPU (GPU0/1 -> mlx5_0, GPU2/3 -> mlx5_4).
#
#   h2d  server buffer on the GPU (--use_cuda), client buffer in host DRAM
#   d2h  client buffer on the GPU, server buffer in host DRAM
#   h2h  both in host DRAM: HCA loopback ceiling without GPUDirect
#
# Runs on the host, not in a container. Every case is wrapped by
# capture_path_counters.py and appended to $RESULT_DIR/cases.jsonl.
#
#   srun -N1 -w g5 -p run --gres=gpu:4 -c 32 --mem=64G -t 60 \
#     io-test/run_nic_loopback_perftest.sh
set -euo pipefail

CODE_HOME=${CODE_HOME:-/data/home/yongxiao/dual-path}
# shellcheck source=../scripts/tests/runtime/pd_network_config.sh
source "$CODE_HOME/scripts/tests/runtime/pd_network_config.sh"

RESULT_DIR=${RESULT_DIR:-$CODE_HOME/io-test/results/${SLURM_JOB_ID:-manual}-h2d-path/perftest}
DIRECTIONS=${DIRECTIONS:-"h2d d2h h2h"}
GPUS=${GPUS:-"0 2"}
SIZES=${SIZES:-"4096 32768 262144 1048576 4194304"}
QPS=${QPS:-"1 4 16"}
SLS=${SLS:-"0"}
# Extra service levels, measured only at SL_SIZE / 4 QPs for h2d and d2h.
SL_EXTRA=${SL_EXTRA:-"2"}
SL_SIZE=${SL_SIZE:-1048576}
DURATION=${DURATION:-4}
SHARED_PAIR=${SHARED_PAIR:-"0 1"}
SHARED_SIZES=${SHARED_SIZES:-"32768 1048576"}
LAT_SIZES=${LAT_SIZES:-"4096 32768 262144"}
LAT_ITERS=${LAT_ITERS:-20000}
CUDA_FLAGS=${CUDA_FLAGS:-}
BASE_PORT=${BASE_PORT:-18700}

mkdir -p "$RESULT_DIR/logs" "$RESULT_DIR/counters"
cases=$RESULT_DIR/cases.jsonl
: >"$cases"

hca_for_gpu() {
  python3 -c 'import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])' \
    "$DUALPATH_PZA100_IB_DEVICE_MAP" "$1"
}

numa_prefix() {
  local node
  node=$(<"/sys/class/infiniband/$1/device/numa_node")
  # Binding fails when the Slurm step owns no CPU on that node; run unbound then.
  if command -v numactl >/dev/null && [[ $node -ge 0 ]] &&
    numactl --cpunodebind="$node" --membind="$node" true 2>/dev/null; then
    echo "numactl --cpunodebind=$node --membind=$node"
  fi
}

buffer_opts() {
  local direction=$1 side=$2 gpu=$3
  case $direction:$side in
    h2d:server | d2h:client) echo "--use_cuda=$gpu $CUDA_FLAGS" ;;
    *) echo "" ;;
  esac
}

# One perftest server/client pair; runs inside `bash -c` under the counters.
run_pair() {
  local tool=$1 hca=$2 port=$3 log=$4 server_opts=$5 client_opts=$6
  shift 6
  local numa
  numa=$(numa_prefix "$hca")
  # shellcheck disable=SC2086
  $numa "$tool" -d "$hca" -p "$port" "$@" $server_opts >"$log.server" 2>&1 &
  local server=$!
  sleep 1
  # shellcheck disable=SC2086
  $numa "$tool" -d "$hca" -p "$port" "$@" $client_opts localhost >"$log.client" 2>&1
  wait "$server"
}
export -f run_pair numa_prefix

record_case() {
  python3 - "$@" >>"$cases" <<'PY'
import json
import re
import sys

kind, label, counters = sys.argv[1], sys.argv[2], sys.argv[3]
meta = json.loads(sys.argv[4])
logs = sys.argv[5:]


def parse(log):
    lines = open(log, errors="replace").read().splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("#bytes"):
            header = re.split(r"\s{2,}|\t", line.strip())
            values = lines[index + 1].split()
            return header, values
    return None, None


record = {"kind": kind, "label": label, **meta, "counters": counters}
if kind == "bw":
    total = 0.0
    for log in logs:
        _, values = parse(log)
        if values:
            total += float(values[3])
    record["bw_gbps"] = total
    record["bw_gbytes_s"] = total / 8
    duration = meta["duration_s"]
    expected = total * 1e9 / 8 * duration
else:
    _, values = parse(logs[0])
    if values:
        names = ["bytes", "iterations", "t_min_us", "t_max_us", "t_typical_us",
                 "t_avg_us", "t_stdev_us", "p99_us", "p999_us"]
        record.update({name: float(value) for name, value in zip(names, values)})
    expected = 0
try:
    verdict = json.load(open(counters))["verdict"]
    if expected:
        verdict["expected_bytes"] = expected
        verdict["gpu_pcie_ratio"] = verdict["gpu_pcie_bytes_est"] / expected
        verdict["hca_wire_ratio"] = verdict["hca_wire_bytes"] / expected
    record["verdict"] = verdict
except (OSError, KeyError, ValueError):
    record["verdict"] = None
print(json.dumps(record, sort_keys=True))
PY
}

capture() {
  local label=$1 hcas=$2 gpus=$3
  shift 3
  local hca_args=()
  for hca in $hcas; do hca_args+=(--hca "$hca"); done
  python3 "$CODE_HOME/io-test/capture_path_counters.py" \
    --out "$RESULT_DIR/counters/$label.json" --label "$label" \
    "${hca_args[@]}" --gpus "$gpus" -- bash -c "$*"
}

port=$BASE_PORT
next_port() { port=$((port + 2)); }

echo "NIC loopback perftest on $(hostname -s) -> $RESULT_DIR"

for gpu in $GPUS; do
  hca=$(hca_for_gpu "$gpu")
  for direction in $DIRECTIONS; do
    for size in $SIZES; do
      for qps in $QPS; do
        for sl in $SLS; do
          label=bw-$direction-gpu$gpu-$hca-s$size-q$qps-sl$sl
          log=$RESULT_DIR/logs/$label
          next_port
          capture "$label" "$hca" "$gpu" \
            "run_pair ib_write_bw $hca $port $log '$(buffer_opts "$direction" server "$gpu")' \
             '$(buffer_opts "$direction" client "$gpu")' \
             -s $size -q $qps -D $DURATION -S $sl --report_gbits -F" || true
          meta=$(printf '{"direction":"%s","gpus":[%s],"hca":"%s","size":%s,"qps":%s,"sl":%s,"duration_s":%s}' \
            "$direction" "$gpu" "$hca" "$size" "$qps" "$sl" "$DURATION")
          record_case bw "$label" "$RESULT_DIR/counters/$label.json" "$meta" "$log.client"
          tail -n 1 "$cases"
        done
      done
    done
  done
  for direction in h2d d2h; do
    for sl in $SL_EXTRA; do
      label=bw-$direction-gpu$gpu-$hca-s$SL_SIZE-q4-sl$sl
      log=$RESULT_DIR/logs/$label
      next_port
      capture "$label" "$hca" "$gpu" \
        "run_pair ib_write_bw $hca $port $log '$(buffer_opts "$direction" server "$gpu")' \
         '$(buffer_opts "$direction" client "$gpu")' \
         -s $SL_SIZE -q 4 -D $DURATION -S $sl --report_gbits -F" || true
      meta=$(printf '{"direction":"%s","gpus":[%s],"hca":"%s","size":%s,"qps":4,"sl":%s,"duration_s":%s}' \
        "$direction" "$gpu" "$hca" "$SL_SIZE" "$sl" "$DURATION")
      record_case bw "$label" "$RESULT_DIR/counters/$label.json" "$meta" "$log.client"
      tail -n 1 "$cases"
    done
  done
done

# Two GPUs behind one HCA at the same time (GPU0 + GPU1 share mlx5_0).
read -r gpu_a gpu_b <<<"$SHARED_PAIR"
hca=$(hca_for_gpu "$gpu_a")
for direction in h2d d2h; do
  for size in $SHARED_SIZES; do
    label=bw-$direction-shared-gpu$gpu_a+$gpu_b-$hca-s$size-q4-sl0
    log=$RESULT_DIR/logs/$label
    next_port
    port_a=$port
    next_port
    port_b=$port
    common="-s $size -q 4 -D $DURATION -S 0 --report_gbits -F"
    capture "$label" "$hca" "$gpu_a,$gpu_b" \
      "run_pair ib_write_bw $hca $port_a $log.a '$(buffer_opts "$direction" server "$gpu_a")' \
         '$(buffer_opts "$direction" client "$gpu_a")' $common &
       run_pair ib_write_bw $hca $port_b $log.b '$(buffer_opts "$direction" server "$gpu_b")' \
         '$(buffer_opts "$direction" client "$gpu_b")' $common &
       wait" || true
    meta=$(printf '{"direction":"%s","gpus":[%s,%s],"hca":"%s","size":%s,"qps":4,"sl":0,"duration_s":%s,"shared_hca":true}' \
      "$direction" "$gpu_a" "$gpu_b" "$hca" "$size" "$DURATION")
    record_case bw "$label" "$RESULT_DIR/counters/$label.json" "$meta" "$log.a.client" "$log.b.client"
    tail -n 1 "$cases"
  done
done

# Ping-pong latency: client host buffer <-> server GPU buffer. perftest only
# supports CUDA write latency with write-with-immediate.
for gpu in $GPUS; do
  hca=$(hca_for_gpu "$gpu")
  for size in $LAT_SIZES; do
    label=lat-host-gpu$gpu-$hca-s$size
    log=$RESULT_DIR/logs/$label
    next_port
    capture "$label" "$hca" "$gpu" \
      "run_pair ib_write_lat $hca $port $log '--use_cuda=$gpu $CUDA_FLAGS' '' \
         -s $size -n $LAT_ITERS -S 0 -F --write_with_imm" || true
    meta=$(printf '{"direction":"pingpong","gpus":[%s],"hca":"%s","size":%s,"sl":0}' "$gpu" "$hca" "$size")
    record_case lat "$label" "$RESULT_DIR/counters/$label.json" "$meta" "$log.client"
    tail -n 1 "$cases"
  done
done

echo "NIC_LOOPBACK_PERFTEST_COMPLETE cases=$cases"
