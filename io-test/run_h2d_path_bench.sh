#!/usr/bin/env bash
# H2D/D2H path mini bench on one node: HiCache copy engine vs NIC loopback.
#
#   A1  perftest same-HCA loopback, host <-> GPU      (run_nic_loopback_perftest.sh)
#   A2  Qwen2.5-32B TP4-shaped KV pools, hicache / hicache-kernel / nic
#       at 1, 2 (sharing one HCA) and 4 concurrent GPUs   (h2d_path_bench.py)
#   A3  every case bracketed by HCA counters and nvidia-smi dmon PCIe samples
#   A4  optional: TP4 all-reduce latency while each mode streams H2D
#
# Without an allocation the script requests one on $NODE and re-runs itself.
#
#   io-test/run_h2d_path_bench.sh                  # everything, default g5
#   RUN_PERFTEST=0 io-test/run_h2d_path_bench.sh   # only the Python bench
#   RUN_INTERFERENCE=1 io-test/run_h2d_path_bench.sh
set -euo pipefail

CODE_HOME=${CODE_HOME:-/data/home/yongxiao/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/data/home/yongxiao/mydata}
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched.sqsh}
NODE=${NODE:-g5}
PARTITION=${PARTITION:-run}
TIME_LIMIT=${TIME_LIMIT:-02:00:00}
RUN_PERFTEST=${RUN_PERFTEST:-1}
RUN_PYBENCH=${RUN_PYBENCH:-1}
RUN_INTERFERENCE=${RUN_INTERFERENCE:-0}
CONCURRENCY=${CONCURRENCY:-"0 0,1 0,1,2,3"}
PYBENCH_ARGS=${PYBENCH_ARGS:-}
NIC_SLS=${NIC_SLS:-"0 2"}
INTERFERENCE_SECONDS=${INTERFERENCE_SECONDS:-30}

STEP_CPUS=${STEP_CPUS:-384}
PYBENCH_APPEND=${PYBENCH_APPEND:-0}

# Exclusive, so every step can bind workers to the NUMA node of each GPU.
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  exec salloc -N1 -w "$NODE" -p "$PARTITION" --exclusive --gres=gpu:4 \
    --mem=0 --time="$TIME_LIMIT" -J h2d-path "$0" "$@"
fi

# shellcheck source=../scripts/tests/runtime/pd_network_config.sh
source "$CODE_HOME/scripts/tests/runtime/pd_network_config.sh"
export DUALPATH_PZA100_IB_DEVICE_MAP

result_dir=${RESULT_DIR:-$CODE_HOME/io-test/results/$SLURM_JOB_ID-h2d-path}
mkdir -p "$result_dir"
pybench_jsonl=$result_dir/pybench.jsonl
log=$result_dir/run.log
exec > >(tee -a "$log") 2>&1

step() {
  srun --jobid="$SLURM_JOB_ID" --overlap --unbuffered --nodes=1 --ntasks=1 \
    --nodelist="$NODE" --gres=gpu:4 --cpus-per-task="$STEP_CPUS" --mpi=none "$@" ||
    echo "STEP_FAILED rc=$? cmd=$*"
}

in_container() {
  step --container-image="$IMAGE" --container-mounts=/data:/data \
    --container-workdir="$CODE_HOME" --no-container-mount-home "$@"
}

echo "H2D path bench job=$SLURM_JOB_ID node=$NODE image=$IMAGE -> $result_dir"

if [[ $RUN_PERFTEST == 1 ]]; then
  echo "== A1 perftest loopback"
  step env RESULT_DIR="$result_dir/${PERFTEST_SUBDIR:-perftest}" CODE_HOME="$CODE_HOME" \
    bash "$CODE_HOME/io-test/run_nic_loopback_perftest.sh"
fi

if [[ $RUN_PYBENCH == 1 ]]; then
  [[ $PYBENCH_APPEND == 1 ]] || : >"$pybench_jsonl"
  for gpus in $CONCURRENCY; do
    echo "== A2 h2d_path_bench gpus=$gpus"
    # shellcheck disable=SC2086
    in_container python3 io-test/h2d_path_bench.py --gpus "$gpus" \
      --label "gpus-$gpus" --output "$pybench_jsonl" $PYBENCH_ARGS
  done
  for sl in $NIC_SLS; do
    [[ $sl == 0 ]] && continue
    echo "== A2 nic with MC_IB_SL=$sl"
    # shellcheck disable=SC2086
    in_container env MC_IB_SL="$sl" python3 io-test/h2d_path_bench.py --gpus 0 \
      --modes nic --label "gpus-0-sl$sl" --output "$pybench_jsonl" $PYBENCH_ARGS
  done
fi

if [[ $RUN_INTERFERENCE == 1 ]]; then
  echo "== A4 TP4 all-reduce under H2D load"
  probe=(torchrun --nproc-per-node 4 --master-port 29631 scripts/smoke/nccl_tp4_path_probe.py)
  nccl_env=(env NCCL_IB_HCA=mlx5_0,mlx5_4 NCCL_NET_GDR_LEVEL=PIX NCCL_SOCKET_IFNAME=ibs3)
  in_container "${nccl_env[@]}" "${probe[@]}" >"$result_dir/interference-idle.log" 2>&1
  for mode in hicache nic; do
    in_container python3 io-test/h2d_path_bench.py --gpus 0,1,2,3 --modes "$mode" \
      --layouts layer_first --patterns contiguous --directions h2d \
      --min-case-seconds "$INTERFERENCE_SECONDS" --label "interference-$mode" \
      --output "$result_dir/interference-load.jsonl" &
    load=$!
    sleep 15
    in_container "${nccl_env[@]}" "${probe[@]}" >"$result_dir/interference-$mode.log" 2>&1
    wait "$load"
  done
fi

python3 "$CODE_HOME/io-test/summarize_h2d_path.py" "$result_dir"
echo "H2D_PATH_BENCH_COMPLETE report=$result_dir/REPORT.md"
