#!/usr/bin/env bash
# TP4 all-reduce slowdown under KV traffic on g5 (Prefill) and g6 (Decode).
# See allreduce_interference_bench.py for the probes and traffic schemes.
#
#   nic-test/run_allreduce_interference.sh
#   LEVELS=0,50 SCHEMES=nic_h2d CASE_SECONDS=12 nic-test/run_allreduce_interference.sh   # smoke
#   DUAL_PATH_NCCL_FORCE_NIC=0 ...   # TP4 probes may use NVLink (default 1: NET/IB only)
#   PROBE_SL=0 TRAFFIC_SL=1 ...      # probes on SL0/VL0, Mooncake traffic on SL1/VL1
set -euo pipefail

CODE_HOME=${CODE_HOME:-/data/home/yongxiao/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/data/home/yongxiao/mydata}
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-l3bps-20260929-nich2d.sqsh}
NODES=${NODES:-g5,g6}
TIME_LIMIT=${TIME_LIMIT:-01:30:00}
SCHEMES=${SCHEMES:-p2d,d2p,hicache_h2d,nic_h2d}
LEVELS=${LEVELS:-0,10,20,30,40,50,60,70,80,90,100}
CASE_SECONDS=${CASE_SECONDS:-20}
EXTRA_ARGS=${EXTRA_ARGS:-}
DUAL_PATH_NCCL_FORCE_NIC=${DUAL_PATH_NCCL_FORCE_NIC:-1}
NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# -1 leaves the SL unset (SL0). See docs/vl-traffic-isolation.md for the SL->VL map.
PROBE_SL=${PROBE_SL:--1}
TRAFFIC_SL=${TRAFFIC_SL:--1}
if [[ $DUAL_PATH_NCCL_FORCE_NIC != 0 && $DUAL_PATH_NCCL_FORCE_NIC != 1 ]]; then
  echo "DUAL_PATH_NCCL_FORCE_NIC must be 0 or 1" >&2
  exit 2
fi

nccl_env=(NCCL_IB_HCA=mlx5_0,mlx5_4 NCCL_NET_GDR_LEVEL=PIX NCCL_SOCKET_IFNAME=ibs3 NCCL_DEBUG="$NCCL_DEBUG")
if [[ $DUAL_PATH_NCCL_FORCE_NIC == 1 ]]; then
  nccl_env+=(NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1)
fi

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  exec salloc -N2 -w "$NODES" -p run --exclusive --gres=gpu:4 --mem=0 \
    --time="$TIME_LIMIT" -J ar-interference "$0" "$@"
fi

result_dir=${RESULT_DIR:-$CODE_HOME/nic-test/results/$SLURM_JOB_ID-allreduce-interference}
mkdir -p "$result_dir"
echo "all-reduce interference job=$SLURM_JOB_ID nodes=$NODES nccl_force_nic=$DUAL_PATH_NCCL_FORCE_NIC" \
     "probe_sl=$PROBE_SL traffic_sl=$TRAFFIC_SL -> $result_dir"

# shellcheck disable=SC2086
srun --jobid="$SLURM_JOB_ID" --nodes=2 --ntasks-per-node=1 --gres=gpu:4 \
  --cpus-per-task=384 --mpi=none --unbuffered \
  --container-image="$IMAGE" --container-mounts=/data:/data \
  --container-workdir="$CODE_HOME" --no-container-mount-home \
  env "${nccl_env[@]}" GLOO_SOCKET_IFNAME=ibs3 \
  python3 nic-test/allreduce_interference_bench.py --out "$result_dir" \
  --schemes "$SCHEMES" --levels "$LEVELS" --case-seconds "$CASE_SECONDS" \
  --probe-sl "$PROBE_SL" --traffic-sl "$TRAFFIC_SL" $EXTRA_ARGS \
  2>&1 | grep --line-buffered -vE '^(I|W)[0-9]{4} |InitGoogleLogging|UserWarning|warn\(' | tee "$result_dir/run.log"

python3 "$CODE_HOME/nic-test/plot_allreduce_interference.py" "$result_dir"
echo "ALLREDUCE_INTERFERENCE_COMPLETE dir=$result_dir"
