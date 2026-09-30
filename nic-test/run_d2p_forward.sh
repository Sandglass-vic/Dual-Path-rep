#!/usr/bin/env bash
# Prefill (g5) / Decode (g6) forward slowdown under D->P RDMA. See d2p_forward_bench.py.
#
#   nic-test/run_d2p_forward.sh
#   LEVELS=0,25 D2P_THREADS=4 CASE_SECONDS=20 nic-test/run_d2p_forward.sh   # max-rate check
set -euo pipefail

CODE_HOME=${CODE_HOME:-/data/home/yongxiao/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/data/home/yongxiao/mydata}
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-l3bps-20260929-nich2d.sqsh}
NODES=${NODES:-g5,g6}
TIME_LIMIT=${TIME_LIMIT:-00:45:00}
LEVELS=${LEVELS:-0,5,10,15,20,25}
CASE_SECONDS=${CASE_SECONDS:-25}
D2P_THREADS=${D2P_THREADS:-2}
EXTRA_ARGS=${EXTRA_ARGS:-}
NCCL_DEBUG=${NCCL_DEBUG:-WARN}

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  exec salloc -N2 -w "$NODES" -p run --exclusive --gres=gpu:4 --mem=0 \
    --time="$TIME_LIMIT" -J d2p-forward "$0" "$@"
fi

result_dir=${RESULT_DIR:-$CODE_HOME/nic-test/results/$SLURM_JOB_ID-d2p-forward}
mkdir -p "$result_dir"
echo "d2p forward job=$SLURM_JOB_ID nodes=$NODES threads=$D2P_THREADS -> $result_dir"

# shellcheck disable=SC2086
srun --jobid="$SLURM_JOB_ID" --nodes=2 --ntasks-per-node=1 --gres=gpu:4 \
  --cpus-per-task=384 --mpi=none --unbuffered \
  --container-image="$IMAGE" --container-mounts=/data:/data \
  --container-workdir="$CODE_HOME" --no-container-mount-home \
  env NCCL_IB_HCA=mlx5_0,mlx5_4 NCCL_NET_GDR_LEVEL=PIX NCCL_SOCKET_IFNAME=ibs3 NCCL_DEBUG="$NCCL_DEBUG" \
  GLOO_SOCKET_IFNAME=ibs3 \
  python3 nic-test/d2p_forward_bench.py --out "$result_dir" \
  --levels "$LEVELS" --case-seconds "$CASE_SECONDS" --d2p-threads "$D2P_THREADS" $EXTRA_ARGS \
  2>&1 | grep --line-buffered -vE '^(I|W)[0-9]{4} |InitGoogleLogging|UserWarning|warn\(' | tee "$result_dir/run.log"

python3 "$CODE_HOME/nic-test/plot_d2p_forward.py" "$result_dir"
echo "D2P_FORWARD_COMPLETE dir=$result_dir"
