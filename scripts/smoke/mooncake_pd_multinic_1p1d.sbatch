#!/usr/bin/env bash
# Minimal TP8 1P1D smoke test for Mooncake PD over eight DGX H100 HCAs.
# HiCache L3 and Decode offload are disabled so only the PD path is exercised.
#SBATCH --job-name=smoke-1p1d-pd-8nic
#SBATCH --partition=GAIA
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --exclude=dgx-gaia-12,dgx-gaia-18,dgx-gaia-30,dgx-gaia-35,dgx-gaia-38,dgx-gaia-46,dgx-gaia-53
#SBATCH --output=/mnt/lustre/users/sezhou/dual-path/logs/slurm-smoke-1p1d-pd-8nic-%j.out

set -euo pipefail

CODE_HOME=${CODE_HOME:-/labhome/sezhou/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/mnt/lustre/users/sezhou/dual-path}
TASK_NAME=${TASK_NAME:-smoke-mooncake-pd-multinic-1p1d-${SLURM_JOB_ID}}

export CODE_HOME RUNTIME_HOME TASK_NAME
export DATASETS=${DATASETS:-32K}
export AGENT_COUNTS=${AGENT_COUNTS:-1}
export REPETITIONS=1
export MAX_TURNS=${MAX_TURNS:-2}
export PAPER_EXPAND_AGENTS=1
export ABLATION_PROFILE=no-l3
export SERVER_SELECTION_TIMEOUT=${SERVER_SELECTION_TIMEOUT:-120}

bash "$CODE_HOME/scripts/pd_1p1d_benchmark.sbatch"

log_dir="$RUNTIME_HOME/logs/latency-benchmark/$TASK_NAME"
python3 - "$log_dir" <<'PY'
from pathlib import Path
import sys

log_dir = Path(sys.argv[1])
prefill_logs = sorted(log_dir.glob("prefill-*.log"))
decode_logs = sorted(log_dir.glob("decode-*.log"))
if len(prefill_logs) != 1 or len(decode_logs) != 1:
    raise SystemExit(
        f"Expected one Prefill and one Decode log, found "
        f"{len(prefill_logs)} and {len(decode_logs)}"
    )

devices = (
    "mlx5_0",
    "mlx5_3",
    "mlx5_4",
    "mlx5_5",
    "mlx5_6",
    "mlx5_9",
    "mlx5_10",
    "mlx5_11",
)
for role, path in (("prefill", prefill_logs[0]), ("decode", decode_logs[0])):
    text = path.read_text(errors="replace")
    missing = [device for device in devices if f"RDMA device: {device}" not in text]
    if missing:
        raise SystemExit(f"{role} did not initialize HCAs: {missing}")
    for failure in (
        "KVTransferError",
        "transport retry counter exceeded",
        "context async event port error",
    ):
        if failure in text:
            raise SystemExit(f"{role} log contains failure: {failure}")

prefill_text = prefill_logs[0].read_text(errors="replace")
decode_text = decode_logs[0].read_text(errors="replace")
if "DUALPATH_TTFT_P_TRANSFER_V2" not in prefill_text:
    raise SystemExit("Prefill PD transfer timing marker is missing")
if "DUALPATH_TTFT_D_RECEIVE" not in decode_text:
    raise SystemExit("Decode PD receive timing marker is missing")

print("PASS: TP8 1P1D Mooncake PD initialized and used all eight HCAs")
print("HCAs:", ",".join(devices))
PY
