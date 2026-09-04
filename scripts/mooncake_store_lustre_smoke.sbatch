#!/usr/bin/env bash
#SBATCH --job-name=mooncake-dfs-lustre
#SBATCH --partition=GAIA
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --exclude=dgx-gaia-18,dgx-gaia-30,dgx-gaia-46,dgx-gaia-53
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/lustre/users/sezhou/dual-path/logs/mooncake-store-smoke-%j.out

set -euo pipefail

CODE_HOME=${CODE_HOME:-/labhome/sezhou/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/mnt/lustre/users/sezhou/dual-path}
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3-mooncake0.3.13.post1-timing-v2.sqsh}
TEST_DIR=${TEST_DIR:-$RUNTIME_HOME/kv-cache/mooncake-store-smoke/$SLURM_JOB_ID}
LOG_DIR=${LOG_DIR:-$RUNTIME_HOME/logs/mooncake-store-smoke/$SLURM_JOB_ID}
MASTER_PORT=${MASTER_PORT:-50051}
METADATA_PORT=${METADATA_PORT:-8080}

mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ ${#hosts[@]} -ne 2 ]]; then
  echo "Expected exactly two nodes" >&2
  exit 1
fi
writer_host=${hosts[0]}
reader_host=${hosts[1]}
master="$writer_host:$MASTER_PORT"
metadata="http://$writer_host:$METADATA_PORT/metadata"
manifest="$TEST_DIR/manifest.json"
ready="$TEST_DIR/writer-complete"
done_marker="$TEST_DIR/reader-complete"

mkdir -p "$TEST_DIR/dfs" "$TEST_DIR/client-init/writer" \
  "$TEST_DIR/client-init/reader" "$LOG_DIR"
touch "$TEST_DIR/.dual-path-test-cache-$SLURM_JOB_ID"

cleanup() {
  if [[ -n ${master_step_pid:-} ]]; then
    kill "$master_step_pid" 2>/dev/null || true
    wait "$master_step_pid" 2>/dev/null || true
  fi
  if [[ -f "$TEST_DIR/.dual-path-test-cache-$SLURM_JOB_ID" ]]; then
    rm -rf -- "$TEST_DIR"
    echo "Removed test KV cache: $TEST_DIR"
  fi
}
trap cleanup EXIT INT TERM

common_env=(
  MOONCAKE_ENABLE_DFS=1
  MOONCAKE_DFS_ROOT_DIR="$TEST_DIR/dfs"
  MOONCAKE_DFS_FS_ADAPTER=posix
  MOONCAKE_DFS_SHARD_COUNT=8
  MOONCAKE_DFS_SHARD_CAPACITY=268435456
  MOONCAKE_DFS_ALIGNMENT=4096
  MOONCAKE_DFS_SINGLE_TENANT=true
  MOONCAKE_OFFLOAD_ENABLED=true
  MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=distributed_storage_backend
)

srun --nodes=1 --ntasks=1 --nodelist="$writer_host" \
  --exclusive --exact --mpi=none \
  --container-image="$IMAGE" \
  --container-mounts="/mnt/lustre:/mnt/lustre,$CODE_HOME:/workspace/dual-path" \
  --container-workdir=/workspace/dual-path \
  --no-container-mount-home \
  env "${common_env[@]}" \
    MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="$TEST_DIR/client-init/writer" \
    bash -lc '
      set -euo pipefail
      mooncake_master \
        --rpc_address=0.0.0.0 \
        --enable_http_metadata_server=true \
        --http_metadata_server_host=0.0.0.0 \
        --http_metadata_server_port='"$METADATA_PORT"' \
        >'"$LOG_DIR"'/master.log 2>&1 &
      master_pid=$!
      trap "kill $master_pid 2>/dev/null || true" EXIT
      python - '"$writer_host"' '"$MASTER_PORT"' "$master_pid" <<'"'"'PY'"'"'
import os
import socket
import sys
import time
host, port, pid = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, port), timeout=1):
            break
    except OSError:
        os.kill(pid, 0)
        time.sleep(1)
else:
    raise SystemExit("Mooncake master startup timeout")
PY
      python scripts/mooncake_store_dfs_smoke.py \
        --role write --host '"$writer_host"' \
        --master '"$master"' --metadata-server '"$metadata"' \
        --manifest '"$manifest"'
      touch '"$ready"'
      while [[ ! -f '"$done_marker"' ]]; do sleep 1; done
    ' &
master_step_pid=$!

deadline=$((SECONDS + 300))
while [[ ! -f "$ready" ]]; do
  if ! kill -0 "$master_step_pid" 2>/dev/null; then
    echo "Writer/master step failed" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for writer" >&2
    exit 1
  fi
  sleep 1
done

srun --nodes=1 --ntasks=1 --nodelist="$reader_host" \
  --exclusive --exact --mpi=none \
  --container-image="$IMAGE" \
  --container-mounts="/mnt/lustre:/mnt/lustre,$CODE_HOME:/workspace/dual-path" \
  --container-workdir=/workspace/dual-path \
  --no-container-mount-home \
  env "${common_env[@]}" \
    MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="$TEST_DIR/client-init/reader" \
    python scripts/mooncake_store_dfs_smoke.py \
      --role read --host "$reader_host" \
      --master "$master" --metadata-server "$metadata" \
      --manifest "$manifest"

touch "$done_marker"
wait "$master_step_pid"
master_step_pid=

python - "$TEST_DIR/dfs" <<'PY'
from pathlib import Path
import sys

shards = list(Path(sys.argv[1]).glob("*"))
assert len(shards) == 8, [str(path) for path in shards]
allocated = sum(path.stat().st_blocks * 512 for path in shards)
assert allocated > 0, allocated
print(f"MOONCAKE_DFS_SHARDS={len(shards)}")
print(f"MOONCAKE_DFS_ALLOCATED_BYTES={allocated}")
print("MOONCAKE_STORE_LUSTRE_SMOKE_OK")
PY
