#!/usr/bin/env bash
#SBATCH --job-name=sglang-mooncake013-image
#SBATCH --partition=GAIA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/lustre/users/sezhou/dual-path/logs/timing-image-%j.out

set -euo pipefail

CODE_HOME=${CODE_HOME:-/labhome/sezhou/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/mnt/lustre/users/sezhou/dual-path}
BASE_IMAGE=${BASE_IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3.sqsh}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3-mooncake0.3.13.post1-timing-v2.sqsh}
MOONCAKE_VERSION=${MOONCAKE_VERSION:-0.3.13.post1}
CONTAINER_NAME=sglang-timing-"$SLURM_JOB_ID"

export ENROOT_DATA_PATH=${ENROOT_DATA_PATH:-$RUNTIME_HOME/enroot-data}
export ENROOT_CACHE_PATH=${ENROOT_CACHE_PATH:-$RUNTIME_HOME/enroot-cache}
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$(dirname "$OUTPUT_IMAGE")"

cleanup() {
  enroot remove --force "$CONTAINER_NAME" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

enroot create --name "$CONTAINER_NAME" "$BASE_IMAGE"
enroot start --root --rw \
  --mount "$CODE_HOME:/workspace/dual-path" \
  "$CONTAINER_NAME" \
  bash -lc '
    set -euo pipefail
    python -m pip uninstall -y \
      mooncake-transfer-engine \
      mooncake-transfer-engine-cuda13 \
      mooncake-transfer-engine-cu13 \
      mooncake-transfer-engine-non-cuda || true
    python -m pip install --no-cache-dir \
      "mooncake-transfer-engine-cuda13=='"$MOONCAKE_VERSION"'" \
      "sglang-router==0.3.1"
    python /workspace/dual-path/scripts/infra/patch_sglang_mooncake_dfs.py
    python /workspace/dual-path/scripts/infra/patch_sglang_benchmark_timing.py
  '

enroot start "$CONTAINER_NAME" python - <<'PY'
from importlib.metadata import version
from pathlib import Path
from mooncake.engine import TransferEngine
from mooncake.store import MooncakeDistributedStore, ReplicateConfig

root = Path("/usr/local/lib/python3.12/dist-packages/sglang/srt")
hiradix = (root / "mem_cache/hiradix_cache.py").read_text()
schedule_batch = (root / "managers/schedule_batch.py").read_text()
prefill = (root / "disaggregation/prefill.py").read_text()
decode = (root / "disaggregation/decode.py").read_text()
mooncake_conn = (root / "disaggregation/mooncake/conn.py").read_text()
output_processor = (
    root / "managers/scheduler_output_processor_mixin.py"
).read_text()
http_server = (root / "entrypoints/http_server.py").read_text()
mooncake_backend = (
    root / "mem_cache/storage/mooncake_store/mooncake_store.py"
).read_text()
assert version("mooncake-transfer-engine-cuda13") == "0.3.13.post1"
replicate_config = ReplicateConfig()
assert hasattr(replicate_config, "dfs_replica_num")
assert "DUALPATH_MOONCAKE_DFS_BACKPORT" in mooncake_backend
assert "config.dfs_replica_num = self.dfs_replica_num" in mooncake_backend
assert "DUALPATH_L3_TIMING" in hiradix
assert "prefill_engine=%s" in hiradix
assert hiradix.index(
    "self.dualpath_prefetch_start[req_id] = time.perf_counter()"
) < hiradix.index("operation = self.cache_controller.prefetch(")
assert "DUALPATH_REQ_TIMING" in schedule_batch
assert '"DUALPATH_TTFT_P_TRANSFER_V2 rid=%s ' in prefill
assert "DUALPATH_PREFILL_CACHE_RELEASE" in prefill
assert "DUALPATH_PREFILL_RELEASE_LOCAL_CACHE" in prefill
assert "dualpath_transfer_page_count" in prefill
assert "page_aligned_tokens=%d" in prefill
assert "DUALPATH_TTFT_D_RECEIVE" in decode
assert "SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES" in mooncake_conn
assert "prefill_data_indices.size <= max_batch_indices" in mooncake_conn
assert "DUALPATH_TTFT_D_DISPATCH" in output_processor
assert "DUALPATH_TTFT_HTTP_YIELD" in http_server
compile(hiradix, str(root / "mem_cache/hiradix_cache.py"), "exec")
compile(schedule_batch, str(root / "managers/schedule_batch.py"), "exec")
compile(prefill, str(root / "disaggregation/prefill.py"), "exec")
compile(decode, str(root / "disaggregation/decode.py"), "exec")
compile(
    mooncake_conn,
    str(root / "disaggregation/mooncake/conn.py"),
    "exec",
)
compile(
    output_processor,
    str(root / "managers/scheduler_output_processor_mixin.py"),
    "exec",
)
compile(http_server, str(root / "entrypoints/http_server.py"), "exec")
compile(
    mooncake_backend,
    str(root / "mem_cache/storage/mooncake_store/mooncake_store.py"),
    "exec",
)
print("MOONCAKE_VERSION=0.3.13.post1")
print("MOONCAKE_STORE_DFS_API_OK")
print("SGLANG_MOONCAKE_DFS_BACKPORT_OK")
print("SGLANG_BENCHMARK_TIMING_PATCH_OK")
PY

rm -f -- "$OUTPUT_IMAGE"
enroot export --output "$OUTPUT_IMAGE" "$CONTAINER_NAME"
sha256sum "$BASE_IMAGE" "$OUTPUT_IMAGE"
echo "TIMING_IMAGE=$OUTPUT_IMAGE"
