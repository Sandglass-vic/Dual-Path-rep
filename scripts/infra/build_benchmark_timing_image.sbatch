#!/usr/bin/env bash
#SBATCH --job-name=sglang-mooncake013-image
#SBATCH --partition=GAIA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --exclude=dgx-gaia-12,dgx-gaia-18,dgx-gaia-30,dgx-gaia-35,dgx-gaia-38,dgx-gaia-46,dgx-gaia-53
#SBATCH --output=/mnt/lustre/users/sezhou/dual-path/logs/timing-image-%j.out

set -euo pipefail

CODE_HOME=${CODE_HOME:-/labhome/sezhou/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/mnt/lustre/users/sezhou/dual-path}
BASE_IMAGE=${BASE_IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3.sqsh}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-$RUNTIME_HOME/images/candidates/sglang-file-l3-dual-staging-${SLURM_JOB_ID}.sqsh}
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
    python /workspace/dual-path/scripts/infra/patch_sglang_benchmark_timing.py
  '

enroot start "$CONTAINER_NAME" python - <<'PY'
from importlib.metadata import version
from pathlib import Path
from mooncake.engine import TransferEngine

root = Path("/usr/local/lib/python3.12/dist-packages/sglang/srt")
hiradix = (root / "mem_cache/hiradix_cache.py").read_text()
schedule_batch = (root / "managers/schedule_batch.py").read_text()
prefill = (root / "disaggregation/prefill.py").read_text()
decode = (root / "disaggregation/decode.py").read_text()
decode_offload = (
    root / "disaggregation/decode_kvcache_offload_manager.py"
).read_text()
mooncake_conn = (root / "disaggregation/mooncake/conn.py").read_text()
memory_pool_host = (root / "mem_cache/memory_pool_host.py").read_text()
output_processor = (
    root / "managers/scheduler_output_processor_mixin.py"
).read_text()
http_server = (root / "entrypoints/http_server.py").read_text()
assert version("mooncake-transfer-engine-cuda13") == "0.3.13.post1"
assert "DUALPATH_L3_TIMING" in hiradix
assert "prefill_engine=%s" in hiradix
assert "DUALPATH_HICACHE_READ_ONLY" in hiradix
assert "DUALPATH_HICACHE_EVICT_UNLOCKED" in hiradix
assert "DUALPATH_PREFILL_DEFERRED_CACHE_RELEASE" in hiradix
local_l3_end = hiradix.index(
    "self.dualpath_prefetch_local_end[operation.request_id]"
)
assert local_l3_end < hiradix.index(
    "torch.distributed.all_reduce(", local_l3_end
)
assert "DUALPATH_REQ_TIMING" in schedule_batch
assert '"DUALPATH_TTFT_P_TRANSFER_V2 rid=%s ' in prefill
assert "DUALPATH_PREFILL_CACHE_RELEASE" in prefill
assert "DUALPATH_PREFILL_RELEASE_LOCAL_CACHE" in prefill
release_local = prefill.index("DUALPATH_PREFILL_RELEASE_LOCAL_CACHE")
assert prefill.index("self.tree_cache.loading_check()", release_local) < (
    prefill.index("self.tree_cache.evict(evictable_before)", release_local)
)
assert "ongoing_write_through" not in prefill[
    release_local:
    prefill.index("DUALPATH_PREFILL_CACHE_RELEASE")
]
assert "dualpath_transfer_page_count" in prefill
assert "page_aligned_tokens=%d" in prefill
assert "DUALPATH_TTFT_D_RECEIVE" in decode
assert "DUALPATH_DECODE_PD_HOST_STAGING" in decode
assert "DUALPATH_DECODE_STAGING_BACKPRESSURE" in decode
assert '"bounded_dedicated_pool"' in decode
assert "DUALPATH_DECODE_OFFLOAD_START" in decode_offload
assert "DUALPATH_DECODE_OFFLOAD_BACKUP_COMPLETE" in decode_offload
assert "DUALPATH_DECODE_L3_SYNC_COMPLETE" in decode_offload
assert "DUALPATH_DECODE_L3_SYNC_DIR" in decode_offload
assert decode_offload.index('getattr(storage_backend, "file_path", None)') < (
    decode_offload.index('os.environ.get("DUALPATH_DECODE_L3_SYNC_DIR")')
)
assert "cached_tokens // self.page_size" in decode_offload
assert "DUALPATH_BOUNDED_DECODE_HOST_STAGING" in memory_pool_host
assert "SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES" in mooncake_conn
assert "prefill_data_indices.size <= max_batch_indices" in mooncake_conn
assert "DUALPATH_DECODE_HOST_STAGING_COPY" in mooncake_conn
assert mooncake_conn.index(
    "stage_received_kv_to_device"
) < mooncake_conn.index("self.conclude_state = status")
assert "DUALPATH_TTFT_D_DISPATCH" in output_processor
assert "DUALPATH_TTFT_HTTP_YIELD" in http_server
compile(hiradix, str(root / "mem_cache/hiradix_cache.py"), "exec")
compile(schedule_batch, str(root / "managers/schedule_batch.py"), "exec")
compile(prefill, str(root / "disaggregation/prefill.py"), "exec")
compile(decode, str(root / "disaggregation/decode.py"), "exec")
compile(
    decode_offload,
    str(root / "disaggregation/decode_kvcache_offload_manager.py"),
    "exec",
)
compile(
    mooncake_conn,
    str(root / "disaggregation/mooncake/conn.py"),
    "exec",
)
compile(
    memory_pool_host,
    str(root / "mem_cache/memory_pool_host.py"),
    "exec",
)
compile(
    output_processor,
    str(root / "managers/scheduler_output_processor_mixin.py"),
    "exec",
)
compile(http_server, str(root / "entrypoints/http_server.py"), "exec")
print("MOONCAKE_VERSION=0.3.13.post1")
print("SGLANG_BENCHMARK_TIMING_PATCH_OK")
PY

rm -f -- "$OUTPUT_IMAGE"
enroot export --output "$OUTPUT_IMAGE" "$CONTAINER_NAME"
sha256sum "$BASE_IMAGE" "$OUTPUT_IMAGE"
echo "TIMING_IMAGE=$OUTPUT_IMAGE"
