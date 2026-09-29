#!/usr/bin/env bash
#SBATCH --job-name=sglang-mooncake013-image
#SBATCH --partition=run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=timing-image-%j.out

set -euo pipefail

CODE_HOME=${CODE_HOME:-/labhome/sezhou/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/mnt/lustre/users/sezhou/dual-path}
BASE_IMAGE=${BASE_IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3.sqsh}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-$RUNTIME_HOME/images/sglang-file-l3-dual-staging-${SLURM_JOB_ID}.sqsh}
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

# The root bind mount leaves a root-owned mountpoint in the writable container.
# Make the now-empty path traversable before the unprivileged export walk.
enroot start --root --rw "$CONTAINER_NAME" chmod 0755 /workspace /workspace/dual-path

enroot start "$CONTAINER_NAME" python - <<'PY'
from importlib.metadata import version
import os
from pathlib import Path
from types import SimpleNamespace
from mooncake.engine import TransferEngine
from sglang.srt.mem_cache.hicache_storage import HiCacheFile

root = Path("/usr/local/lib/python3.12/dist-packages/sglang/srt")
hiradix = (root / "mem_cache/hiradix_cache.py").read_text()
schedule_batch = (root / "managers/schedule_batch.py").read_text()
scheduler = (root / "managers/scheduler.py").read_text()
schedule_policy = (root / "managers/schedule_policy.py").read_text()
runtime_checker = (
    root / "managers/scheduler_runtime_checker_mixin.py"
).read_text()
prefill = (root / "disaggregation/prefill.py").read_text()
decode = (root / "disaggregation/decode.py").read_text()
decode_offload = (
    root / "disaggregation/decode_kvcache_offload_manager.py"
).read_text()
hicache_storage = (root / "mem_cache/hicache_storage.py").read_text()
cache_controller = (root / "managers/cache_controller.py").read_text()
mooncake_conn = (root / "disaggregation/mooncake/conn.py").read_text()
memory_pool_host = (root / "mem_cache/memory_pool_host.py").read_text()
allocator = (root / "mem_cache/allocator.py").read_text()
output_processor = (
    root / "managers/scheduler_output_processor_mixin.py"
).read_text()
http_server = (root / "entrypoints/http_server.py").read_text()
ttft_v3 = (root / "utils/dualpath_ttft.py").read_text()
read_path = (root / "utils/dualpath_read_path.py").read_text()
assert version("mooncake-transfer-engine-cuda13") == "0.3.13.post1"
assert "DUALPATH_L3_TIMING" in hiradix
assert "prefill_engine=%s" in hiradix
assert "DUALPATH_HICACHE_READ_ONLY" in hiradix
assert "DUALPATH_HICACHE_EVICT_UNLOCKED" in hiradix
assert "DUALPATH_PREFILL_DEFERRED_CACHE_RELEASE" in hiradix
# patch_hiradix_prefetch_skips() applies to both a pristine and an
# already-patched base, so these must always hold.
for dualpath_marker in (
    "DUALPATH_L3_PREFETCH_SKIP",
    'dualpath_skip_reason = "rate_limited"',
    "reason=no_host_mem",
):
    if dualpath_marker not in hiradix:
        raise SystemExit(f"hiradix_cache.py is missing {dualpath_marker}")
if (
    "last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[rid]"
    in hiradix
):
    raise SystemExit("hiradix_cache.py still unpacks a 4-tuple ongoing_prefetch")
assert "self.dualpath_prefetch_local_end[operation.request_id]" in hiradix
assert "shared_prefetch_can_terminate" in hiradix
assert "shared_prefetch_finish" in hiradix
# The durable L3 hash lookup stays namespace-free, but the restored pages must
# be inserted back under the request's radix namespace or the prefix never
# matches and Prefill recomputes the whole prompt.
assert "extra_key=req.extra_key," in scheduler
assert "extra_key=extra_key" in hiradix
assert "extra_key" not in hiradix[
    hiradix.index("operation = self.cache_controller.prefetch(") :
][:200]
assert "DUALPATH_REQ_TIMING" in schedule_batch
assert "DETAIL as DUALPATH_TTFT_DETAIL" in schedule_batch
assert '"DUALPATH_NO_CROSS_TURN_KV_REUSE", "1"' in schedule_batch
assert 'self.extra_key = f"dualpath:{self.rid}:{extra_key or \'\'}"' in schedule_batch
assert '"DUALPATH_TTFT_P_TRANSFER_V2 rid=%s ' in prefill
assert "DUALPATH_PREFILL_CACHE_RELEASE" in prefill
assert "DUALPATH_PREFILL_DECODE_L3_PREFIX" in prefill
assert "start_decode_l3_reverse" in prefill
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
assert "DUALPATH_DECODE_SCHEDULER_PREFETCH" in decode
assert "_dualpath_progress_l3_prefetch" in decode
assert "drain_decode_l3_prefetch_cancellations" in decode
assert "dualpath_l3_prefetch_operation" in decode
assert '"dualpath_scheduler_prefetch_supported"' in decode
assert 'hasattr(self.kv_manager, "start_decode_l3_prefetch"):\n            decode_req.kv_receiver.init' not in decode
assert "kv_args.dualpath_decode_tp_group = self.gloo_group" in decode
assert "kv_args.dualpath_decode_tp_group = self.attn_tp_cpu_group" not in decode
assert "DUALPATH_DECODE_OFFLOAD_START" in decode_offload
assert "DUAL_PATH_O_DIRECT_READ" in hicache_storage
assert "dual_path_o_direct_read" in hicache_storage
assert "__import__(\"os\").O_DIRECT" in hicache_storage
assert "DUALPATH_O_DIRECT_REUSABLE_PAGE" in cache_controller
assert "DUALPATH_SHARED_PREFETCH_PROGRESS" in cache_controller
assert "shared_prefetch_cancel" in cache_controller
assert "retain_host_indices" in cache_controller
assert "DUALPATH_DECODE_L3_READ" in cache_controller
assert "_dualpath_o_direct_page_buffer" in cache_controller
direct_reuse = cache_controller.index("DUALPATH_O_DIRECT_REUSABLE_PAGE")
direct_get = cache_controller.index(
    "self.storage_backend.get(hash_value, data_page)", direct_reuse
)
direct_scatter = cache_controller.index(
    "self.mem_pool_host.set_from_flat_data_page(", direct_get
)
direct_increment = cache_controller.index(
    "operation.increment(self.page_size)", direct_scatter
)
assert direct_get < direct_scatter < direct_increment
os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = "/tmp/dualpath-odirect-config-check"
file_backend = HiCacheFile(SimpleNamespace(
    tp_rank=0,
    tp_size=8,
    model_name="dualpath-config-check",
    is_mla_model=False,
    extra_config='{"dual_path_o_direct_read":true,"dual_path_o_direct_alignment":4096}',
))
assert file_backend.dualpath_o_direct_read is True
assert file_backend.dualpath_o_direct_alignment == 4096
assert "DUALPATH_DECODE_OFFLOAD_BACKUP_COMPLETE" in decode_offload
# Prefill read path must skip the durable L3 prefix instead of republishing
# the whole context every turn.
assert "dualpath_prefill_l3_prefix_trusted" in decode_offload
assert 'dualpath_durable_prefix = getattr(req, "cached_tokens", 0)' in decode_offload
# The Decode-restored L3 prefix is reported once, not once per side:
# prepare_for_extend counts prefix_indices, so Prefill must not stamp it too.
assert "req.cached_tokens = 0" in prefill
assert "req.cached_tokens = hit_tokens" not in prefill
# The offload guard runs on Decode, so it must not read Prefill-only env.
assert "DUALPATH_HICACHE_READ_ONLY" not in decode_offload
# Queue wait is an explicit TTFT V3 stage on both sides.
assert '"queue_wait",' in ttft_v3
assert "def queue_window_ns" in ttft_v3
assert "dualpath_queue_window_ns" in prefill
assert "dualpath_queue_window_ns" in output_processor
assert '"queue_wait_raw_seconds"' in prefill
assert '"queue_wait_raw_seconds"' in output_processor
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
assert "DUALPATH_DECODE_L3_REVERSE" in mooncake_conn
assert "start_decode_l3_prefetch" in mooncake_conn
assert "shared_prefetch_can_terminate" in mooncake_conn
assert "shared_prefetch_finish" in mooncake_conn
assert "retain_host_indices=True" in mooncake_conn
assert "decode_l3_cancelled_prefetch" in mooncake_conn
receiver_source = mooncake_conn[mooncake_conn.index("class MooncakeKVReceiver"):]
assert "decode_l3_prefetch_operation" not in receiver_source
assert "self.dualpath_scheduler_prefetch_supported = True" in receiver_source
assert "Decode scheduler did not provide Host staging" in receiver_source
assert "load_decode_l3_prefix" not in mooncake_conn
assert "dualpath_host_alloc_start" in decode
assert "DUALPATH_D2P_REVERSE" in prefill
assert "decode_l3_hit_tokens" in mooncake_conn
assert "DUALPATH_DECODE_L3_READ_PATH" in scheduler
assert "dualpath_decode_l3_prefix_ready" in schedule_policy
assert "DUALPATH_DECODE_L3_BOOTSTRAP_ALLOCATION" in runtime_checker
assert 'self.kv_args.prefill_start_layer = 0' in mooncake_conn
assert mooncake_conn.index(
    "stage_received_kv_to_device"
) < mooncake_conn.index("self.conclude_state = status")
assert "DUALPATH_TTFT_D_DISPATCH" in output_processor
assert "SCHEMA_VERSION = 3" in ttft_v3
# Dynamic read path: one decision per request, derived not negotiated.
assert "def resolve" in read_path
assert "alternate_per_turn" not in read_path or "same_turn_lockstep" in read_path
assert '"configured_path": CONFIGURED_PATH' in ttft_v3
assert "resolve_read_path(rid)" in ttft_v3
for dualpath_rel, dualpath_src in (
    ("managers/scheduler.py", scheduler),
    ("disaggregation/prefill.py", prefill),
    ("disaggregation/decode.py", decode),
    ("disaggregation/decode_kvcache_offload_manager.py", decode_offload),
    ("disaggregation/mooncake/conn.py", mooncake_conn),
):
    if "DUALPATH_L3_READ_PATH" in dualpath_src:
        raise SystemExit(
            f"{dualpath_rel} still switches on the process-level read path"
        )
assert "dualpath_is_decode_read(req.rid)" in scheduler
assert "dualpath_decode_capable()" in mooncake_conn
assert "DUALPATH_TTFT_V3_CACHE_CONTROLLER" in cache_controller
assert "dualpath_before_read_start_ns" in cache_controller
assert "dualpath_lustre_read_ns" in cache_controller
assert "dualpath_scatter_ns" in cache_controller
# L3 read pacing must sit inside the timed read on both the O_DIRECT and the
# buffered path, or DUALPATH_L3_READ_BPS would not show in L3 bandwidth.
assert "DUALPATH_L3_READ_BPS" in cache_controller
assert cache_controller.count("dualpath_l3_pace(") == 2
assert (root / "utils/dualpath_l3_throttle.py").exists()
assert "DUALPATH_TTFT_V3_HIRADIX" in hiradix
assert "dualpath_completed_prefetch_timing" in hiradix
assert "DUALPATH_TTFT_V3_SCHEDULER" in scheduler
assert "dualpath_compute_start_event" in scheduler
assert "DUALPATH_TTFT_V3_PREFILL" in prefill
assert "dualpath_ttft_emit" in prefill
assert "DUALPATH_TTFT_V3_DECODE" in decode
assert "dualpath_preallocation_ns" in decode
assert "DUALPATH_TTFT_V3_MOONCAKE" in mooncake_conn
assert "dualpath_d2p_ns" in mooncake_conn
assert "DUALPATH_TTFT_V3_DISPATCH" in output_processor
assert "side=\"decode\"" in output_processor
assert "if DUALPATH_TTFT_DETAIL:" in cache_controller
assert "if DUALPATH_TTFT_DETAIL:" in schedule_batch
compile(hiradix, str(root / "mem_cache/hiradix_cache.py"), "exec")
compile(hicache_storage, str(root / "mem_cache/hicache_storage.py"), "exec")
compile(cache_controller, str(root / "managers/cache_controller.py"), "exec")
compile(schedule_batch, str(root / "managers/schedule_batch.py"), "exec")
compile(scheduler, str(root / "managers/scheduler.py"), "exec")
compile(schedule_policy, str(root / "managers/schedule_policy.py"), "exec")
compile(
    runtime_checker,
    str(root / "managers/scheduler_runtime_checker_mixin.py"),
    "exec",
)
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
compile(allocator, str(root / "mem_cache/allocator.py"), "exec")
compile(
    output_processor,
    str(root / "managers/scheduler_output_processor_mixin.py"),
    "exec",
)
compile(http_server, str(root / "entrypoints/http_server.py"), "exec")
compile(ttft_v3, str(root / "utils/dualpath_ttft.py"), "exec")
compile(read_path, str(root / "utils/dualpath_read_path.py"), "exec")
print("MOONCAKE_VERSION=0.3.13.post1")
print("SGLANG_BENCHMARK_TIMING_PATCH_OK")
PY

rm -f -- "$OUTPUT_IMAGE"
enroot export --output "$OUTPUT_IMAGE" "$CONTAINER_NAME"
sha256sum "$BASE_IMAGE" "$OUTPUT_IMAGE"
echo "TIMING_IMAGE=$OUTPUT_IMAGE"
