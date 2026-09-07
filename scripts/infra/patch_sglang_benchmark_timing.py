#!/usr/bin/env python3
"""Add request-scoped benchmark timing logs to NVIDIA SGLang 0.5.7."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return source.replace(old, new, 1)


def patch_hiradix(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_L3_TIMING"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "    def check_hicache_events(self):\n"
        "        self.writing_check()\n"
        "        self.loading_check()\n"
        "        if self.enable_storage:\n"
        "            self.drain_storage_control_queues()\n"
        "        if self.enable_storage_metrics:\n"
        "            self.storage_metrics_collector.log_storage_metrics(\n"
        "                self.cache_controller.storage_backend.get_stats()\n"
        "            )\n",
        "    def check_hicache_events(self):\n"
        "        self.writing_check()\n"
        "        self.loading_check()\n"
        "        if self.enable_storage:\n"
        "            self.drain_storage_control_queues()\n"
        "        if self.enable_storage_metrics:\n"
        "            self.storage_metrics_collector.log_storage_metrics(\n"
        "                self.cache_controller.storage_backend.get_stats()\n"
        "            )\n"
        "        if __import__(\"os\").environ.get(\n"
        "            \"DUALPATH_HICACHE_EVICT_UNLOCKED\", \"0\"\n"
        "        ) == \"1\":\n"
        "            evictable_before = self.evictable_size()\n"
        "            if evictable_before > 0:\n"
        "                self.evict(evictable_before)\n"
        "                logger.info(\n"
        "                    \"DUALPATH_PREFILL_DEFERRED_CACHE_RELEASE \"\n"
        "                    \"evictable_before=%d evictable_after=%d\",\n"
        "                    evictable_before,\n"
        "                    self.evictable_size(),\n"
        "                )\n"
        "            self.evict_host(1 << 60)\n",
        "evict unlocked Prefill cache after HiCache progress",
    )
    source = replace_once(
        source,
        "    def _inc_hit_count(self, node: TreeNode, chunked=False):\n"
        "        # skip the hit count update for chunked requests\n",
        "    def _inc_hit_count(self, node: TreeNode, chunked=False):\n"
        "        # Prefill is an L3 reader only; Decode publishes the full context.\n"
        "        if __import__(\"os\").environ.get(\n"
        "            \"DUALPATH_HICACHE_READ_ONLY\", \"0\"\n"
        "        ) == \"1\":\n"
        "            return\n"
        "        # skip the hit count update for chunked requests\n",
        "disable Prefill local write-through backup",
    )
    source = replace_once(
        source,
        "        self.ongoing_prefetch = {}\n"
        "        self.ongoing_backup = {}\n",
        "        self.ongoing_prefetch = {}\n"
        "        self.dualpath_prefetch_start = {}\n"
        "        self.dualpath_prefetch_local_end = {}\n"
        "        self.ongoing_backup = {}\n",
        "initialize benchmark timing state",
    )
    source = replace_once(
        source,
        "        operation = self.cache_controller.prefetch(\n"
        "            req_id, host_indices, new_input_tokens, last_hash, prefix_keys\n"
        "        )\n"
        "        self.ongoing_prefetch[req_id] = (\n",
        "        self.dualpath_prefetch_start[req_id] = time.perf_counter()\n"
        "        operation = self.cache_controller.prefetch(\n"
        "            req_id, host_indices, new_input_tokens, last_hash, prefix_keys\n"
        "        )\n"
        "        self.ongoing_prefetch[req_id] = (\n",
        "record L3 prefetch start",
    )
    source = replace_once(
        source,
        "        if self.prefetch_stop_policy == \"wait_complete\":\n"
        "            can_terminate = completed\n",
        "        if completed and operation.request_id not in (\n"
        "            self.dualpath_prefetch_local_end\n"
        "        ):\n"
        "            # Capture this rank's storage IO completion before the TP\n"
        "            # completion-consensus all_reduce below.\n"
        "            self.dualpath_prefetch_local_end[operation.request_id] = (\n"
        "                time.perf_counter()\n"
        "            )\n"
        "\n"
        "        if self.prefetch_stop_policy == \"wait_complete\":\n"
        "            can_terminate = completed\n",
        "record rank-local L3 completion before TP consensus",
    )
    source = replace_once(
        source,
        "        logger.debug(f\"Prefetch {req_id} completed with {completed_tokens} tokens\")\n",
        "        logger.debug(f\"Prefetch {req_id} completed with {completed_tokens} tokens\")\n"
        "        dualpath_start = self.dualpath_prefetch_start.pop(req_id, None)\n"
        "        if dualpath_start is not None:\n"
        "            dualpath_end = self.dualpath_prefetch_local_end.pop(\n"
        "                req_id, time.perf_counter()\n"
        "            )\n"
        "            logger.info(\n"
        "                \"DUALPATH_L3_TIMING rid=%s prefill_engine=%s \"\n"
        "                \"tp_rank=%d read_start=%.9f \"\n"
        "                \"read_end=%.9f read_s=%.9f tokens=%d\",\n"
        "                req_id,\n"
        "                __import__(\"os\").environ.get(\"PREFILL_ID\", \"0\"),\n"
        "                self.cache_controller.tp_rank,\n"
        "                dualpath_start,\n"
        "                dualpath_end,\n"
        "                dualpath_end - dualpath_start,\n"
        "                completed_tokens,\n"
        "            )\n",
        "record L3 prefetch completion",
    )
    source = replace_once(
        source,
        "            req_id = cc.prefetch_revoke_queue.get()\n"
        "            info = self.ongoing_prefetch.pop(req_id, None)\n",
        "            req_id = cc.prefetch_revoke_queue.get()\n"
        "            self.dualpath_prefetch_start.pop(req_id, None)\n"
        "            self.dualpath_prefetch_local_end.pop(req_id, None)\n"
        "            info = self.ongoing_prefetch.pop(req_id, None)\n",
        "discard revoked prefetch timing",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_schedule_batch(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_REQ_TIMING"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "        logger.info(f\"{prefix}: {self.time_stats.convert_to_duration()}\")\n"
        "        self.has_log_time_stats = True\n",
        "        logger.info(f\"{prefix}: {self.time_stats.convert_to_duration()}\")\n"
        "        stats = self.time_stats\n"
        "        prefill_s = stats.get_prefill_launch_latency()\n"
        "        logger.info(\n"
        "            \"DUALPATH_REQ_TIMING rid=%s type=%s queue_s=%.9f \"\n"
        "            \"alloc_s=%.9f prefill_s=%.9f forward_s=%.9f \"\n"
        "            \"queue_start=%.9f queue_end=%.9f prefill_start=%.9f \"\n"
        "            \"prefill_end=%.9f prefill_bootstrap_start=%.9f \"\n"
        "            \"decode_prealloc_start=%.9f decode_receive_start=%.9f\",\n"
        "            self.rid,\n"
        "            stats.disagg_mode_str(),\n"
        "            max(0.0, stats.get_queueing_time()),\n"
        "            max(0.0, stats.alloc_waiting_duration),\n"
        "            max(0.0, prefill_s or 0.0),\n"
        "            max(0.0, stats.completion_time - stats.forward_entry_time),\n"
        "            stats.wait_queue_entry_time,\n"
        "            stats.forward_entry_time,\n"
        "            stats.prefill_start_time_host,\n"
        "            stats.prefill_end_time_host,\n"
        "            stats.prefill_bootstrap_queue_entry_time,\n"
        "            stats.decode_prealloc_queue_entry_time,\n"
        "            stats.decode_transfer_queue_entry_time,\n"
        "        )\n"
        "        self.has_log_time_stats = True\n",
        "record request stage durations",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_mooncake_bounded_transfer(path: Path) -> None:
    """Bound all-layer Mooncake calls by KV-index count.

    Backport of sglang#32758 for the 0.5.7 image. Large synchronous calls can
    exhaust transient Mooncake resources and permanently fail the RDMA session.
    """
    source = path.read_text()
    marker = "SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "        self.init_engine()\n"
        "        self.register_buffer_to_engine()\n"
        "        if self.disaggregation_mode == DisaggregationMode.PREFILL:\n",
        "        self.init_engine()\n"
        "        self.register_buffer_to_engine()\n"
        "        self.max_transfer_batch_indices = get_int_env_var(\n"
        "            \"SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES\", 1024\n"
        "        )\n"
        "        if self.disaggregation_mode == DisaggregationMode.PREFILL:\n",
        "initialize bounded Mooncake transfer size",
    )
    source = replace_once(
        source,
        "        else:\n"
        "            # Combining all layers' params in one batch transfer is more efficient\n"
        "            # compared to using multiple threads\n"
        "            return process_layers(layers_params)\n"
        "\n"
        "        return 0\n",
        "        else:\n"
        "            # Combining all layers is faster for short transfers. For long\n"
        "            # transfers, bound each synchronous Mooncake call by KV-index\n"
        "            # count before contiguous address ranges are formed.\n"
        "            max_batch_indices = self.max_transfer_batch_indices\n"
        "            if (\n"
        "                max_batch_indices <= 0\n"
        "                or prefill_data_indices.size <= max_batch_indices\n"
        "            ):\n"
        "                return process_layers(layers_params)\n"
        "            for start in range(\n"
        "                0, prefill_data_indices.size, max_batch_indices\n"
        "            ):\n"
        "                prefill_blocks, decode_blocks = group_concurrent_contiguous(\n"
        "                    prefill_data_indices[start : start + max_batch_indices],\n"
        "                    dst_data_indices[start : start + max_batch_indices],\n"
        "                )\n"
        "                transfer_blocks = []\n"
        "                for src_ptr, dst_ptr, item_len in layers_params:\n"
        "                    for prefill_index, decode_index in zip(\n"
        "                        prefill_blocks, decode_blocks\n"
        "                    ):\n"
        "                        transfer_blocks.append(\n"
        "                            (\n"
        "                                src_ptr + int(prefill_index[0]) * item_len,\n"
        "                                dst_ptr + int(decode_index[0]) * item_len,\n"
        "                                item_len * len(prefill_index),\n"
        "                            )\n"
        "                        )\n"
        "                status = self._transfer_data(\n"
        "                    mooncake_session_id, transfer_blocks\n"
        "                )\n"
        "                if status != 0:\n"
        "                    return status\n"
        "\n"
        "        return 0\n",
        "bound all-layer Mooncake transfer batches",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_prefill_transfer(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_TTFT_P_TRANSFER_V2"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "        req.disagg_kv_sender.send(page_indices, state_indices)\n",
        "        # DUALPATH_TTFT_P_TRANSFER_V2: use an independent timestamp for\n"
        "        # the first actual send.  prefill_transfer_queue_entry_time is\n"
        "        # already assigned when the request enters the transfer queue.\n"
        "        if not hasattr(req, \"dualpath_transfer_start\"):\n"
        "            req.dualpath_transfer_start = time.perf_counter()\n"
        "            req.dualpath_transfer_logical_tokens = 0\n"
        "            req.dualpath_transfer_page_count = 0\n"
        "            req.dualpath_transfer_page_size = page_size\n"
        "        req.dualpath_transfer_logical_tokens += max(0, end_idx - start_idx)\n"
        "        req.dualpath_transfer_page_count += len(page_indices)\n"
        "        req.disagg_kv_sender.send(page_indices, state_indices)\n",
        "record actual KV chunks and first submission",
    )
    source = replace_once(
        source,
        "            elif poll == KVPoll.Success:  # transfer done\n"
        "                release_kv_cache(req, self.tree_cache)  # unlock the tree\n",
        "            elif poll == KVPoll.Success:  # transfer done\n"
        "                dualpath_transfer_end = time.perf_counter()\n"
        "                if self.attn_tp_rank == 0:\n"
        "                    dualpath_transfer_start = getattr(\n"
        "                        req, \"dualpath_transfer_start\", dualpath_transfer_end\n"
        "                    )\n"
        "                    dualpath_logical_tokens = getattr(\n"
        "                        req, \"dualpath_transfer_logical_tokens\", 0\n"
        "                    )\n"
        "                    dualpath_page_count = getattr(\n"
        "                        req, \"dualpath_transfer_page_count\", 0\n"
        "                    )\n"
        "                    dualpath_page_size = getattr(\n"
        "                        req, \"dualpath_transfer_page_size\", 0\n"
        "                    )\n"
        "                    logger.info(\n"
        "                        \"DUALPATH_TTFT_P_TRANSFER_V2 rid=%s transfer_start=%.9f \"\n"
        "                        \"transfer_end=%.9f transfer_s=%.9f logical_tokens=%d \"\n"
        "                        \"page_aligned_tokens=%d page_count=%d page_size=%d\",\n"
        "                        req.rid,\n"
        "                        dualpath_transfer_start,\n"
        "                        dualpath_transfer_end,\n"
        "                        dualpath_transfer_end - dualpath_transfer_start,\n"
        "                        dualpath_logical_tokens,\n"
        "                        dualpath_page_count * dualpath_page_size,\n"
        "                        dualpath_page_count,\n"
        "                        dualpath_page_size,\n"
        "                    )\n"
        "                release_kv_cache(req, self.tree_cache)  # unlock the tree\n"
        "                if __import__(\"os\").environ.get(\n"
        "                    \"DUALPATH_PREFILL_RELEASE_LOCAL_CACHE\", \"0\"\n"
        "                ) == \"1\":\n"
        "                    # L3 load-back adds its own node lock until the CUDA\n"
        "                    # copy event is collected. Prefill compute and P-to-D\n"
        "                    # transfer have already consumed this KV, so collect\n"
        "                    # completed events before measuring evictability.\n"
        "                    if hasattr(self.tree_cache, \"loading_check\"):\n"
        "                        self.tree_cache.loading_check()\n"
        "                    evictable_before = self.tree_cache.evictable_size()\n"
        "                    self.tree_cache.evict(evictable_before)\n"
        "                    if hasattr(self.tree_cache, \"evict_host\"):\n"
        "                        self.tree_cache.evict_host(1 << 60)\n"
        "                    evictable_after = self.tree_cache.evictable_size()\n"
        "                    if self.attn_tp_rank == 0:\n"
        "                        logger.info(\n"
        "                            \"DUALPATH_PREFILL_CACHE_RELEASE rid=%s \"\n"
        "                            \"evictable_before=%d evictable_after=%d\",\n"
        "                            req.rid,\n"
        "                            evictable_before,\n"
        "                            evictable_after,\n"
        "                        )\n",
        "record final KV transfer completion",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_decode_receive(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_TTFT_D_RECEIVE"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "            elif poll == KVPoll.Success:\n"
        "                self._commit_transfer_to_req(decode_req)\n"
        "                indices_to_remove.add(i)\n"
        "                transferred_reqs.append(decode_req.req)\n",
        "            elif poll == KVPoll.Success:\n"
        "                self._commit_transfer_to_req(decode_req)\n"
        "                req = decode_req.req\n"
        "                if self.tp_rank == 0:\n"
        "                    logger.info(\n"
        "                        \"DUALPATH_TTFT_D_RECEIVE rid=%s receive_start=%.9f \"\n"
        "                        \"receive_end=%.9f receive_wait_s=%.9f\",\n"
        "                        req.rid,\n"
        "                        req.time_stats.decode_transfer_queue_entry_time,\n"
        "                        req.time_stats.wait_queue_entry_time,\n"
        "                        req.time_stats.wait_queue_entry_time\n"
        "                        - req.time_stats.decode_transfer_queue_entry_time,\n"
        "                    )\n"
        "                indices_to_remove.add(i)\n"
        "                transferred_reqs.append(req)\n",
        "record Decode KV receive completion",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_first_token_dispatch(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_TTFT_D_DISPATCH"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "        # Note: Logprobs should be handled on the prefill engine.\n"
        "        trace_slice_batch(RequestStage.DECODE_FAKE_OUTPUT, batch.reqs)\n"
        "        self.stream_output(batch.reqs, batch.return_logprob)\n",
        "        # Note: Logprobs should be handled on the prefill engine.\n"
        "        trace_slice_batch(RequestStage.DECODE_FAKE_OUTPUT, batch.reqs)\n"
        "        dualpath_dispatch_time = time.perf_counter()\n"
        "        if self.attn_tp_rank == 0:\n"
        "            for req in batch.reqs:\n"
        "                logger.info(\n"
        "                    \"DUALPATH_TTFT_D_DISPATCH rid=%s dispatch_time=%.9f \"\n"
        "                    \"ready_to_dispatch_s=%.9f\",\n"
        "                    req.rid,\n"
        "                    dualpath_dispatch_time,\n"
        "                    dualpath_dispatch_time\n"
        "                    - req.time_stats.wait_queue_entry_time,\n"
        "                )\n"
        "        self.stream_output(batch.reqs, batch.return_logprob)\n",
        "record first prebuilt token dispatch",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_http_yield(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_TTFT_HTTP_YIELD"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "async def generate_request(obj: GenerateReqInput, request: Request):\n"
        "    \"\"\"Handle a generate request.\"\"\"\n"
        "    if obj.stream:\n"
        "\n"
        "        async def stream_results() -> AsyncIterator[bytes]:\n"
        "            try:\n",
        "async def generate_request(obj: GenerateReqInput, request: Request):\n"
        "    \"\"\"Handle a generate request.\"\"\"\n"
        "    dualpath_handler_entry = time.perf_counter()\n"
        "    if obj.stream:\n"
        "\n"
        "        async def stream_results() -> AsyncIterator[bytes]:\n"
        "            dualpath_first_payload = True\n"
        "            try:\n",
        "record generate handler entry",
    )
    source = replace_once(
        source,
        "                ):\n"
        "                    yield b\"data: \" + orjson.dumps(\n"
        "                        out, option=orjson.OPT_NON_STR_KEYS\n"
        "                    ) + b\"\\n\\n\"\n",
        "                ):\n"
        "                    payload = b\"data: \" + orjson.dumps(\n"
        "                        out, option=orjson.OPT_NON_STR_KEYS\n"
        "                    ) + b\"\\n\\n\"\n"
        "                    if dualpath_first_payload:\n"
        "                        dualpath_yield_time = time.perf_counter()\n"
        "                        logger.info(\n"
        "                            \"DUALPATH_TTFT_HTTP_YIELD rid=%s \"\n"
        "                            \"handler_entry=%.9f yield_time=%.9f \"\n"
        "                            \"handler_to_yield_s=%.9f\",\n"
        "                            out.get(\"meta_info\", {}).get(\"id\", obj.rid),\n"
        "                            dualpath_handler_entry,\n"
        "                            dualpath_yield_time,\n"
        "                            dualpath_yield_time - dualpath_handler_entry,\n"
        "                        )\n"
        "                        dualpath_first_payload = False\n"
        "                    yield payload\n",
        "record first SSE payload yield",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_decode_host_staging(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_DECODE_PD_HOST_STAGING"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "from sglang.srt.mem_cache.memory_pool import (\n"
        "    HybridLinearKVPool,\n"
        "    HybridReqToTokenPool,\n"
        "    KVCache,\n"
        "    NSATokenToKVPool,\n",
        "from sglang.srt.mem_cache.memory_pool import (\n"
        "    HybridLinearKVPool,\n"
        "    HybridReqToTokenPool,\n"
        "    KVCache,\n"
        "    MHATokenToKVPool,\n"
        "    MLATokenToKVPool,\n"
        "    NSATokenToKVPool,\n",
        "import device KV pool types for Decode Host staging",
    )
    source = replace_once(
        source,
        "    SWAKVPool,\n"
        ")\n"
        "from sglang.srt.tracing.trace import trace_event_batch, trace_slice_end\n"
        "from sglang.srt.utils import get_int_env_var\n",
        "    SWAKVPool,\n"
        ")\n"
        "from sglang.srt.mem_cache.memory_pool_host import (\n"
        "    MHATokenToKVPoolHost,\n"
        "    MLATokenToKVPoolHost,\n"
        ")\n"
        "from sglang.srt.tracing.trace import trace_event_batch, trace_slice_end\n"
        "from sglang.srt.utils import get_bool_env_var, get_int_env_var\n",
        "import Host KV pools for Decode Host staging",
    )
    source = replace_once(
        source,
        "        kv_args.kv_data_ptrs = kv_data_ptrs\n"
        "        kv_args.kv_data_lens = kv_data_lens\n"
        "        kv_args.kv_item_lens = kv_item_lens\n",
        "        if get_bool_env_var(\"DUALPATH_DECODE_PD_HOST_STAGING\"):\n"
        "            if self.transfer_backend != TransferBackend.MOONCAKE:\n"
        "                raise ValueError(\n"
        "                    \"DUALPATH_DECODE_PD_HOST_STAGING requires Mooncake PD\"\n"
        "                )\n"
        "            if self.draft_token_to_kv_pool is not None:\n"
        "                raise ValueError(\n"
        "                    \"Decode Host staging does not support draft KV pools\"\n"
        "                )\n"
        "            staging_size_gb = float(\n"
        "                __import__(\"os\").environ.get(\n"
        "                    \"DUALPATH_DECODE_PD_HOST_STAGING_GB\", \"40\"\n"
        "                )\n"
        "            )\n"
        "            if staging_size_gb <= 0:\n"
        "                raise ValueError(\n"
        "                    \"DUALPATH_DECODE_PD_HOST_STAGING_GB must be positive\"\n"
        "                )\n"
        "            staging_args = dict(\n"
        "                device_pool=self.token_to_kv_pool,\n"
        "                host_to_device_ratio=0.0,\n"
        "                host_size=staging_size_gb,\n"
        "                page_size=self.token_to_kv_pool.page_size,\n"
        "                layout=\"layer_first\",\n"
        "                pin_memory=True,\n"
        "            )\n"
        "            if isinstance(self.token_to_kv_pool, MHATokenToKVPool):\n"
        "                host_staging_pool = MHATokenToKVPoolHost(**staging_args)\n"
        "            elif isinstance(self.token_to_kv_pool, MLATokenToKVPool):\n"
        "                host_staging_pool = MLATokenToKVPoolHost(**staging_args)\n"
        "            else:\n"
        "                raise TypeError(\n"
        "                    \"Unsupported KV pool for Decode Host staging: \"\n"
        "                    f\"{type(self.token_to_kv_pool).__name__}\"\n"
        "                )\n"
        "            staging_source = \"bounded_dedicated_pool\"\n"
        "            if isinstance(self.token_to_kv_pool, MHATokenToKVPool):\n"
        "                staging_refs = (\n"
        "                    host_staging_pool.k_data_refs\n"
        "                    + host_staging_pool.v_data_refs\n"
        "                )\n"
        "            elif isinstance(self.token_to_kv_pool, MLATokenToKVPool):\n"
        "                staging_refs = host_staging_pool.data_refs\n"
        "            else:\n"
        "                raise TypeError(\n"
        "                    \"Unsupported KV pool for Decode Host staging: \"\n"
        "                    f\"{type(self.token_to_kv_pool).__name__}\"\n"
        "                )\n"
        "            kv_args.dualpath_decode_device_pool = self.token_to_kv_pool\n"
        "            kv_args.dualpath_decode_host_staging_pool = host_staging_pool\n"
        "            kv_data_ptrs = [tensor.data_ptr() for tensor in staging_refs]\n"
        "            kv_data_lens = [tensor.nbytes for tensor in staging_refs]\n"
        "            logger.info(\n"
        "                \"DUALPATH_DECODE_PD_HOST_STAGING source=%s \"\n"
        "                \"buffers=%d bytes=%d\",\n"
        "                staging_source,\n"
        "                len(kv_data_ptrs),\n"
        "                sum(kv_data_lens),\n"
        "            )\n"
        "\n"
        "        kv_args.kv_data_ptrs = kv_data_ptrs\n"
        "        kv_args.kv_data_lens = kv_data_lens\n"
        "        kv_args.kv_item_lens = kv_item_lens\n",
        "route Decode PD destination to pinned Host staging",
    )
    source = replace_once(
        source,
        "            elif poll == KVPoll.Success:\n"
        "                self._commit_transfer_to_req(decode_req)\n"
        "                req = decode_req.req\n",
        "            elif poll == KVPoll.Success:\n"
        "                dualpath_host_staging_receiver = decode_req.kv_receiver\n"
        "                self._commit_transfer_to_req(decode_req)\n"
        "                req = decode_req.req\n",
        "retain receiver timing across Decode commit",
    )
    source = replace_once(
        source,
        "                req = decode_req.req\n"
        "                if self.tp_rank == 0:\n",
        "                req = decode_req.req\n"
        "                if (\n"
        "                    dualpath_host_staging_receiver is not None\n"
        "                    and getattr(\n"
        "                        dualpath_host_staging_receiver,\n"
        "                        \"host_staging_h2d_end\",\n"
        "                        None,\n"
        "                    )\n"
        "                    is not None\n"
        "                ):\n"
        "                    logger.info(\n"
        "                        \"DUALPATH_TTFT_D_HOST_STAGING rid=%s tp_rank=%d \"\n"
        "                        \"h2d_start=%.9f h2d_end=%.9f h2d_s=%.9f\",\n"
        "                        req.rid,\n"
        "                        self.tp_rank,\n"
        "                        dualpath_host_staging_receiver.host_staging_h2d_start,\n"
        "                        dualpath_host_staging_receiver.host_staging_h2d_end,\n"
        "                        dualpath_host_staging_receiver.host_staging_h2d_end\n"
        "                        - dualpath_host_staging_receiver.host_staging_h2d_start,\n"
        "                    )\n"
        "                if self.tp_rank == 0:\n",
        "record per-rank Decode Host to HBM timing",
    )
    source = replace_once(
        source,
        "            if required_tokens_for_request > allocatable_tokens:\n"
        "                break\n"
        "\n"
        "            allocatable_tokens -= required_tokens_for_request\n"
        "            self._pre_alloc(decode_req.req)\n",
        "            if required_tokens_for_request > allocatable_tokens:\n"
        "                break\n"
        "\n"
        "            if get_bool_env_var(\"DUALPATH_DECODE_PD_HOST_STAGING\"):\n"
        "                staging_page_count = (\n"
        "                    origin_input_len\n"
        "                    + self.token_to_kv_pool_allocator.page_size\n"
        "                    - 1\n"
        "                ) // self.token_to_kv_pool_allocator.page_size\n"
        "                if not self.kv_manager.can_allocate_decode_host_staging(\n"
        "                    staging_page_count\n"
        "                ):\n"
        "                    # DUALPATH_DECODE_STAGING_BACKPRESSURE: preserve FIFO\n"
        "                    # and retry after an in-flight H2D releases Host pages.\n"
        "                    break\n"
        "\n"
        "            allocatable_tokens -= required_tokens_for_request\n"
        "            self._pre_alloc(decode_req.req)\n",
        "apply Decode Host staging admission backpressure",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_bounded_decode_host_staging_pool(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_BOUNDED_DECODE_HOST_STAGING"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "        assert (\n"
        "            self.size > device_pool.size\n"
        "        ), \"The host memory should be larger than the device memory with the current protocol\"\n",
        "        dualpath_bounded_staging = (\n"
        "            __import__(\"os\").environ.get(\n"
        "                \"DUALPATH_DECODE_PD_HOST_STAGING\", \"0\"\n"
        "            ) == \"1\"\n"
        "            and layout == \"layer_first\"\n"
        "            and host_size > 0\n"
        "        )\n"
        "        assert self.size > device_pool.size or dualpath_bounded_staging, (\n"
        "            \"The host memory should be larger than the device memory with \"\n"
        "            \"the current protocol (DUALPATH_BOUNDED_DECODE_HOST_STAGING \"\n"
        "            \"only permits an explicitly sized layer_first PD staging pool)\"\n"
        "        )\n",
        "allow an explicitly bounded Decode PD Host staging pool",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_mooncake_decode_host_staging(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_DECODE_HOST_STAGING_COPY"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "import requests\n"
        "import zmq\n",
        "import requests\n"
        "import torch\n"
        "import zmq\n",
        "import torch for Decode Host staging",
    )
    source = replace_once(
        source,
        "        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)\n"
        "        self.init_engine()\n",
        "        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)\n"
        "        self.decode_device_pool = getattr(\n"
        "            args, \"dualpath_decode_device_pool\", None\n"
        "        )\n"
        "        self.decode_host_staging_pool = getattr(\n"
        "            args, \"dualpath_decode_host_staging_pool\", None\n"
        "        )\n"
        "        self.decode_host_staging_enabled = (\n"
        "            self.decode_host_staging_pool is not None\n"
        "        )\n"
        "        self.decode_h2d_stream = (\n"
        "            torch.cuda.Stream(device=args.gpu_id)\n"
        "            if self.decode_host_staging_enabled\n"
        "            else None\n"
        "        )\n"
        "        self.init_engine()\n",
        "initialize Decode Host staging state",
    )
    source = replace_once(
        source,
        "    def _transfer_data(self, mooncake_session_id, transfer_blocks):\n",
        "    def allocate_decode_host_staging(self, device_page_indices):\n"
        "        page_count = len(device_page_indices)\n"
        "        page_size = self.decode_host_staging_pool.page_size\n"
        "        token_count = page_count * page_size\n"
        "        host_token_indices = self.decode_host_staging_pool.alloc(token_count)\n"
        "        if host_token_indices is None:\n"
        "            raise RuntimeError(\n"
        "                f\"Decode Host staging exhausted for {page_count} pages\"\n"
        "            )\n"
        "        host_page_indices = (\n"
        "            host_token_indices[::page_size]\n"
        "            // page_size\n"
        "        ).numpy().astype(np.int32, copy=False)\n"
        "        return host_token_indices, host_page_indices\n"
        "\n"
        "    def can_allocate_decode_host_staging(self, page_count):\n"
        "        return self.decode_host_staging_pool.available_size() >= (\n"
        "            page_count * self.decode_host_staging_pool.page_size\n"
        "        )\n"
        "\n"
        "    def release_decode_host_staging(self, host_token_indices):\n"
        "        if host_token_indices is not None:\n"
        "            self.decode_host_staging_pool.free(host_token_indices)\n"
        "\n"
        "    def stage_received_kv_to_device(\n"
        "        self, host_token_indices, device_page_indices\n"
        "    ):\n"
        "        \"\"\"Copy one completed request from pinned Host staging to HBM.\"\"\"\n"
        "        assert self.decode_host_staging_enabled\n"
        "        device_pages = torch.from_numpy(\n"
        "            np.asarray(device_page_indices, dtype=np.int64)\n"
        "        )\n"
        "        page_size = self.decode_host_staging_pool.page_size\n"
        "        page_offsets = torch.arange(page_size, dtype=torch.int64)\n"
        "        device_token_indices = (\n"
        "            device_pages[:, None] * page_size\n"
        "            + page_offsets[None, :]\n"
        "        ).flatten()\n"
        "        start = time.perf_counter()\n"
        "        with torch.cuda.stream(self.decode_h2d_stream):\n"
        "            for layer_id in range(self.decode_host_staging_pool.layer_num):\n"
        "                self.decode_host_staging_pool.load_to_device_per_layer(\n"
        "                    self.decode_device_pool,\n"
        "                    host_token_indices,\n"
        "                    device_token_indices,\n"
        "                    layer_id,\n"
        "                    \"direct\",\n"
        "                )\n"
        "        self.decode_h2d_stream.synchronize()\n"
        "        end = time.perf_counter()\n"
        "        logger.info(\n"
        "            \"DUALPATH_DECODE_HOST_STAGING_COPY tp_rank=%d pages=%d \"\n"
        "            \"h2d_start=%.9f h2d_end=%.9f h2d_s=%.9f\",\n"
        "            self.kv_args.engine_rank,\n"
        "            len(device_pages),\n"
        "            start,\n"
        "            end,\n"
        "            end - start,\n"
        "        )\n"
        "        return start, end\n"
        "\n"
        "    def _transfer_data(self, mooncake_session_id, transfer_blocks):\n",
        "copy completed Decode Host staging pages to HBM",
    )
    source = replace_once(
        source,
        "        self.session_id = mgr.get_session_id()\n"
        "        self.conclude_state = None\n"
        "        self.init_time = None\n"
        "        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)\n",
        "        self.session_id = mgr.get_session_id()\n"
        "        self.conclude_state = None\n"
        "        self.init_time = None\n"
        "        self.kv_indices = None\n"
        "        self.host_staging_token_indices = None\n"
        "        self.host_staging_h2d_start = None\n"
        "        self.host_staging_h2d_end = None\n"
        "        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)\n",
        "initialize per-request Decode Host staging state",
    )
    source = replace_once(
        source,
        "    ):\n"
        "        if self.bootstrap_infos is None:\n"
        "            self.kv_mgr.record_failure(\n",
        "    ):\n"
        "        if self.kv_mgr.decode_host_staging_enabled:\n"
        "            self.kv_indices = np.array(kv_indices, dtype=np.int64, copy=True)\n"
        "            (\n"
        "                self.host_staging_token_indices,\n"
        "                kv_indices,\n"
        "            ) = self.kv_mgr.allocate_decode_host_staging(self.kv_indices)\n"
        "        if self.bootstrap_infos is None:\n"
        "            self.kv_mgr.record_failure(\n",
        "retain request page indices for Host to HBM copy",
    )
    source = replace_once(
        source,
        "            if status in (KVPoll.Success, KVPoll.Failed):\n"
        "                self.conclude_state = status\n"
        "            elif status == KVPoll.WaitingForInput:\n",
        "            if status in (KVPoll.Success, KVPoll.Failed):\n"
        "                if (\n"
        "                    status == KVPoll.Success\n"
        "                    and self.kv_mgr.decode_host_staging_enabled\n"
        "                    and self.host_staging_h2d_end is None\n"
        "                ):\n"
        "                    (\n"
        "                        self.host_staging_h2d_start,\n"
        "                        self.host_staging_h2d_end,\n"
        "                    ) = self.kv_mgr.stage_received_kv_to_device(\n"
        "                        self.host_staging_token_indices,\n"
        "                        self.kv_indices,\n"
        "                    )\n"
        "                    self.kv_mgr.release_decode_host_staging(\n"
        "                        self.host_staging_token_indices\n"
        "                    )\n"
        "                    self.host_staging_token_indices = None\n"
        "                self.conclude_state = status\n"
        "            elif status == KVPoll.WaitingForInput:\n",
        "delay Decode transfer success until Host to HBM completes",
    )
    source = replace_once(
        source,
        "    def clear(self) -> None:\n"
        "        if self.bootstrap_room in self.kv_mgr.request_status:\n"
        "            self.kv_mgr.request_status.pop(self.bootstrap_room)\n"
        "\n"
        "        if self.bootstrap_room in self.kv_mgr.required_prefill_response_num_table:\n",
        "    def clear(self) -> None:\n"
        "        if self.host_staging_token_indices is not None:\n"
        "            self.kv_mgr.release_decode_host_staging(\n"
        "                self.host_staging_token_indices\n"
        "            )\n"
        "            self.host_staging_token_indices = None\n"
        "        if self.bootstrap_room in self.kv_mgr.request_status:\n"
        "            self.kv_mgr.request_status.pop(self.bootstrap_room)\n"
        "\n"
        "        if self.bootstrap_room in self.kv_mgr.required_prefill_response_num_table:\n",
        "release Decode Host staging allocation on clear or failure",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_decode_owns_l3_offload(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_DECODE_OFFLOAD_START"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "    def offload_kv_cache(self, req) -> bool:\n",
        "    def _dualpath_mark_l3_complete(self, req_id: str) -> None:\n"
        "        \"\"\"Publish this TP rank's request-scoped L3 completion.\"\"\"\n"
        "        os = __import__(\"os\")\n"
        "        storage_backend = getattr(\n"
        "            self.cache_controller, \"storage_backend\", None\n"
        "        )\n"
        "        file_path = getattr(storage_backend, \"file_path\", None)\n"
        "        sync_dir_value = None\n"
        "        sync_dir_source = None\n"
        "        if file_path:\n"
        "            sync_dir_value = str(\n"
        "                __import__(\"pathlib\").Path(file_path).parent\n"
        "                / \"l3-sync\"\n"
        "            )\n"
        "            sync_dir_source = \"file_backend\"\n"
        "        elif os.environ.get(\"DUALPATH_DECODE_L3_SYNC_DIR\"):\n"
        "            sync_dir_value = os.environ[\"DUALPATH_DECODE_L3_SYNC_DIR\"]\n"
        "            sync_dir_source = \"environment\"\n"
        "        if not sync_dir_value:\n"
        "            return\n"
        "        hashlib = __import__(\"hashlib\")\n"
        "        json = __import__(\"json\")\n"
        "        pathlib = __import__(\"pathlib\")\n"
        "        sync_dir = pathlib.Path(sync_dir_value)\n"
        "        sync_dir.mkdir(parents=True, exist_ok=True)\n"
        "        rank = torch.distributed.get_rank(group=self.tp_group)\n"
        "        stem = hashlib.sha256(req_id.encode()).hexdigest()\n"
        "        final_path = sync_dir / f\"{stem}.rank-{rank}.done\"\n"
        "        temp_path = sync_dir / (\n"
        "            f\".{stem}.rank-{rank}.{os.getpid()}.tmp\"\n"
        "        )\n"
        "        temp_path.write_text(\n"
        "            json.dumps({\"request_id\": req_id, \"tp_rank\": rank}) + \"\\n\"\n"
        "        )\n"
        "        os.replace(temp_path, final_path)\n"
        "        logger.info(\n"
        "            \"DUALPATH_DECODE_L3_SYNC_COMPLETE rid=%s tp_rank=%d \"\n"
        "            \"source=%s path=%s\",\n"
        "            req_id,\n"
        "            rank,\n"
        "            sync_dir_source,\n"
        "            final_path,\n"
        "        )\n"
        "\n"
        "    def offload_kv_cache(self, req) -> bool:\n",
        "add per-rank Decode L3 completion marker",
    )
    source = replace_once(
        source,
        "        # Prefill side offloads page-aligned origin_input_ids, decode side offloads the incremental part\n"
        "        all_tokens = req.origin_input_ids + req.output_ids[:-1]\n"
        "        prefill_offloaded_len = (\n"
        "            len(req.origin_input_ids) // self.page_size * self.page_size\n"
        "        )\n",
        "        # Prefill is read-only and never publishes KV to L3. cached_tokens\n"
        "        # is transferred in Prefill metadata and marks the prefix already\n"
        "        # restored from L3. Decode publishes everything after that prefix:\n"
        "        # Prefill's newly computed KV plus Decode's generated KV.\n"
        "        all_tokens = req.origin_input_ids + req.output_ids[:-1]\n"
        "        cached_tokens = min(\n"
        "            max(int(req.cached_tokens), 0), len(req.origin_input_ids)\n"
        "        )\n"
        "        prefill_offloaded_len = (\n"
        "            cached_tokens // self.page_size * self.page_size\n"
        "        )\n",
        "make Decode own newly computed KV persistence",
    )
    source = replace_once(
        source,
        "        if incremental_aligned_len == 0:\n"
        "            return False\n",
        "        if incremental_aligned_len == 0:\n"
        "            logger.info(\n"
        "                \"DUALPATH_DECODE_OFFLOAD_START rid=%s cached_tokens=%d \"\n"
        "                \"offload_start=%d offload_tokens=0\",\n"
        "                req.rid,\n"
        "                cached_tokens,\n"
        "                prefill_offloaded_len,\n"
        "            )\n"
        "            self._dualpath_mark_l3_complete(req.rid)\n"
        "            self._release_finished_req(req, 0)\n"
        "            return True\n",
        "release Decode HBM when no complete new page exists",
    )
    source = replace_once(
        source,
        "        incremental_tokens = all_tokens[start:end]\n"
        "        incremental_indices = token_indices[start:end]\n"
        "\n"
        "        # Early free prefill-offloaded GPU memory\n",
        "        incremental_tokens = all_tokens[start:end]\n"
        "        incremental_indices = token_indices[start:end]\n"
        "        logger.info(\n"
        "            \"DUALPATH_DECODE_OFFLOAD_START rid=%s cached_tokens=%d \"\n"
        "            \"offload_start=%d offload_tokens=%d\",\n"
        "            req.rid,\n"
        "            cached_tokens,\n"
        "            start,\n"
        "            incremental_aligned_len,\n"
        "        )\n"
        "\n"
        "        # The prefix before start is already durable in L3.\n",
        "record Decode-owned offload range",
    )
    source = replace_once(
        source,
        "            req_id, host_indices, start_time = self.ongoing_backup.pop(ack_id)\n"
        "\n"
        "            # Release host memory\n",
        "            (\n"
        "                req_id,\n"
        "                host_indices,\n"
        "                start_time,\n"
        "                offload_start,\n"
        "                offload_tokens,\n"
        "            ) = self.ongoing_backup.pop(ack_id)\n"
        "\n"
        "            logger.info(\n"
        "                \"DUALPATH_DECODE_OFFLOAD_BACKUP_COMPLETE rid=%s \"\n"
        "                \"offload_start=%d offload_tokens=%d\",\n"
        "                req_id,\n"
        "                offload_start,\n"
        "                offload_tokens,\n"
        "            )\n"
        "            self._dualpath_mark_l3_complete(req_id)\n"
        "            # Release host memory\n",
        "record Decode L3 backup completion",
    )
    source = replace_once(
        source,
        "        self.ongoing_backup[ack_id] = (req.rid, host_indices, start_time)\n",
        "        self.ongoing_backup[ack_id] = (\n"
        "            req.rid,\n"
        "            host_indices,\n"
        "            start_time,\n"
        "            prefill_offloaded_len,\n"
        "            len(incremental_tokens),\n"
        "        )\n",
        "retain Decode offload range until L3 completion",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=Path("/usr/local/lib/python3.12/dist-packages"),
    )
    args = parser.parse_args()
    srt = args.site_packages / "sglang/srt"
    patch_hiradix(srt / "mem_cache/hiradix_cache.py")
    patch_schedule_batch(srt / "managers/schedule_batch.py")
    patch_mooncake_bounded_transfer(
        srt / "disaggregation/mooncake/conn.py"
    )
    patch_prefill_transfer(srt / "disaggregation/prefill.py")
    patch_decode_receive(srt / "disaggregation/decode.py")
    patch_bounded_decode_host_staging_pool(
        srt / "mem_cache/memory_pool_host.py"
    )
    patch_decode_host_staging(srt / "disaggregation/decode.py")
    patch_mooncake_decode_host_staging(
        srt / "disaggregation/mooncake/conn.py"
    )
    patch_decode_owns_l3_offload(
        srt / "disaggregation/decode_kvcache_offload_manager.py"
    )
    patch_first_token_dispatch(
        srt / "managers/scheduler_output_processor_mixin.py"
    )
    patch_http_yield(srt / "entrypoints/http_server.py")


if __name__ == "__main__":
    main()
