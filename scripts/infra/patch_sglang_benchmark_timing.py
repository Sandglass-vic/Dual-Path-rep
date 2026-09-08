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
        "                \"node=%s tp_rank=%d read_start=%.9f \"\n"
        "                \"read_end=%.9f read_s=%.9f tokens=%d\",\n"
        "                req_id,\n"
        "                __import__(\"os\").environ.get(\"PREFILL_ID\", \"0\"),\n"
        "                __import__(\"os\").environ.get(\n"
        "                    \"PREFILL_HOST\",\n"
        "                    __import__(\"os\").environ.get(\"HOSTNAME\", \"unknown\"),\n"
        "                ),\n"
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


def patch_decode_l3_scheduler(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_DECODE_L3_READ_PATH"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "        if self.enable_hicache_storage:\n"
        "            req.init_next_round_input(self.tree_cache)\n"
        "            if req.last_node.backuped:\n",
        "        if self.enable_hicache_storage:\n"
        "            req.init_next_round_input(self.tree_cache)\n"
        "            dualpath_decode_l3_read = (\n"
        "                __import__(\"os\").environ.get(\n"
        "                    \"DUALPATH_L3_READ_PATH\", \"prefill\"\n"
        "                )\n"
        "                == \"decode\"\n"
        "            )\n"
        "            if dualpath_decode_l3_read:\n"
        "                # DUALPATH_DECODE_L3_READ_PATH: Decode loads the durable\n"
        "                # prefix into its PD Host staging and reverse-sends it.\n"
        "                return\n"
        "            if req.last_node.backuped:\n",
        "route L3 data prefetch to Decode",
    )
    source = replace_once(
        source,
        "            req.init_next_round_input(self.tree_cache)\n"
        "            res = adder.add_one_req(\n",
        "            if not getattr(\n"
        "                req, \"dualpath_decode_l3_prefix_ready\", False\n"
        "            ):\n"
        "                req.init_next_round_input(self.tree_cache)\n"
        "            res = adder.add_one_req(\n",
        "preserve Decode-restored prefix at final Prefill admission",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_decode_l3_schedule_policy(path: Path) -> None:
    source = path.read_text()
    marker = "dualpath_decode_l3_prefix_ready"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "        for r in waiting_queue:\n"
        "            prefix_ids = r.origin_input_ids + r.output_ids\n",
        "        for r in waiting_queue:\n"
        "            if getattr(r, \"dualpath_decode_l3_prefix_ready\", False):\n"
        "                # The indices point at Prefill HBM pages populated by a\n"
        "                # Decode-to-Prefill reverse transfer. Do not overwrite\n"
        "                # them with the still-evicted radix metadata match.\n"
        "                continue\n"
        "            prefix_ids = r.origin_input_ids + r.output_ids\n",
        "preserve Decode-provided Prefill prefix indices",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_decode_l3_runtime_checker(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_DECODE_L3_BOOTSTRAP_ALLOCATION"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "        if self.disaggregation_mode == DisaggregationMode.PREFILL:\n"
        "            if len(self.disagg_prefill_inflight_queue) > 0:\n"
        "                return\n",
        "        if self.disaggregation_mode == DisaggregationMode.PREFILL:\n"
        "            if len(self.disagg_prefill_inflight_queue) > 0:\n"
        "                return\n"
        "            if any(\n"
        "                getattr(\n"
        "                    req, \"dualpath_decode_l3_prefix_indices\", None\n"
        "                )\n"
        "                is not None\n"
        "                for req in self.disagg_prefill_bootstrap_queue.queue\n"
        "            ):\n"
        "                # DUALPATH_DECODE_L3_BOOTSTRAP_ALLOCATION: the reverse\n"
        "                # RDMA destination is live but not radix-owned yet.\n"
        "                return\n",
        "defer idle memory check during Decode L3 reverse bootstrap",
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


def patch_prefill_prepare_breakdown(
    scheduler_path: Path, schedule_batch_path: Path
) -> None:
    source = schedule_batch_path.read_text()
    marker = "dualpath_prepare_input_tensors_end"
    if marker not in source:
        source = replace_once(
            source,
            "    def prepare_for_extend(self):\n"
            "        self.forward_mode = ForwardMode.EXTEND\n",
            "    def prepare_for_extend(self):\n"
            "        dualpath_prepare_start = time.perf_counter()\n"
            "        self.forward_mode = ForwardMode.EXTEND\n",
            "start Prefill batch preparation timing",
        )
        source = replace_once(
            source,
            "        # Allocate memory\n"
            "        out_cache_loc, req_pool_indices_tensor, req_pool_indices = alloc_for_extend(\n"
            "            self\n"
            "        )\n",
            "        self.dualpath_prepare_input_tensors_end = time.perf_counter()\n"
            "        # Allocate memory\n"
            "        dualpath_alloc_start = time.perf_counter()\n"
            "        out_cache_loc, req_pool_indices_tensor, req_pool_indices = alloc_for_extend(\n"
            "            self\n"
            "        )\n"
            "        self.dualpath_prepare_alloc_start = dualpath_alloc_start\n"
            "        self.dualpath_prepare_alloc_end = time.perf_counter()\n",
            "time Prefill KV and request-slot allocation",
        )
        source = replace_once(
            source,
            "        if self.return_logprob:\n"
            "            extend_input_logprob_token_ids = torch.tensor(\n",
            "        self.dualpath_prepare_req_fields_end = time.perf_counter()\n"
            "        if self.return_logprob:\n"
            "            extend_input_logprob_token_ids = torch.tensor(\n",
            "time Prefill request field preparation",
        )
        source = replace_once(
            source,
            "        # Build sampling info\n"
            "        self.sampling_info = SamplingBatchInfo.from_schedule_batch(\n"
            "            self,\n"
            "            self.model_config.vocab_size,\n"
            "        )\n",
            "        # Build sampling info\n"
            "        dualpath_sampling_start = time.perf_counter()\n"
            "        self.sampling_info = SamplingBatchInfo.from_schedule_batch(\n"
            "            self,\n"
            "            self.model_config.vocab_size,\n"
            "        )\n"
            "        self.dualpath_prepare_sampling_start = dualpath_sampling_start\n"
            "        self.dualpath_prepare_sampling_end = time.perf_counter()\n"
            "        self.dualpath_prepare_start = dualpath_prepare_start\n"
            "        self.dualpath_prepare_end = time.perf_counter()\n",
            "time Prefill sampling metadata preparation",
        )
        schedule_batch_path.write_text(source)
        print(f"Patched: {schedule_batch_path}")
    else:
        print(f"Already patched: {schedule_batch_path}")

    source = scheduler_path.read_text()
    marker = "DUALPATH_PREFILL_PREP_BREAKDOWN"
    if marker in source:
        print(f"Already patched: {scheduler_path}")
        return
    source = replace_once(
        source,
        "        # Create a new batch\n"
        "        new_batch = ScheduleBatch.init_new(\n",
        "        # Create a new batch\n"
        "        dualpath_batch_init_start = time.perf_counter()\n"
        "        new_batch = ScheduleBatch.init_new(\n",
        "start Prefill ScheduleBatch initialization timing",
    )
    source = replace_once(
        source,
        "            dllm_config=self.dllm_config,\n"
        "        )\n"
        "        if self.enable_hierarchical_cache:\n",
        "            dllm_config=self.dllm_config,\n"
        "        )\n"
        "        dualpath_batch_init_end = time.perf_counter()\n"
        "        if self.enable_hierarchical_cache:\n",
        "finish Prefill ScheduleBatch initialization timing",
    )
    source = replace_once(
        source,
        "        new_batch.prepare_for_extend()\n"
        "\n"
        "        # Mixed-style chunked prefill\n",
        "        dualpath_hicache_consumer_end = time.perf_counter()\n"
        "        new_batch.prepare_for_extend()\n"
        "        dualpath_schedule_return = time.perf_counter()\n"
        "        for dualpath_req in can_run_list:\n"
        "            dualpath_req.dualpath_batch_init_start = dualpath_batch_init_start\n"
        "            dualpath_req.dualpath_batch_init_end = dualpath_batch_init_end\n"
        "            dualpath_req.dualpath_hicache_consumer_end = dualpath_hicache_consumer_end\n"
        "            dualpath_req.dualpath_schedule_return = dualpath_schedule_return\n"
        "\n"
        "        # Mixed-style chunked prefill\n",
        "retain Prefill preparation timestamps on requests",
    )
    source = replace_once(
        source,
        "        if batch.forward_mode == ForwardMode.EXTEND:\n"
        "            current_time = time.perf_counter()\n"
        "            for req in batch.reqs:\n"
        "                req.time_stats.prefill_start_time_host = current_time\n",
        "        if batch.forward_mode == ForwardMode.EXTEND:\n"
        "            current_time = time.perf_counter()\n"
        "            for req in batch.reqs:\n"
        "                req.time_stats.prefill_start_time_host = current_time\n"
        "                logger.info(\n"
        "                    \"DUALPATH_PREFILL_PREP_BREAKDOWN rid=%s tp_rank=%d \"\n"
        "                    \"selected=%.9f batch_init_start=%.9f batch_init_end=%.9f \"\n"
        "                    \"hicache_end=%.9f hicache_consumer=%d \"\n"
        "                    \"prepare_start=%.9f tensors_end=%.9f \"\n"
        "                    \"alloc_start=%.9f alloc_end=%.9f req_fields_end=%.9f \"\n"
        "                    \"sampling_start=%.9f sampling_end=%.9f prepare_end=%.9f \"\n"
        "                    \"schedule_return=%.9f run_batch_start=%.9f tokens=%d cached=%d\",\n"
        "                    req.rid,\n"
        "                    self.attn_tp_rank,\n"
        "                    req.time_stats.forward_entry_time,\n"
        "                    req.dualpath_batch_init_start,\n"
        "                    req.dualpath_batch_init_end,\n"
        "                    req.dualpath_hicache_consumer_end,\n"
        "                    batch.hicache_consumer_index,\n"
        "                    batch.dualpath_prepare_start,\n"
        "                    batch.dualpath_prepare_input_tensors_end,\n"
        "                    batch.dualpath_prepare_alloc_start,\n"
        "                    batch.dualpath_prepare_alloc_end,\n"
        "                    batch.dualpath_prepare_req_fields_end,\n"
        "                    batch.dualpath_prepare_sampling_start,\n"
        "                    batch.dualpath_prepare_sampling_end,\n"
        "                    batch.dualpath_prepare_end,\n"
        "                    req.dualpath_schedule_return,\n"
        "                    current_time,\n"
        "                    len(req.fill_ids),\n"
        "                    len(req.prefix_indices),\n"
        "                )\n",
        "log Prefill preparation breakdown",
    )
    scheduler_path.write_text(source)
    print(f"Patched: {scheduler_path}")


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
        "                        \"DUALPATH_TTFT_P_TRANSFER_V2 rid=%s prefill_engine=%s \"\n"
        "                        \"node=%s transfer_start=%.9f transfer_end=%.9f \"\n"
        "                        \"transfer_s=%.9f logical_tokens=%d \"\n"
        "                        \"page_aligned_tokens=%d page_count=%d page_size=%d\",\n"
        "                        req.rid,\n"
        "                        __import__(\"os\").environ.get(\"PREFILL_ID\", \"0\"),\n"
        "                        __import__(\"os\").environ.get(\n"
        "                            \"PREFILL_HOST\",\n"
        "                            __import__(\"os\").environ.get(\"HOSTNAME\", \"unknown\"),\n"
        "                        ),\n"
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


def patch_prefill_decode_l3_reverse(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_PREFILL_DECODE_L3_PREFIX"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "    def pop_bootstrapped(\n"
        "        self,\n"
        "        return_failed_reqs: bool = False,\n"
        "        rids_to_check: Optional[List[str]] = None,\n"
        "    ) -> List[Req]:\n",
        "    def _prepare_decode_l3_prefix(self, req: Req) -> bool:\n"
        "        if (\n"
        "            __import__(\"os\").environ.get(\n"
        "                \"DUALPATH_L3_READ_PATH\", \"prefill\"\n"
        "            )\n"
        "            != \"decode\"\n"
        "        ):\n"
        "            return True\n"
        "\n"
        "        page_size = self.token_to_kv_pool.page_size\n"
        "        if not getattr(req, \"dualpath_decode_l3_reverse_started\", False):\n"
        "            hit_tokens = self.kv_manager.get_decode_l3_hit_tokens(\n"
        "                req.bootstrap_room\n"
        "            )\n"
        "            hit_tensor = torch.tensor([hit_tokens], dtype=torch.int64)\n"
        "            torch.distributed.all_reduce(\n"
        "                hit_tensor,\n"
        "                op=torch.distributed.ReduceOp.MIN,\n"
        "                group=self.gloo_group,\n"
        "            )\n"
        "            hit_tokens = int(hit_tensor.item())\n"
        "            max_hit_tokens = max(0, len(req.origin_input_ids) - 1)\n"
        "            hit_tokens = min(hit_tokens, max_hit_tokens)\n"
        "            hit_tokens -= hit_tokens % page_size\n"
        "            req.dualpath_decode_l3_hit_tokens = hit_tokens\n"
        "\n"
        "            if hit_tokens > 0:\n"
        "                allocator = self.scheduler.token_to_kv_pool_allocator\n"
        "                prefix_indices = allocator.alloc(hit_tokens)\n"
        "                if prefix_indices is None:\n"
        "                    self.scheduler.tree_cache.evict(hit_tokens)\n"
        "                    prefix_indices = allocator.alloc(hit_tokens)\n"
        "                local_success = int(prefix_indices is not None)\n"
        "                success_tensor = torch.tensor(\n"
        "                    [local_success], dtype=torch.int64\n"
        "                )\n"
        "                torch.distributed.all_reduce(\n"
        "                    success_tensor,\n"
        "                    op=torch.distributed.ReduceOp.MIN,\n"
        "                    group=self.gloo_group,\n"
        "                )\n"
        "                if int(success_tensor.item()) == 0:\n"
        "                    if prefix_indices is not None:\n"
        "                        allocator.free(prefix_indices)\n"
        "                    return False\n"
        "\n"
        "                req.dualpath_decode_l3_prefix_indices = prefix_indices\n"
        "                page_indices = kv_to_page_indices(\n"
        "                    prefix_indices.cpu().numpy(), page_size\n"
        "                )\n"
        "                req.dualpath_decode_l3_reverse_start = (\n"
        "                    __import__(\"time\").perf_counter()\n"
        "                )\n"
        "                self.kv_manager.start_decode_l3_reverse(\n"
        "                    req.bootstrap_room, page_indices\n"
        "                )\n"
        "            req.dualpath_decode_l3_reverse_started = True\n"
        "            if hit_tokens > 0:\n"
        "                return False\n"
        "\n"
        "        hit_tokens = req.dualpath_decode_l3_hit_tokens\n"
        "        if hit_tokens > 0 and not self.kv_manager.decode_l3_reverse_is_done(\n"
        "            req.bootstrap_room\n"
        "        ):\n"
        "            return False\n"
        "\n"
        "        if hit_tokens > 0 and not getattr(\n"
        "            req, \"dualpath_decode_l3_reverse_logged\", False\n"
        "        ):\n"
        "            reverse_end = __import__(\"time\").perf_counter()\n"
        "            reverse_start = req.dualpath_decode_l3_reverse_start\n"
        "            logger.info(\n"
        "                \"DUALPATH_D2P_REVERSE rid=%s prefill_engine=%s \"\n"
        "                \"node=%s tp_rank=%d reverse_start=%.9f \"\n"
        "                \"reverse_end=%.9f reverse_s=%.9f tokens=%d\",\n"
        "                req.rid,\n"
        "                __import__(\"os\").environ.get(\"PREFILL_ID\", \"0\"),\n"
        "                __import__(\"os\").environ.get(\n"
        "                    \"PREFILL_HOST\",\n"
        "                    __import__(\"os\").environ.get(\"HOSTNAME\", \"unknown\"),\n"
        "                ),\n"
        "                self.tp_rank,\n"
        "                reverse_start,\n"
        "                reverse_end,\n"
        "                reverse_end - reverse_start,\n"
        "                hit_tokens,\n"
        "            )\n"
        "            req.dualpath_decode_l3_reverse_logged = True\n"
        "\n"
        "        req.init_next_round_input(None)\n"
        "        req.prefix_indices = getattr(\n"
        "            req,\n"
        "            \"dualpath_decode_l3_prefix_indices\",\n"
        "            torch.empty((0,), dtype=torch.int64),\n"
        "        )\n"
        "        req.last_node = self.scheduler.tree_cache.root_node\n"
        "        req.last_host_node = self.scheduler.tree_cache.root_node\n"
        "        req.host_hit_length = 0\n"
        "        req.cache_protected_len = 0\n"
        "        # ScheduleBatch.prepare_for_extend accounts prefix_indices once.\n"
        "        req.cached_tokens = 0\n"
        "        req.start_send_idx = hit_tokens\n"
        "        req.set_extend_input_len(len(req.fill_ids) - hit_tokens)\n"
        "        req.dualpath_decode_l3_prefix_ready = True\n"
        "        if self.tp_rank == 0:\n"
        "            logger.info(\n"
        "                \"DUALPATH_PREFILL_DECODE_L3_PREFIX rid=%s \"\n"
        "                \"hit_tokens=%d suffix_tokens=%d\",\n"
        "                req.rid,\n"
        "                hit_tokens,\n"
        "                len(req.fill_ids) - hit_tokens,\n"
        "            )\n"
        "        return True\n"
        "\n"
        "    def _release_decode_l3_prefix(self, req: Req) -> None:\n"
        "        prefix_indices = getattr(\n"
        "            req, \"dualpath_decode_l3_prefix_indices\", None\n"
        "        )\n"
        "        if prefix_indices is not None:\n"
        "            self.scheduler.token_to_kv_pool_allocator.free(prefix_indices)\n"
        "            req.dualpath_decode_l3_prefix_indices = None\n"
        "        self.kv_manager.clear_decode_l3_reverse(req.bootstrap_room)\n"
        "\n"
        "    def pop_bootstrapped(\n"
        "        self,\n"
        "        return_failed_reqs: bool = False,\n"
        "        rids_to_check: Optional[List[str]] = None,\n"
        "    ) -> List[Req]:\n",
        "add Decode L3 reverse bootstrap state",
    )
    source = replace_once(
        source,
        "            elif poll == KVPoll.Failed:\n"
        "                error_message = f\"Prefill bootstrap failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}\"\n",
        "            elif poll == KVPoll.Failed:\n"
        "                self._release_decode_l3_prefix(req)\n"
        "                error_message = f\"Prefill bootstrap failed for request rank={self.tp_rank} {req.rid=} {req.bootstrap_room=}\"\n",
        "release reverse prefix after bootstrap failure",
    )
    source = replace_once(
        source,
        "            # KV.WaitingForInput - init here\n"
        "            num_kv_indices = len(req.origin_input_ids)\n",
        "            # KV.WaitingForInput - init here\n"
        "            if not self._prepare_decode_l3_prefix(req):\n"
        "                continue\n"
        "            num_kv_indices = len(req.origin_input_ids)\n",
        "wait for Decode L3 reverse transfer",
    )
    source = replace_once(
        source,
        "            num_pages = kv_to_page_num(num_kv_indices, self.token_to_kv_pool.page_size)\n"
        "            req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)\n",
        "            num_pages = kv_to_page_num(num_kv_indices, self.token_to_kv_pool.page_size)\n"
        "            req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)\n"
        "            if getattr(req, \"dualpath_decode_l3_prefix_ready\", False):\n"
        "                req.disagg_kv_sender.curr_idx = (\n"
        "                    req.dualpath_decode_l3_hit_tokens\n"
        "                    // self.token_to_kv_pool.page_size\n"
        "                )\n",
        "start normal PD send after Decode-restored prefix",
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


def patch_decode_l3_read(path: Path) -> None:
    source = path.read_text()
    marker = "dualpath_decode_storage_backend"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "            kv_args.dualpath_decode_device_pool = self.token_to_kv_pool\n"
        "            kv_args.dualpath_decode_host_staging_pool = host_staging_pool\n"
        "            kv_data_ptrs = [tensor.data_ptr() for tensor in staging_refs]\n",
        "            kv_args.dualpath_decode_device_pool = self.token_to_kv_pool\n"
        "            kv_args.dualpath_decode_host_staging_pool = host_staging_pool\n"
        "            kv_args.dualpath_decode_storage_backend = (\n"
        "                self.scheduler.decode_offload_manager.cache_controller.storage_backend\n"
        "                if self.scheduler.decode_offload_manager is not None\n"
        "                else None\n"
        "            )\n"
        "            kv_data_ptrs = [tensor.data_ptr() for tensor in staging_refs]\n",
        "expose Decode file backend to PD staging manager",
    )
    source = replace_once(
        source,
        "            decode_req.kv_receiver.init(\n"
        "                page_indices, decode_req.metadata_buffer_index, state_indices\n"
        "            )\n",
        "            if (\n"
        "                __import__(\"os\").environ.get(\n"
        "                    \"DUALPATH_L3_READ_PATH\", \"prefill\"\n"
        "                )\n"
        "                == \"decode\"\n"
        "                and hasattr(decode_req.kv_receiver, \"kv_mgr\")\n"
        "                and hasattr(\n"
        "                    decode_req.kv_receiver.kv_mgr,\n"
        "                    \"load_decode_l3_prefix\",\n"
        "                )\n"
        "            ):\n"
        "                decode_req.kv_receiver.init(\n"
        "                    page_indices,\n"
        "                    decode_req.metadata_buffer_index,\n"
        "                    state_indices,\n"
        "                    token_ids=decode_req.req.origin_input_ids,\n"
        "                    request_id=decode_req.req.rid,\n"
        "                )\n"
        "            else:\n"
        "                decode_req.kv_receiver.init(\n"
        "                    page_indices, decode_req.metadata_buffer_index, state_indices\n"
        "                )\n",
        "load Decode L3 prefix into PD staging",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_decode_handoff_breakdown(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_DECODE_HANDOFF_BREAKDOWN"
    if marker in source:
        print(f"Already patched: {path}")
        return
    source = replace_once(
        source,
        "            elif poll == KVPoll.WaitingForInput:\n"
        "                decode_req.waiting_for_input = True\n",
        "            elif poll == KVPoll.WaitingForInput:\n"
        "                decode_req.waiting_for_input = True\n"
        "                if not hasattr(decode_req, \"dualpath_handshake_ready\"):\n"
        "                    decode_req.dualpath_handshake_ready = time.perf_counter()\n",
        "record Decode handshake completion",
    )
    source = replace_once(
        source,
        "            if not decode_req.waiting_for_input:\n"
        "                continue\n"
        "\n"
        "            if self.req_to_token_pool.available_size() <= 0:\n",
        "            if not decode_req.waiting_for_input:\n"
        "                continue\n"
        "\n"
        "            if not hasattr(decode_req, \"dualpath_admission_first_check\"):\n"
        "                decode_req.dualpath_admission_first_check = time.perf_counter()\n"
        "            if self.req_to_token_pool.available_size() <= 0:\n",
        "record first Decode admission check",
    )
    source = replace_once(
        source,
        "            allocatable_tokens -= required_tokens_for_request\n"
        "            self._pre_alloc(decode_req.req)\n"
        "\n"
        "            kv_indices = (\n",
        "            allocatable_tokens -= required_tokens_for_request\n"
        "            dualpath_admission_pass = time.perf_counter()\n"
        "            dualpath_prealloc_start = time.perf_counter()\n"
        "            self._pre_alloc(decode_req.req)\n"
        "            dualpath_prealloc_end = time.perf_counter()\n"
        "\n"
        "            kv_indices = (\n",
        "time Decode HBM preallocation",
    )
    source = replace_once(
        source,
        "            page_indices = kv_to_page_indices(kv_indices, page_size)\n"
        "            if (\n",
        "            page_indices = kv_to_page_indices(kv_indices, page_size)\n"
        "            dualpath_indices_end = time.perf_counter()\n"
        "            dualpath_receiver_init_start = time.perf_counter()\n"
        "            if (\n",
        "time Decode page-index preparation",
    )
    source = replace_once(
        source,
        "                decode_req.kv_receiver.init(\n"
        "                    page_indices, decode_req.metadata_buffer_index, state_indices\n"
        "                )\n"
        "            preallocated_reqs.append(decode_req)\n",
        "                decode_req.kv_receiver.init(\n"
        "                    page_indices, decode_req.metadata_buffer_index, state_indices\n"
        "                )\n"
        "            dualpath_receiver_init_end = time.perf_counter()\n"
        "            logger.info(\n"
        "                \"DUALPATH_DECODE_HANDOFF_BREAKDOWN rid=%s decode_engine=%s \"\n"
        "                \"node=%s tp_rank=%d queue_entry=%.9f handshake_ready=%.9f \"\n"
        "                \"admission_first=%.9f admission_pass=%.9f prealloc_start=%.9f \"\n"
        "                \"prealloc_end=%.9f indices_end=%.9f receiver_init_start=%.9f \"\n"
        "                \"host_alloc_start=%.9f host_alloc_end=%.9f l3_start=%.9f \"\n"
        "                \"l3_end=%.9f receiver_init_end=%.9f tokens=%d\",\n"
        "                decode_req.req.rid,\n"
        "                __import__(\"os\").environ.get(\"DECODE_ID\", \"0\"),\n"
        "                __import__(\"os\").environ.get(\n"
        "                    \"DECODE_HOST\", __import__(\"os\").environ.get(\"HOSTNAME\", \"unknown\")\n"
        "                ),\n"
        "                self.tp_rank,\n"
        "                decode_req.req.time_stats.decode_prealloc_queue_entry_time,\n"
        "                getattr(decode_req, \"dualpath_handshake_ready\", dualpath_admission_pass),\n"
        "                getattr(decode_req, \"dualpath_admission_first_check\", dualpath_admission_pass),\n"
        "                dualpath_admission_pass,\n"
        "                dualpath_prealloc_start,\n"
        "                dualpath_prealloc_end,\n"
        "                dualpath_indices_end,\n"
        "                dualpath_receiver_init_start,\n"
        "                getattr(decode_req.kv_receiver, \"dualpath_host_alloc_start\", dualpath_receiver_init_start),\n"
        "                getattr(decode_req.kv_receiver, \"dualpath_host_alloc_end\", dualpath_receiver_init_start),\n"
        "                getattr(decode_req.kv_receiver, \"dualpath_l3_load_start\", dualpath_receiver_init_start),\n"
        "                getattr(decode_req.kv_receiver, \"dualpath_l3_load_end\", dualpath_receiver_init_start),\n"
        "                dualpath_receiver_init_end,\n"
        "                len(decode_req.req.origin_input_ids),\n"
        "            )\n"
        "            preallocated_reqs.append(decode_req)\n",
        "log Decode handoff breakdown",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_decode_prealloc_inner_breakdown(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_DECODE_PREALLOC_INNER"
    if marker in source:
        print(f"Already patched: {path}")
        return
    source = replace_once(
        source,
        "    def _pre_alloc(self, req: Req) -> torch.Tensor:\n"
        "        \"\"\"Pre-allocate the memory for req_to_token and token_kv_pool\"\"\"\n",
        "    def _pre_alloc(self, req: Req) -> torch.Tensor:\n"
        "        \"\"\"Pre-allocate the memory for req_to_token and token_kv_pool\"\"\"\n"
        "        req.dualpath_prealloc_inner_start = time.perf_counter()\n",
        "start Decode inner preallocation timing",
    )
    source = replace_once(
        source,
        "        req.req_pool_idx = req_pool_indices[0]\n"
        "\n"
        "        # Alloc all tokens for the prebuilt req",
        "        req.req_pool_idx = req_pool_indices[0]\n"
        "        req.dualpath_prealloc_req_slot_end = time.perf_counter()\n"
        "\n"
        "        # Alloc all tokens for the prebuilt req",
        "time Decode request-slot allocation",
    )
    source = replace_once(
        source,
        "        req.kv_allocated_len = fill_len\n"
        "        req.kv_committed_len = fill_len\n"
        "        if self.token_to_kv_pool_allocator.page_size == 1:\n",
        "        req.kv_allocated_len = fill_len\n"
        "        req.kv_committed_len = fill_len\n"
        "        req.dualpath_prealloc_kv_start = time.perf_counter()\n"
        "        if self.token_to_kv_pool_allocator.page_size == 1:\n",
        "start Decode KV-slot allocation timing",
    )
    source = replace_once(
        source,
        "                extend_num_tokens=fill_len,\n"
        "            )\n"
        "\n"
        "        assert (\n",
        "                extend_num_tokens=fill_len,\n"
        "            )\n"
        "        req.dualpath_prealloc_kv_end = time.perf_counter()\n"
        "\n"
        "        assert (\n",
        "finish Decode KV-slot allocation timing",
    )
    source = replace_once(
        source,
        "        self.req_to_token_pool.write((req.req_pool_idx, slice(0, len(kv_loc))), kv_loc)\n"
        "\n"
        "        # populate metadata\n",
        "        req.dualpath_prealloc_pool_write_start = time.perf_counter()\n"
        "        self.req_to_token_pool.write((req.req_pool_idx, slice(0, len(kv_loc))), kv_loc)\n"
        "        req.dualpath_prealloc_pool_write_end = time.perf_counter()\n"
        "\n"
        "        # populate metadata\n",
        "time Decode request-to-token pool write",
    )
    source = replace_once(
        source,
        "        req.fill_ids = req.origin_input_ids + req.output_ids\n"
        "        req.set_extend_input_len(len(req.fill_ids))\n"
        "\n"
        "        return kv_loc\n",
        "        req.fill_ids = req.origin_input_ids + req.output_ids\n"
        "        req.set_extend_input_len(len(req.fill_ids))\n"
        "        req.dualpath_prealloc_inner_end = time.perf_counter()\n"
        "        logger.info(\n"
        "            \"DUALPATH_DECODE_PREALLOC_INNER rid=%s tp_rank=%d \"\n"
        "            \"inner_start=%.9f req_slot_end=%.9f kv_start=%.9f kv_end=%.9f \"\n"
        "            \"pool_write_start=%.9f pool_write_end=%.9f inner_end=%.9f \"\n"
        "            \"tokens=%d page_size=%d\",\n"
        "            req.rid,\n"
        "            self.tp_rank,\n"
        "            req.dualpath_prealloc_inner_start,\n"
        "            req.dualpath_prealloc_req_slot_end,\n"
        "            req.dualpath_prealloc_kv_start,\n"
        "            req.dualpath_prealloc_kv_end,\n"
        "            req.dualpath_prealloc_pool_write_start,\n"
        "            req.dualpath_prealloc_pool_write_end,\n"
        "            req.dualpath_prealloc_inner_end,\n"
        "            fill_len,\n"
        "            self.token_to_kv_pool_allocator.page_size,\n"
        "        )\n"
        "\n"
        "        return kv_loc\n",
        "log Decode inner preallocation breakdown",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_decode_allocator_merge_breakdown(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_DECODE_ALLOCATOR_MERGE"
    if marker in source:
        print(f"Already patched: {path}")
        return
    source = replace_once(
        source,
        "        else:\n"
        "            device = self.token_to_kv_pool_allocator.device\n"
        "            kv_loc = self.token_to_kv_pool_allocator.alloc_extend(\n",
        "        else:\n"
        "            allocator = self.token_to_kv_pool_allocator\n"
        "            device = allocator.device\n"
        "            dualpath_free_before = len(allocator.free_pages)\n"
        "            dualpath_release_before = len(allocator.release_pages)\n"
        "            dualpath_need_pages = fill_len // allocator.page_size + 2\n"
        "            dualpath_merge_start = time.perf_counter()\n"
        "            dualpath_did_merge = bool(\n"
        "                allocator.need_sort\n"
        "                and dualpath_need_pages > dualpath_free_before\n"
        "                and dualpath_release_before > 0\n"
        "            )\n"
        "            if dualpath_did_merge:\n"
        "                allocator.merge_and_sort_free()\n"
        "            dualpath_merge_end = time.perf_counter()\n"
        "            kv_loc = allocator.alloc_extend(\n",
        "separate Decode allocator free-page merge timing",
    )
    source = replace_once(
        source,
        "        req.dualpath_prealloc_inner_end = time.perf_counter()\n"
        "        logger.info(\n"
        "            \"DUALPATH_DECODE_PREALLOC_INNER rid=%s tp_rank=%d \"\n",
        "        req.dualpath_prealloc_inner_end = time.perf_counter()\n"
        "        if self.token_to_kv_pool_allocator.page_size != 1:\n"
        "            logger.info(\n"
        "                \"DUALPATH_DECODE_ALLOCATOR_MERGE rid=%s tp_rank=%d \"\n"
        "                \"merge_start=%.9f merge_end=%.9f merge_s=%.9f \"\n"
        "                \"did_merge=%d free_before=%d release_before=%d \"\n"
        "                \"free_after=%d release_after=%d need_pages=%d tokens=%d\",\n"
        "                req.rid,\n"
        "                self.tp_rank,\n"
        "                dualpath_merge_start,\n"
        "                dualpath_merge_end,\n"
        "                dualpath_merge_end - dualpath_merge_start,\n"
        "                int(dualpath_did_merge),\n"
        "                dualpath_free_before,\n"
        "                dualpath_release_before,\n"
        "                len(self.token_to_kv_pool_allocator.free_pages),\n"
        "                len(self.token_to_kv_pool_allocator.release_pages),\n"
        "                dualpath_need_pages,\n"
        "                fill_len,\n"
        "            )\n"
        "        logger.info(\n"
        "            \"DUALPATH_DECODE_PREALLOC_INNER rid=%s tp_rank=%d \"\n",
        "log Decode allocator merge details",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_allocator_kernel_breakdown(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_ALLOC_EXTEND_DETAIL"
    if marker in source:
        print(f"Already patched: {path}")
        return
    source = replace_once(
        source,
        "import abc\n",
        "import abc\nimport logging\nimport time\n",
        "import logging and time for allocator timing",
    )
    source = replace_once(
        source,
        "if TYPE_CHECKING:\n",
        "logger = logging.getLogger(__name__)\n\n\nif TYPE_CHECKING:\n",
        "initialize allocator timing logger",
    )
    source = replace_once(
        source,
        "        out_indices = torch.empty(\n"
        "            (extend_num_tokens,), dtype=torch.int64, device=self.device\n"
        "        )\n"
        "        alloc_extend_kernel[(bs,)](\n",
        "        dualpath_detail_start = time.perf_counter()\n"
        "        out_indices = torch.empty(\n"
        "            (extend_num_tokens,), dtype=torch.int64, device=self.device\n"
        "        )\n"
        "        dualpath_empty_end = time.perf_counter()\n"
        "        dualpath_kernel_start = time.perf_counter()\n"
        "        alloc_extend_kernel[(bs,)](\n",
        "time Decode paged allocator tensor creation and kernel",
    )
    source = replace_once(
        source,
        "            self.seen_max_num_extend_tokens_next_power_of_2,\n"
        "        )\n"
        "\n"
        "        if self.debug_mode:\n",
        "            self.seen_max_num_extend_tokens_next_power_of_2,\n"
        "        )\n"
        "        dualpath_kernel_end = time.perf_counter()\n"
        "\n"
        "        if self.debug_mode:\n",
        "finish Decode paged allocator kernel timing",
    )
    source = replace_once(
        source,
        "        num_new_pages = get_num_new_pages(\n"
        "            seq_lens=seq_lens_cpu,\n"
        "            page_size=self.page_size,\n"
        "            prefix_lens=prefix_lens_cpu,\n"
        "        )\n"
        "        if num_new_pages > len(self.free_pages):\n",
        "        num_new_pages = get_num_new_pages(\n"
        "            seq_lens=seq_lens_cpu,\n"
        "            page_size=self.page_size,\n"
        "            prefix_lens=prefix_lens_cpu,\n"
        "        )\n"
        "        dualpath_num_pages_end = time.perf_counter()\n"
        "        logger.info(\n"
        "            \"DUALPATH_ALLOC_EXTEND_DETAIL tokens=%d max_tokens=%d \"\n"
        "            \"empty_s=%.9f kernel_s=%.9f num_pages_s=%.9f total_s=%.9f\",\n"
        "            extend_num_tokens,\n"
        "            self.seen_max_num_extend_tokens_next_power_of_2,\n"
        "            dualpath_empty_end - dualpath_detail_start,\n"
        "            dualpath_kernel_end - dualpath_kernel_start,\n"
        "            dualpath_num_pages_end - dualpath_kernel_end,\n"
        "            dualpath_num_pages_end - dualpath_detail_start,\n"
        "        )\n"
        "        if num_new_pages > len(self.free_pages):\n",
        "log Decode paged allocator kernel breakdown",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_decode_receiver_breakdown(path: Path) -> None:
    source = path.read_text()
    marker = "dualpath_host_alloc_start"
    if marker in source:
        print(f"Already patched: {path}")
        return
    source = replace_once(
        source,
        "        if self.kv_mgr.decode_host_staging_enabled:\n"
        "            self.kv_indices = np.array(kv_indices, dtype=np.int64, copy=True)\n"
        "            (\n"
        "                self.host_staging_token_indices,\n"
        "                kv_indices,\n"
        "            ) = self.kv_mgr.allocate_decode_host_staging(self.kv_indices)\n"
        "            self.host_staging_page_indices = np.array(\n",
        "        if self.kv_mgr.decode_host_staging_enabled:\n"
        "            self.kv_indices = np.array(kv_indices, dtype=np.int64, copy=True)\n"
        "            self.dualpath_host_alloc_start = time.perf_counter()\n"
        "            (\n"
        "                self.host_staging_token_indices,\n"
        "                kv_indices,\n"
        "            ) = self.kv_mgr.allocate_decode_host_staging(self.kv_indices)\n"
        "            self.dualpath_host_alloc_end = time.perf_counter()\n"
        "            self.host_staging_page_indices = np.array(\n",
        "time Decode Host staging allocation",
    )
    source = replace_once(
        source,
        "            if token_ids is not None and request_id is not None:\n"
        "                self.decode_l3_hit_tokens = self.kv_mgr.load_decode_l3_prefix(\n"
        "                    request_id, token_ids, self.host_staging_token_indices\n"
        "                )\n",
        "            if token_ids is not None and request_id is not None:\n"
        "                self.dualpath_l3_load_start = time.perf_counter()\n"
        "                self.decode_l3_hit_tokens = self.kv_mgr.load_decode_l3_prefix(\n"
        "                    request_id, token_ids, self.host_staging_token_indices\n"
        "                )\n"
        "                self.dualpath_l3_load_end = time.perf_counter()\n",
        "retain Decode L3 load timestamps on receiver",
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


def patch_mooncake_decode_l3_reverse(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_DECODE_L3_READ"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "from sglang.srt.disaggregation.utils import DisaggregationMode\n"
        "from sglang.srt.server_args import ServerArgs\n",
        "from sglang.srt.disaggregation.utils import DisaggregationMode\n"
        "from sglang.srt.mem_cache.hicache_storage import get_hash_str\n"
        "from sglang.srt.server_args import ServerArgs\n",
        "import HiCache page hash for Decode L3 reads",
    )
    source = replace_once(
        source,
        "    required_dst_info_num: int\n"
        "    is_dummy: bool\n",
        "    required_dst_info_num: int\n"
        "    decode_l3_hit_tokens: int\n"
        "    is_dummy: bool\n",
        "add Decode L3 hit metadata to bootstrap",
    )
    source = replace_once(
        source,
        "            required_dst_info_num=int(msg[7].decode(\"ascii\")),\n"
        "            is_dummy=is_dummy,\n",
        "            required_dst_info_num=int(msg[7].decode(\"ascii\")),\n"
        "            decode_l3_hit_tokens=(\n"
        "                int(msg[8].decode(\"ascii\")) if len(msg) > 8 else 0\n"
        "            ),\n"
        "            is_dummy=is_dummy,\n",
        "parse Decode L3 hit metadata",
    )
    source = replace_once(
        source,
        "class MooncakeKVManager(CommonKVManager):\n"
        "    AUX_DATA_HEADER = b\"AUX_DATA\"\n",
        "class MooncakeKVManager(CommonKVManager):\n"
        "    AUX_DATA_HEADER = b\"AUX_DATA\"\n"
        "    DECODE_L3_REVERSE_FETCH_HEADER = b\"DUALPATH_DECODE_L3_FETCH\"\n"
        "    DECODE_L3_REVERSE_COMPLETE_HEADER = b\"DUALPATH_DECODE_L3_DONE\"\n",
        "define Decode L3 reverse control messages",
    )
    source = replace_once(
        source,
        "        self.max_transfer_batch_indices = get_int_env_var(\n"
        "            \"SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES\", 1024\n"
        "        )\n"
        "        if self.disaggregation_mode == DisaggregationMode.PREFILL:\n",
        "        self.max_transfer_batch_indices = get_int_env_var(\n"
        "            \"SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES\", 1024\n"
        "        )\n"
        "        self.enable_custom_mem_pool = False\n"
        "        self.custom_mem_pool_type = None\n"
        "        self.decode_l3_read_path = (\n"
        "            os.environ.get(\"DUALPATH_L3_READ_PATH\", \"prefill\")\n"
        "            == \"decode\"\n"
        "        )\n"
        "        self.decode_l3_storage_backend = getattr(\n"
        "            args, \"dualpath_decode_storage_backend\", None\n"
        "        )\n"
        "        self.decode_l3_receivers = {}\n"
        "        self.decode_l3_reverse_pending = set()\n"
        "        self.decode_l3_reverse_done = set()\n"
        "        self.decode_l3_reverse_expected = {}\n"
        "        self.decode_l3_reverse_responses = defaultdict(set)\n"
        "        if self.disaggregation_mode == DisaggregationMode.PREFILL:\n",
        "initialize Decode L3 reverse state",
    )
    source = replace_once(
        source,
        "    def _transfer_data(self, mooncake_session_id, transfer_blocks):\n",
        "    def load_decode_l3_prefix(\n"
        "        self, request_id, token_ids, host_token_indices\n"
        "    ):\n"
        "        if (\n"
        "            not self.decode_l3_read_path\n"
        "            or self.decode_l3_storage_backend is None\n"
        "            or not self.decode_host_staging_enabled\n"
        "        ):\n"
        "            return 0\n"
        "        page_size = self.decode_host_staging_pool.page_size\n"
        "        candidate_tokens = max(0, len(token_ids) - 1)\n"
        "        candidate_tokens -= candidate_tokens % page_size\n"
        "        read_start = time.perf_counter()\n"
        "        hit_tokens = 0\n"
        "        prior_hash = None\n"
        "        target = self.decode_host_staging_pool.get_dummy_flat_data_page()\n"
        "        for offset in range(0, candidate_tokens, page_size):\n"
        "            page_tokens = token_ids[offset : offset + page_size]\n"
        "            page_hash = get_hash_str(page_tokens, prior_hash=prior_hash)\n"
        "            data_page = self.decode_l3_storage_backend.get(page_hash, target)\n"
        "            if data_page is None:\n"
        "                break\n"
        "            self.decode_host_staging_pool.set_from_flat_data_page(\n"
        "                int(host_token_indices[offset]), data_page\n"
        "            )\n"
        "            hit_tokens += page_size\n"
        "            prior_hash = page_hash\n"
        "        read_end = time.perf_counter()\n"
        "        logger.info(\n"
        "            \"DUALPATH_DECODE_L3_READ rid=%s decode_engine=%s \"\n"
        "            \"node=%s tp_rank=%d read_start=%.9f read_end=%.9f \"\n"
        "            \"read_s=%.9f tokens=%d\",\n"
        "            request_id,\n"
        "            os.environ.get(\"DECODE_ID\", \"0\"),\n"
        "            os.environ.get(\n"
        "                \"DECODE_HOST\", os.environ.get(\"HOSTNAME\", \"unknown\")\n"
        "            ),\n"
        "            self.kv_args.engine_rank,\n"
        "            read_start,\n"
        "            read_end,\n"
        "            read_end - read_start,\n"
        "            hit_tokens,\n"
        "        )\n"
        "        return hit_tokens\n"
        "\n"
        "    def register_decode_l3_receiver(self, room, receiver):\n"
        "        self.decode_l3_receivers[room] = receiver\n"
        "\n"
        "    def unregister_decode_l3_receiver(self, room, receiver):\n"
        "        if self.decode_l3_receivers.get(room) is receiver:\n"
        "            self.decode_l3_receivers.pop(room, None)\n"
        "\n"
        "    def get_decode_l3_hit_tokens(self, room):\n"
        "        infos = self.transfer_infos.get(room, {}).values()\n"
        "        hits = [info.decode_l3_hit_tokens for info in infos]\n"
        "        return min(hits) if hits else 0\n"
        "\n"
        "    def start_decode_l3_reverse(self, room, prefill_page_indices):\n"
        "        infos = self.transfer_infos.get(room, {})\n"
        "        if not infos or len(prefill_page_indices) == 0:\n"
        "            return\n"
        "        self.decode_l3_reverse_pending.add(room)\n"
        "        self.decode_l3_reverse_done.discard(room)\n"
        "        self.decode_l3_reverse_expected[room] = set(infos.keys())\n"
        "        self.decode_l3_reverse_responses.pop(room, None)\n"
        "        packed_ptrs = b\"\".join(\n"
        "            struct.pack(\"Q\", ptr) for ptr in self.kv_args.kv_data_ptrs\n"
        "        )\n"
        "        for info in infos.values():\n"
        "            self._connect(\n"
        "                format_tcp_address(info.endpoint, info.dst_port),\n"
        "                is_ipv6=is_valid_ipv6_address(info.endpoint),\n"
        "            ).send_multipart(\n"
        "                [\n"
        "                    self.DECODE_L3_REVERSE_FETCH_HEADER,\n"
        "                    str(room).encode(\"ascii\"),\n"
        "                    self.local_ip.encode(\"ascii\"),\n"
        "                    str(self.rank_port).encode(\"ascii\"),\n"
        "                    self.get_session_id().encode(\"ascii\"),\n"
        "                    packed_ptrs,\n"
        "                    np.asarray(\n"
        "                        prefill_page_indices, dtype=np.int32\n"
        "                    ).tobytes(),\n"
        "                ]\n"
        "            )\n"
        "\n"
        "    def decode_l3_reverse_is_pending(self, room):\n"
        "        return room in self.decode_l3_reverse_pending\n"
        "\n"
        "    def decode_l3_reverse_is_done(self, room):\n"
        "        return room in self.decode_l3_reverse_done\n"
        "\n"
        "    def clear_decode_l3_reverse(self, room):\n"
        "        self.decode_l3_reverse_pending.discard(room)\n"
        "        self.decode_l3_reverse_done.discard(room)\n"
        "        self.decode_l3_reverse_expected.pop(room, None)\n"
        "        self.decode_l3_reverse_responses.pop(room, None)\n"
        "\n"
        "    def _handle_decode_l3_reverse_fetch(self, msg):\n"
        "        room = int(msg[1].decode(\"ascii\"))\n"
        "        prefill_endpoint = msg[2].decode(\"ascii\")\n"
        "        prefill_port = int(msg[3].decode(\"ascii\"))\n"
        "        prefill_session_id = msg[4].decode(\"ascii\")\n"
        "        dst_ptrs = list(struct.unpack(f\"{len(msg[5]) // 8}Q\", msg[5]))\n"
        "        dst_indices = np.frombuffer(msg[6], dtype=np.int32)\n"
        "        status = KVPoll.Failed\n"
        "        try:\n"
        "            receiver = self.decode_l3_receivers[room]\n"
        "            src_indices = receiver.host_staging_page_indices[\n"
        "                : len(dst_indices)\n"
        "            ]\n"
        "            if len(src_indices) != len(dst_indices):\n"
        "                raise RuntimeError(\n"
        "                    f\"Decode L3 prefix page mismatch for room {room}: \"\n"
        "                    f\"{len(src_indices)} != {len(dst_indices)}\"\n"
        "                )\n"
        "            reverse_start = time.perf_counter()\n"
        "            if not hasattr(self.kv_args, \"prefill_start_layer\"):\n"
        "                self.kv_args.prefill_start_layer = 0\n"
        "            ret = self._send_kvcache_generic(\n"
        "                mooncake_session_id=prefill_session_id,\n"
        "                src_data_ptrs=self.kv_args.kv_data_ptrs,\n"
        "                dst_data_ptrs=dst_ptrs,\n"
        "                item_lens=self.kv_args.kv_item_lens,\n"
        "                prefill_data_indices=src_indices,\n"
        "                dst_data_indices=dst_indices,\n"
        "                executor=None,\n"
        "            )\n"
        "            reverse_end = time.perf_counter()\n"
        "            status = KVPoll.Success if ret == 0 else KVPoll.Failed\n"
        "            logger.info(\n"
        "                \"DUALPATH_DECODE_L3_REVERSE rid_room=%d \"\n"
        "                \"decode_engine=%s node=%s tp_rank=%d pages=%d \"\n"
        "                \"transfer_s=%.9f status=%d\",\n"
        "                room,\n"
        "                os.environ.get(\"DECODE_ID\", \"0\"),\n"
        "                os.environ.get(\n"
        "                    \"DECODE_HOST\", os.environ.get(\"HOSTNAME\", \"unknown\")\n"
        "                ),\n"
        "                self.kv_args.engine_rank,\n"
        "                len(dst_indices),\n"
        "                reverse_end - reverse_start,\n"
        "                status,\n"
        "            )\n"
        "        except Exception as error:\n"
        "            logger.exception(\n"
        "                \"Decode L3 reverse transfer failed for room %d: %s\",\n"
        "                room,\n"
        "                error,\n"
        "            )\n"
        "        self._connect(\n"
        "            format_tcp_address(prefill_endpoint, prefill_port),\n"
        "            is_ipv6=is_valid_ipv6_address(prefill_endpoint),\n"
        "        ).send_multipart(\n"
        "            [\n"
        "                self.DECODE_L3_REVERSE_COMPLETE_HEADER,\n"
        "                str(room).encode(\"ascii\"),\n"
        "                str(status).encode(\"ascii\"),\n"
        "                self.get_session_id().encode(\"ascii\"),\n"
        "            ]\n"
        "        )\n"
        "\n"
        "    def _transfer_data(self, mooncake_session_id, transfer_blocks):\n",
        "read Decode L3 pages and reverse-send them to Prefill",
    )
    source = replace_once(
        source,
        "                waiting_req_bytes = self.server_socket.recv_multipart()\n"
        "                room = waiting_req_bytes[0].decode(\"ascii\")\n"
        "                mooncake_session_id = waiting_req_bytes[3].decode(\"ascii\")\n",
        "                waiting_req_bytes = self.server_socket.recv_multipart()\n"
        "                if (\n"
        "                    waiting_req_bytes[0]\n"
        "                    == self.DECODE_L3_REVERSE_COMPLETE_HEADER\n"
        "                ):\n"
        "                    reverse_room = int(waiting_req_bytes[1].decode(\"ascii\"))\n"
        "                    reverse_status = int(\n"
        "                        waiting_req_bytes[2].decode(\"ascii\")\n"
        "                    )\n"
        "                    reverse_session = waiting_req_bytes[3].decode(\"ascii\")\n"
        "                    if reverse_status != KVPoll.Success:\n"
        "                        self.record_failure(\n"
        "                            reverse_room,\n"
        "                            \"Decode-to-Prefill L3 reverse transfer failed\",\n"
        "                        )\n"
        "                        self.update_status(reverse_room, KVPoll.Failed)\n"
        "                    else:\n"
        "                        self.decode_l3_reverse_responses[reverse_room].add(\n"
        "                            reverse_session\n"
        "                        )\n"
        "                        expected = self.decode_l3_reverse_expected.get(\n"
        "                            reverse_room, set()\n"
        "                        )\n"
        "                        if expected and expected.issubset(\n"
        "                            self.decode_l3_reverse_responses[reverse_room]\n"
        "                        ):\n"
        "                            self.decode_l3_reverse_done.add(reverse_room)\n"
        "                    continue\n"
        "                room = waiting_req_bytes[0].decode(\"ascii\")\n"
        "                mooncake_session_id = waiting_req_bytes[3].decode(\"ascii\")\n",
        "handle Decode L3 reverse completion on Prefill",
    )
    source = replace_once(
        source,
        "                if msg[0] == MooncakeKVManager.AUX_DATA_HEADER:\n"
        "                    self._handle_aux_data(msg)\n"
        "                    continue\n",
        "                if msg[0] == MooncakeKVManager.AUX_DATA_HEADER:\n"
        "                    self._handle_aux_data(msg)\n"
        "                    continue\n"
        "                if msg[0] == self.DECODE_L3_REVERSE_FETCH_HEADER:\n"
        "                    self._handle_decode_l3_reverse_fetch(msg)\n"
        "                    continue\n",
        "handle Prefill reverse-fetch command on Decode",
    )
    source = replace_once(
        source,
        "                state_indices=state_indices,\n"
        "            )\n"
        "\n"
        "    def poll(self) -> KVPoll:\n"
        "        if self.conclude_state is None:\n"
        "            status = self.kv_mgr.check_status(self.bootstrap_room)\n"
        "            if status in (KVPoll.Success, KVPoll.Failed):\n",
        "                state_indices=state_indices,\n"
        "            )\n"
        "\n"
        "    def poll(self) -> KVPoll:\n"
        "        if self.conclude_state is None:\n"
        "            status = self.kv_mgr.check_status(self.bootstrap_room)\n"
        "            if (\n"
        "                status == KVPoll.WaitingForInput\n"
        "                and self.kv_mgr.decode_l3_reverse_is_pending(\n"
        "                    self.bootstrap_room\n"
        "                )\n"
        "                and not self.kv_mgr.decode_l3_reverse_is_done(\n"
        "                    self.bootstrap_room\n"
        "                )\n"
        "            ):\n"
        "                return KVPoll.Bootstrapping\n"
        "            if status in (KVPoll.Success, KVPoll.Failed):\n",
        "keep Prefill request bootstrapping during reverse transfer",
    )
    source = replace_once(
        source,
        "    def clear(self) -> None:\n"
        "        if self.bootstrap_room in self.kv_mgr.request_status:\n"
        "            self.kv_mgr.request_status.pop(self.bootstrap_room)\n",
        "    def clear(self) -> None:\n"
        "        self.kv_mgr.clear_decode_l3_reverse(self.bootstrap_room)\n"
        "        if self.bootstrap_room in self.kv_mgr.request_status:\n"
        "            self.kv_mgr.request_status.pop(self.bootstrap_room)\n",
        "clear Prefill reverse-transfer state",
    )
    source = replace_once(
        source,
        "        self.host_staging_h2d_start = None\n"
        "        self.host_staging_h2d_end = None\n"
        "        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)\n",
        "        self.host_staging_h2d_start = None\n"
        "        self.host_staging_h2d_end = None\n"
        "        self.host_staging_page_indices = np.array([], dtype=np.int32)\n"
        "        self.decode_l3_hit_tokens = 0\n"
        "        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)\n",
        "initialize Decode L3 receiver state",
    )
    source = replace_once(
        source,
        "        state_indices: Optional[List[int]] = None,\n"
        "    ):\n"
        "        if self.kv_mgr.decode_host_staging_enabled:\n"
        "            self.kv_indices = np.array(kv_indices, dtype=np.int64, copy=True)\n"
        "            (\n"
        "                self.host_staging_token_indices,\n"
        "                kv_indices,\n"
        "            ) = self.kv_mgr.allocate_decode_host_staging(self.kv_indices)\n",
        "        state_indices: Optional[List[int]] = None,\n"
        "        token_ids: Optional[List[int]] = None,\n"
        "        request_id: Optional[str] = None,\n"
        "    ):\n"
        "        if self.kv_mgr.decode_host_staging_enabled:\n"
        "            self.kv_indices = np.array(kv_indices, dtype=np.int64, copy=True)\n"
        "            (\n"
        "                self.host_staging_token_indices,\n"
        "                kv_indices,\n"
        "            ) = self.kv_mgr.allocate_decode_host_staging(self.kv_indices)\n"
        "            self.host_staging_page_indices = np.array(\n"
        "                kv_indices, dtype=np.int32, copy=True\n"
        "            )\n"
        "            if token_ids is not None and request_id is not None:\n"
        "                self.decode_l3_hit_tokens = self.kv_mgr.load_decode_l3_prefix(\n"
        "                    request_id, token_ids, self.host_staging_token_indices\n"
        "                )\n"
        "                self.kv_mgr.register_decode_l3_receiver(\n"
        "                    self.bootstrap_room, self\n"
        "                )\n",
        "load Decode L3 prefix during receiver initialization",
    )
    source = replace_once(
        source,
        "                        str(self.required_dst_info_num).encode(\"ascii\"),\n"
        "                    ]\n",
        "                        str(self.required_dst_info_num).encode(\"ascii\"),\n"
        "                        str(self.decode_l3_hit_tokens).encode(\"ascii\"),\n"
        "                    ]\n",
        "send Decode L3 hit tokens to Prefill",
    )
    source = replace_once(
        source,
        "    def clear(self) -> None:\n"
        "        if self.host_staging_token_indices is not None:\n",
        "    def clear(self) -> None:\n"
        "        self.kv_mgr.unregister_decode_l3_receiver(self.bootstrap_room, self)\n"
        "        if self.host_staging_token_indices is not None:\n",
        "unregister Decode L3 receiver",
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
    patch_decode_l3_scheduler(srt / "managers/scheduler.py")
    patch_decode_l3_schedule_policy(srt / "managers/schedule_policy.py")
    patch_decode_l3_runtime_checker(
        srt / "managers/scheduler_runtime_checker_mixin.py"
    )
    patch_schedule_batch(srt / "managers/schedule_batch.py")
    patch_prefill_prepare_breakdown(
        srt / "managers/scheduler.py", srt / "managers/schedule_batch.py"
    )
    patch_mooncake_bounded_transfer(
        srt / "disaggregation/mooncake/conn.py"
    )
    patch_prefill_transfer(srt / "disaggregation/prefill.py")
    patch_prefill_decode_l3_reverse(srt / "disaggregation/prefill.py")
    patch_decode_receive(srt / "disaggregation/decode.py")
    patch_bounded_decode_host_staging_pool(
        srt / "mem_cache/memory_pool_host.py"
    )
    patch_decode_host_staging(srt / "disaggregation/decode.py")
    patch_decode_l3_read(srt / "disaggregation/decode.py")
    patch_decode_handoff_breakdown(srt / "disaggregation/decode.py")
    patch_decode_prealloc_inner_breakdown(srt / "disaggregation/decode.py")
    patch_decode_allocator_merge_breakdown(srt / "disaggregation/decode.py")
    patch_allocator_kernel_breakdown(srt / "mem_cache/allocator.py")
    patch_mooncake_decode_host_staging(
        srt / "disaggregation/mooncake/conn.py"
    )
    patch_mooncake_decode_l3_reverse(
        srt / "disaggregation/mooncake/conn.py"
    )
    patch_decode_receiver_breakdown(
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
