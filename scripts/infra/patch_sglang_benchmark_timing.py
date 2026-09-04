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
        "        self.ongoing_prefetch = {}\n"
        "        self.ongoing_backup = {}\n",
        "        self.ongoing_prefetch = {}\n"
        "        self.dualpath_prefetch_start = {}\n"
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
        "        logger.debug(f\"Prefetch {req_id} completed with {completed_tokens} tokens\")\n",
        "        logger.debug(f\"Prefetch {req_id} completed with {completed_tokens} tokens\")\n"
        "        dualpath_start = self.dualpath_prefetch_start.pop(req_id, None)\n"
        "        if dualpath_start is not None:\n"
        "            dualpath_end = time.perf_counter()\n"
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
    patch_first_token_dispatch(
        srt / "managers/scheduler_output_processor_mixin.py"
    )
    patch_http_yield(srt / "entrypoints/http_server.py")


if __name__ == "__main__":
    main()
