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


def replace_function(source: str, signature: str, body: str, label: str) -> str:
    start = source.find(signature)
    if start < 0:
        raise RuntimeError(f"{label}: function signature not found")
    indent = signature[: len(signature) - len(signature.lstrip())]
    end = source.find(f"\n{indent}def ", start + len(signature))
    if end < 0:
        end = len(source)
    return source[:start] + body.rstrip() + "\n" + source[end:]


def write_read_path_helper(path: Path) -> None:
    path.write_text(
        '''"""Request-scoped DualPath L3 read-path decision.

`dynamic` lets each request pick Prefill or Decode as its L3 reader. Both
engines must agree, and neither may block waiting for the decision: Prefill
resolves it inside `_prefetch_kvcache()`, which runs on the scheduler event
loop before the PD bootstrap handshake even exists.

The decision is therefore derived, not negotiated. The benchmark client owns
the request id and encodes `(trajectory, turn)` in it; Router hands the same
id to both engines, and `Scheduler.recv_requests()` broadcasts the request to
every TP rank. A pure function of the rid is consequently identical on both
sides and on every rank, with no coordinator, no RPC and no timeout.
"""

from __future__ import annotations

import os
import re
import zlib
from functools import lru_cache

CONFIGURED = os.environ.get("DUALPATH_L3_READ_PATH", "prefill")
POLICY = os.environ.get("DUALPATH_DYNAMIC_PATH_POLICY", "alternate_per_turn")
IS_DYNAMIC = CONFIGURED == "dynamic"

# Requests without a trajectory id (warmup, health checks) must not fail.
FALLBACK = "prefill"

_RID = re.compile(r"([^-]+)-(\\d+)-turn-(\\d+)")


def _phase(prefix: str, index: str) -> int:
    """Per-trajectory offset so they do not all switch on the same turn.

    Trajectory ids are numbered (32k-0000, 32k-0001, ...), so the index
    parity spreads neighbours to opposite phases exactly. Hashing the whole
    id does not: crc32 of 32k-0000..32k-0003 happens to share one parity bit,
    which would put every trajectory in lockstep.
    """
    try:
        return int(index) & 1
    except ValueError:
        return zlib.crc32(f"{prefix}-{index}".encode()) & 1


@lru_cache(maxsize=8192)
def resolve(rid: str) -> str:
    """Return "prefill" or "decode" for this request. Never raises."""
    if not IS_DYNAMIC:
        return CONFIGURED
    match = _RID.search(rid or "")
    if match is None:
        return FALLBACK
    turn = int(match.group(3))
    if POLICY == "same_turn_lockstep":
        # Diagnostic: every trajectory switches on the same turn. Reproduces
        # path/agent aliasing on purpose; not for measurement.
        phase = 0
    else:
        phase = _phase(match.group(1), match.group(2))
    return "prefill" if (turn + phase) % 2 == 0 else "decode"


def is_decode_read(rid: str) -> bool:
    return resolve(rid) == "decode"


def decode_capable() -> bool:
    """Whether Decode-side L3 machinery must be constructed at startup."""
    return CONFIGURED in ("decode", "dynamic")
'''
    )
    print(f"Wrote: {path}")


def write_ttft_v3_helper(path: Path) -> None:
    path.write_text(
        '''"""Low-overhead request-scoped DualPath TTFT timing helpers."""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from typing import Any

from sglang.srt.utils.dualpath_read_path import resolve as resolve_read_path


SCHEMA_VERSION = 3
ENABLED = os.environ.get("DUALPATH_TTFT_TIMING", "0").lower() in (
    "1", "true", "yes", "on"
)
DETAIL = os.environ.get("DUALPATH_TTFT_TIMING_DETAIL", "0").lower() in (
    "1", "true", "yes", "on"
)
CONFIGURED_PATH = os.environ.get("DUALPATH_L3_READ_PATH", "prefill")
NODE = socket.gethostname().split(".", 1)[0]
PREFILL_ENGINE = os.environ.get("PREFILL_ID", "0")
DECODE_ENGINE = os.environ.get("DECODE_ID", "0")
PREFILL_NODE = os.environ.get("PREFILL_HOST", NODE)
DECODE_NODE = os.environ.get("DECODE_HOST", NODE)

STAGES = (
    "prefill_before_lustre_read",
    "prefill_lustre_read",
    "prefill_scatter",
    "prefill_host_to_device",
    "prefill_computation",
    "p_to_d_transfer",
    "decode_preallocation",
    "decode_before_lustre_read",
    "decode_lustre_read",
    "decode_scatter",
    "d_to_p_transfer",
    "decode_host_to_device",
    "decode_dispatch",
    # Time the request spent queued behind other requests. The engine reports
    # the raw window; the summarizer subtracts the part that overlaps an
    # already-measured stage, since a queued request's own L3 read runs while
    # it is still sitting in that queue.
    "queue_wait",
)


def now_ns() -> int:
    return time.perf_counter_ns()


def ns_seconds(value: int | float | None) -> float:
    return max(0.0, float(value or 0) / 1_000_000_000.0)


def queue_window_ns(stats, start_attr: str, end_attr: str) -> tuple[int, int]:
    """Reuse the scheduler's existing queue stamps as a timing window.

    RequestTimeStats records time.perf_counter(); now_ns() is
    time.perf_counter_ns(), so the two are directly comparable and no extra
    instrumentation point is needed.
    """
    start = float(getattr(stats, start_attr, 0.0) or 0.0)
    end = float(getattr(stats, end_attr, 0.0) or 0.0)
    if start <= 0.0 or end <= start:
        return 0, 0
    return int(start * 1_000_000_000), int(end * 1_000_000_000)


def new_cuda_event(device_module):
    if not ENABLED:
        return None
    return device_module.Event(enable_timing=True)


def event_seconds(start, end) -> float:
    if start is None or end is None or not end.query():
        return 0.0
    return max(0.0, float(start.elapsed_time(end)) / 1000.0)


def event_overlap_seconds(first_start, first_end, second_start, second_end) -> float:
    if any(item is None for item in (first_start, first_end, second_start, second_end)):
        return 0.0
    if not first_end.query() or not second_end.query():
        return 0.0
    first_duration = event_seconds(first_start, first_end)
    second_duration = event_seconds(second_start, second_end)
    if first_duration <= 0 or second_duration <= 0:
        return 0.0
    second_offset = float(first_start.elapsed_time(second_start)) / 1000.0
    return max(
        0.0,
        min(first_duration, second_offset + second_duration)
        - max(0.0, second_offset),
    )


def emit(
    logger: logging.Logger,
    *,
    rid: str,
    side: str,
    tp_rank: int,
    values: dict[str, Any] | None = None,
) -> None:
    if not ENABLED:
        return
    values = dict(values or {})
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rid": rid,
        # Under `dynamic` these differ: configured_path is the run mode,
        # path is what this request actually did.
        "configured_path": CONFIGURED_PATH,
        "path": resolve_read_path(rid),
        "side": side,
        "engine": PREFILL_ENGINE if side == "prefill" else DECODE_ENGINE,
        "node": PREFILL_NODE if side == "prefill" else DECODE_NODE,
        "tp_rank": int(tp_rank),
        "clock_scope": "process_monotonic",
    }
    for stage in STAGES:
        raw = max(0.0, float(values.pop(f"{stage}_raw_seconds", 0.0)))
        critical = max(0.0, float(values.pop(f"{stage}_seconds", raw)))
        payload[f"{stage}_raw_seconds"] = raw
        payload[f"{stage}_seconds"] = critical
        payload[f"{stage}_overlap_removed_seconds"] = max(
            0.0,
            float(values.pop(f"{stage}_overlap_removed_seconds", raw - critical)),
        )
    payload.update(values)
    logger.info("DUALPATH_TTFT_V3 %s", json.dumps(payload, sort_keys=True))
'''
    )
    print(f"Wrote: {path}")


def gate_logger_call(source: str, marker: str) -> str:
    """Gate the logger call containing marker behind the cached detail flag."""
    search_from = 0
    while True:
        marker_pos = source.find(marker, search_from)
        if marker_pos < 0:
            return source
        logger_pos = source.rfind("logger.", 0, marker_pos)
        if logger_pos < 0:
            raise RuntimeError(f"Cannot locate logger call for {marker}")
        line_start = source.rfind("\n", 0, logger_pos) + 1
        indent = source[line_start:logger_pos]
        close = source.find(f"\n{indent})\n", marker_pos)
        if close < 0:
            raise RuntimeError(f"Cannot locate logger call end for {marker}")
        call_end = close + len(f"\n{indent})\n")
        block = source[line_start:call_end]
        if source[max(0, line_start - len(indent) - 40):line_start].endswith(
            f"{indent}if DUALPATH_TTFT_DETAIL:\n"
        ):
            search_from = call_end
            continue
        gated = (
            f"{indent}if DUALPATH_TTFT_DETAIL:\n"
            + "".join(
                f"    {line}" if line.strip() else line
                for line in block.splitlines(keepends=True)
            )
        )
        source = source[:line_start] + gated + source[call_end:]
        search_from = line_start + len(gated)


def patch_hicache_file_o_direct_read(path: Path) -> None:
    source = path.read_text()
    marker = "DUAL_PATH_O_DIRECT_READ"
    if marker in source:
        print(f"Already patched: {path}")
        return

    init_anchor = (
        "        self.file_path = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get() or file_path\n"
        if "self.file_path = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get() or file_path\n"
        in source
        else "        self.file_path = os.getenv(\"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR\", file_path)\n"
    )
    source = replace_once(
        source,
        init_anchor,
        init_anchor +
        "        extra_config = storage_config.extra_config or {}\n"
        "        if isinstance(extra_config, str):\n"
        "            extra_config = __import__(\"json\").loads(extra_config)\n"
        "        if not isinstance(extra_config, dict):\n"
        "            raise TypeError(\"HiCache extra_config must be a dict or JSON object\")\n"
        "        direct_read_raw = extra_config.get(\"dual_path_o_direct_read\")\n"
        "        if direct_read_raw is None:\n"
        "            direct_read_raw = __import__(\"os\").environ.get(\n"
        "                \"DUAL_PATH_O_DIRECT_READ\", \"0\"\n"
        "            )\n"
        "        self.dualpath_o_direct_read = str(direct_read_raw).lower() in (\n"
        "            \"1\", \"true\", \"yes\", \"on\"\n"
        "        )\n"
        "        self.dualpath_o_direct_alignment = int(\n"
        "            extra_config.get(\"dual_path_o_direct_alignment\", 4096)\n"
        "        )\n"
        "        if self.dualpath_o_direct_alignment <= 0 or (\n"
        "            self.dualpath_o_direct_alignment\n"
        "            & (self.dualpath_o_direct_alignment - 1)\n"
        "        ):\n"
        "            raise ValueError(\n"
        "                \"dual_path_o_direct_alignment must be a positive power of two\"\n"
        "            )\n",
        "configure optional DualPath O_DIRECT file reads",
    )
    key_name = "suffixed" if "f\"{suffixed}.bin\"" in source else "key"
    old_get = (
        "            expected = target_location.numel() * target_location.element_size()\n"
        "            with open(tensor_path, \"rb\", buffering=0) as f:\n"
        "                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())\n"
        "                if f.readinto(buf) != expected:\n"
        f"                    raise IOError(f\"Short read for {{{key_name}}}\")\n"
    )
    direct_error_key = key_name
    new_get = (
        "            expected = target_location.numel() * target_location.element_size()\n"
        "            if self.dualpath_o_direct_read:\n"
        "                if not target_location.is_contiguous():\n"
        "                    raise ValueError(\n"
        "                        \"O_DIRECT HiCache target must be contiguous\"\n"
        "                    )\n"
        "                ctypes = __import__(\"ctypes\")\n"
        "                mmap = __import__(\"mmap\")\n"
        "                alignment = self.dualpath_o_direct_alignment\n"
        "                if expected % alignment:\n"
        "                    raise ValueError(\n"
        "                        f\"O_DIRECT read length {expected} is not aligned to {alignment}\"\n"
        "                    )\n"
        "                buf = memoryview(target_location.view(torch.uint8).numpy())\n"
        "                address = ctypes.addressof(ctypes.c_char.from_buffer(buf))\n"
        "                scratch = None\n"
        "                if address % alignment:\n"
        "                    scratch = mmap.mmap(-1, expected + alignment)\n"
        "                    scratch_address = ctypes.addressof(\n"
        "                        ctypes.c_char.from_buffer(scratch)\n"
        "                    )\n"
        "                    offset = (-scratch_address) % alignment\n"
        "                    io_buf = memoryview(scratch)[offset : offset + expected]\n"
        "                else:\n"
        "                    io_buf = buf\n"
        "                try:\n"
        "                    flags = __import__(\"os\").O_RDONLY | __import__(\"os\").O_DIRECT\n"
        "                    fd = __import__(\"os\").open(tensor_path, flags)\n"
        "                    try:\n"
        "                        offset = 0\n"
        "                        while offset < expected:\n"
        f"                            count = __import__(\"os\").readv(fd, [io_buf[offset:]])\n"
        "                            if count == 0:\n"
        f"                                raise IOError(f\"Short read for {{{direct_error_key}}}\")\n"
        "                            offset += count\n"
        "                    finally:\n"
        "                        __import__(\"os\").close(fd)\n"
        "                    if scratch is not None:\n"
        "                        buf[:] = io_buf\n"
        "                finally:\n"
        "                    if scratch is not None:\n"
        "                        io_buf.release()\n"
        "                        scratch.close()\n"
        "            else:\n"
        "                with open(tensor_path, \"rb\", buffering=0) as f:\n"
        "                    buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())\n"
        f"                    if f.readinto(buf) != expected:\n"
        f"                        raise IOError(f\"Short read for {{{direct_error_key}}}\")\n"
    )
    source = replace_once(
        source,
        old_get,
        new_get,
        "add optional O_DIRECT read while preserving buffered default",
    )
    source = source.replace(
        "class HiCacheFile(HiCacheStorage):\n",
        "class HiCacheFile(HiCacheStorage):\n"
        "    # DUAL_PATH_O_DIRECT_READ: optional Lustre client page-cache bypass.\n",
        1,
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_prefill_o_direct_reusable_buffer(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_O_DIRECT_REUSABLE_PAGE"
    if marker in source:
        print(f"Already patched: {path}")
        return

    old = (
        "    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):\n"
        "        dummy_page_dst = [\n"
        "            self.mem_pool_host.get_dummy_flat_data_page() for _ in hash_values\n"
        "        ]\n"
        "        page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)\n"
        "        if page_data is None:\n"
        "            return\n"
        "        for i in range(len(hash_values)):\n"
    )
    new = (
        "    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):\n"
        "        # DUALPATH_O_DIRECT_REUSABLE_PAGE: file O_DIRECT reads are\n"
        "        # synchronous, so one aligned pinned page can be scattered to\n"
        "        # the Host pool and reused immediately. This avoids allocating\n"
        "        # and zeroing up to storage_batch_size temporary pages.\n"
        "        dualpath_lustre_read_s = 0.0\n"
        "        dualpath_scatter_s = 0.0\n"
        "        dualpath_pages = 0\n"
        "        if operation.dualpath_read_start is None:\n"
        "            operation.dualpath_read_start = __import__(\"time\").perf_counter()\n"
        "        dualpath_log_prefix = (\n"
        "            \"DUALPATH_DECODE_KV_READ_BREAKDOWN\"\n"
        "            if operation.retain_host_indices\n"
        "            else \"DUALPATH_PREFILL_KV_READ_BREAKDOWN\"\n"
        "        )\n"
        "        if getattr(self.storage_backend, \"dualpath_o_direct_read\", False):\n"
        "            data_page = getattr(\n"
        "                self, \"_dualpath_o_direct_page_buffer\", None\n"
        "            )\n"
        "            if data_page is None:\n"
        "                data_page = self.mem_pool_host.get_dummy_flat_data_page()\n"
        "                alignment = getattr(\n"
        "                    self.storage_backend,\n"
        "                    \"dualpath_o_direct_alignment\",\n"
        "                    4096,\n"
        "                )\n"
        "                if data_page.data_ptr() % alignment:\n"
        "                    element_size = data_page.element_size()\n"
        "                    extra_elements = (\n"
        "                        alignment + element_size - 1\n"
        "                    ) // element_size\n"
        "                    owner = torch.empty(\n"
        "                        data_page.numel() + extra_elements,\n"
        "                        dtype=data_page.dtype,\n"
        "                        device=data_page.device,\n"
        "                        pin_memory=data_page.is_pinned(),\n"
        "                    )\n"
        "                    offset_bytes = (-owner.data_ptr()) % alignment\n"
        "                    if offset_bytes % element_size:\n"
        "                        raise RuntimeError(\n"
        "                            \"Cannot align O_DIRECT Prefill page buffer\"\n"
        "                        )\n"
        "                    offset = offset_bytes // element_size\n"
        "                    data_page = owner[\n"
        "                        offset : offset + data_page.numel()\n"
        "                    ]\n"
        "                    self._dualpath_o_direct_page_buffer_owner = owner\n"
        "                self._dualpath_o_direct_page_buffer = data_page\n"
        "\n"
        "            for i, hash_value in enumerate(hash_values):\n"
        "                dualpath_io_start = __import__(\"time\").perf_counter()\n"
        "                page_data = self.storage_backend.get(hash_value, data_page)\n"
        "                dualpath_lustre_read_s += (\n"
        "                    __import__(\"time\").perf_counter() - dualpath_io_start\n"
        "                )\n"
        "                if page_data is None:\n"
        "                    logger.warning(\n"
        "                        f\"Prefetch operation {operation.request_id} failed \"\n"
        "                        f\"to retrieve page {hash_value}.\"\n"
        "                    )\n"
        "                    break\n"
        "                dualpath_scatter_start = __import__(\"time\").perf_counter()\n"
        "                self.mem_pool_host.set_from_flat_data_page(\n"
        "                    host_indices[i * self.page_size],\n"
        "                    page_data,\n"
        "                )\n"
        "                dualpath_scatter_s += (\n"
        "                    __import__(\"time\").perf_counter()\n"
        "                    - dualpath_scatter_start\n"
        "                )\n"
        "                dualpath_pages += 1\n"
        "                if not operation.increment(self.page_size):\n"
        "                    break\n"
        "            operation.dualpath_lustre_read_s += dualpath_lustre_read_s\n"
        "            operation.dualpath_scatter_s += dualpath_scatter_s\n"
        "            operation.dualpath_read_end = __import__(\"time\").perf_counter()\n"
        "            logger.info(\n"
        "                \"%s rid=%s \"\n"
        "                \"prefill_engine=%s node=%s tp_rank=%d \"\n"
        "                \"lustre_read_s=%.9f scatter_s=%.9f pages=%d\",\n"
        "                dualpath_log_prefix,\n"
        "                operation.request_id,\n"
        "                __import__(\"os\").environ.get(\"PREFILL_ID\", \"0\"),\n"
        "                __import__(\"os\").environ.get(\n"
        "                    \"PREFILL_HOST\",\n"
        "                    __import__(\"os\").environ.get(\"HOSTNAME\", \"unknown\"),\n"
        "                ),\n"
        "                self.tp_rank,\n"
        "                dualpath_lustre_read_s,\n"
        "                dualpath_scatter_s,\n"
        "                dualpath_pages,\n"
        "            )\n"
        "            return\n"
        "\n"
        "        dummy_page_dst = [\n"
        "            self.mem_pool_host.get_dummy_flat_data_page() for _ in hash_values\n"
        "        ]\n"
        "        dualpath_io_start = __import__(\"time\").perf_counter()\n"
        "        page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)\n"
        "        dualpath_lustre_read_s = (\n"
        "            __import__(\"time\").perf_counter() - dualpath_io_start\n"
        "        )\n"
        "        if page_data is None:\n"
        "            operation.dualpath_lustre_read_s += dualpath_lustre_read_s\n"
        "            operation.dualpath_read_end = __import__(\"time\").perf_counter()\n"
        "            logger.info(\n"
        "                \"%s rid=%s \"\n"
        "                \"prefill_engine=%s node=%s tp_rank=%d \"\n"
        "                \"lustre_read_s=%.9f scatter_s=0.000000000 pages=0\",\n"
        "                dualpath_log_prefix,\n"
        "                operation.request_id,\n"
        "                __import__(\"os\").environ.get(\"PREFILL_ID\", \"0\"),\n"
        "                __import__(\"os\").environ.get(\n"
        "                    \"PREFILL_HOST\",\n"
        "                    __import__(\"os\").environ.get(\"HOSTNAME\", \"unknown\"),\n"
        "                ),\n"
        "                self.tp_rank,\n"
        "                dualpath_lustre_read_s,\n"
        "            )\n"
        "            return\n"
        "        for i in range(len(hash_values)):\n"
    )
    source = replace_once(
        source,
        old,
        new,
        "reuse one aligned Prefill page for O_DIRECT reads",
    )
    function_start = source.index(
        "    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):\n"
    )
    function_end = source.find("\n    def ", function_start + 1)
    if function_end < 0:
        function_end = len(source)
    batch_get = source.index(
        "self.storage_backend.batch_get(hash_values, dummy_page_dst)",
        function_start,
        function_end,
    )
    scatter_start = source.index(
        "self.mem_pool_host.set_from_flat_data_page(",
        batch_get,
        function_end,
    )
    scatter_line_start = source.rfind("\n", function_start, scatter_start) + 1
    scatter_indent = source[scatter_line_start:scatter_start]
    open_paren = source.index("(", scatter_start, function_end)
    depth = 0
    scatter_end = None
    for position in range(open_paren, function_end):
        character = source[position]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                scatter_end = position + 1
                break
    if scatter_end is None:
        raise RuntimeError("Cannot locate buffered Prefill scatter call end")
    if scatter_end < len(source) and source[scatter_end] == "\n":
        scatter_end += 1
    source = (
        source[:scatter_line_start]
        + scatter_indent
        + "dualpath_scatter_start = __import__(\"time\").perf_counter()\n"
        + source[scatter_line_start:scatter_end]
        + scatter_indent
        + "dualpath_scatter_s += (\n"
        + scatter_indent
        + "    __import__(\"time\").perf_counter() - dualpath_scatter_start\n"
        + scatter_indent
        + ")\n"
        + scatter_indent
        + "dualpath_pages += 1\n"
        + source[scatter_end:]
    )
    function_end = source.find("\n    def ", function_start + 1)
    if function_end < 0:
        function_end = len(source)
    # The buffered implementation in the base SGLang version has no log
    # statement after the page loop. Add the aggregate timing immediately
    # before the function ends, after all pages have been scattered.
    source = (
        source[:function_end]
        + "\n        operation.dualpath_lustre_read_s += dualpath_lustre_read_s\n"
        + "        operation.dualpath_scatter_s += dualpath_scatter_s\n"
        + "        operation.dualpath_read_end = __import__(\"time\").perf_counter()\n"
        + source[function_end:]
    )
    function_end = source.find("\n    def ", function_start + 1)
    if function_end < 0:
        function_end = len(source)
    buffered_log = (
        "        logger.info(\n"
        "            \"%s rid=%s \"\n"
        "            \"prefill_engine=%s node=%s tp_rank=%d \"\n"
        "            \"lustre_read_s=%.9f scatter_s=%.9f pages=%d\",\n"
        "            dualpath_log_prefix,\n"
        "            operation.request_id,\n"
        "            __import__(\"os\").environ.get(\"PREFILL_ID\", \"0\"),\n"
        "            __import__(\"os\").environ.get(\n"
        "                \"PREFILL_HOST\",\n"
        "                __import__(\"os\").environ.get(\"HOSTNAME\", \"unknown\"),\n"
        "            ),\n"
        "            self.tp_rank,\n"
        "            dualpath_lustre_read_s,\n"
        "            dualpath_scatter_s,\n"
        "            dualpath_pages,\n"
        "        )\n"
    )
    source = source[:function_end] + "\n" + buffered_log + source[function_end:]
    path.write_text(source)
    print(f"Patched: {path}")


def patch_shared_prefetch_progress(path: Path) -> None:
    """Expose the same async prefetch completion protocol to Decode.

    Prefill owns the radix-tree bookkeeping around HiCacheController. Decode
    intentionally has no radix tree for the L3 staging pages, but it must use
    the exact same storage query/read worker and TP consensus. These helpers
    keep that protocol in the shared controller instead of reimplementing it
    in the Mooncake receiver.
    """
    source = path.read_text()
    marker = "DUALPATH_SHARED_PREFETCH_PROGRESS"
    if marker in source:
        if "DUALPATH_L3_HIT_QUERY" not in source:
            source = replace_once(
                source,
                "                hash_value, storage_hit_count = self._storage_hit_query(operation)\n"
                "                if self.tp_world_size > 1:\n",
                "                hash_value, storage_hit_count = self._storage_hit_query(operation)\n"
                "                if __import__(\"os\").environ.get(\n"
                "                    \"DUALPATH_L3_HIT_DIAGNOSTIC\", \"1\"\n"
                "                ) == \"1\":\n"
                "                    logger.info(\n"
                "                        \"DUALPATH_L3_HIT_QUERY rid=%s tp_rank=%d \"\n"
                "                        \"requested_pages=%d local_hit_tokens=%d first_hash=%s\",\n"
                "                        operation.request_id, self.tp_rank,\n"
                "                        len(hash_value), storage_hit_count,\n"
                "                        hash_value[0] if hash_value else \"none\",\n"
                "                    )\n"
                "                if self.tp_world_size > 1:\n",
                "add L3 hit diagnostics to an existing shared prefetch patch",
            )
            source = replace_once(
                source,
                "                    storage_hit_count = storage_hit_count_tensor.item()\n"
                "                operation.query_complete = True\n",
                "                    storage_hit_count = storage_hit_count_tensor.item()\n"
                "                if __import__(\"os\").environ.get(\n"
                "                    \"DUALPATH_L3_HIT_DIAGNOSTIC\", \"1\"\n"
                "                ) == \"1\":\n"
                "                    logger.info(\n"
                "                        \"DUALPATH_L3_HIT_CONSENSUS rid=%s tp_rank=%d \"\n"
                "                        \"hit_tokens=%d threshold=%d\",\n"
                "                        operation.request_id, self.tp_rank,\n"
                "                        storage_hit_count, self.prefetch_threshold,\n"
                "                    )\n"
                "                operation.query_complete = True\n",
                "add L3 consensus diagnostics to an existing shared prefetch patch",
            )
            path.write_text(source)
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "        self._terminated_flag = False\n"
        "        self.start_time = time.monotonic()\n",
        "        self._terminated_flag = False\n"
        "        self.start_time = time.monotonic()\n"
        "        # DUALPATH_SHARED_PREFETCH_PROGRESS: set by the shared\n"
        "        # prefetch worker after prefix hash lookup and TP hit\n"
        "        # consensus complete.\n"
        "        self.query_complete = False\n"
        "        self.revoked = False\n"
        "        # Set after the auxiliary I/O worker has stopped using the\n"
        "        # operation's Host buffer.\n"
        "        self.io_complete = threading.Event()\n",
        "track shared prefetch query completion",
    )
    source = replace_once(
        source,
        "        self.io_complete = threading.Event()\n",
        "        self.io_complete = threading.Event()\n"
        "        # Decode retains every page in this allocation because\n"
        "        # Lustre misses remain P->D destinations.\n"
        "        self.retain_host_indices = False\n",
        "retain Decode staging pages through shared prefetch",
    )
    source = replace_once(
        source,
        "        self.retain_host_indices = False\n",
        "        self.retain_host_indices = False\n"
        "        self.dualpath_read_start = None\n"
        "        self.dualpath_read_end = None\n"
        "        self.dualpath_lustre_read_s = 0.0\n"
        "        self.dualpath_scatter_s = 0.0\n",
        "track aggregate shared prefetch read timing",
    )
    source = replace_once(
        source,
        "        prefix_keys: Optional[List[str]] = None,\n"
        "    ) -> PrefetchOperation:\n",
        "        prefix_keys: Optional[List[str]] = None,\n"
        "        retain_host_indices: bool = False,\n"
        "    ) -> PrefetchOperation:\n",
        "allow Decode to retain shared prefetch allocation",
    )
    source = replace_once(
        source,
        "        operation = PrefetchOperation(\n"
        "            request_id, host_indices, new_input_tokens, last_hash, prefix_keys\n"
        "        )\n"
        "        self.prefetch_queue.put(operation)\n",
        "        operation = PrefetchOperation(\n"
        "            request_id, host_indices, new_input_tokens, last_hash, prefix_keys\n"
        "        )\n"
        "        operation.retain_host_indices = retain_host_indices\n"
        "        self.prefetch_queue.put(operation)\n",
        "set shared prefetch allocation ownership before queueing",
    )

    source = replace_once(
        source,
        "                self.prefetch_tp_group = create_custom_parallel_group(\n"
        "                    group_ranks=group_ranks, backend=\"gloo\"\n"
        "                )\n",
        "                self.prefetch_tp_group = create_custom_parallel_group(\n"
        "                    group_ranks=group_ranks, backend=\"gloo\"\n"
        "                )\n"
        "                # Keep scheduler-side progress collectives isolated from\n"
        "                # the prefetch worker's hit-query collectives. The two\n"
        "                # threads can otherwise issue different collective orders\n"
        "                # on different TP ranks and deadlock.\n"
        "                self.prefetch_progress_tp_group = (\n"
        "                    create_custom_parallel_group(\n"
        "                        group_ranks=group_ranks, backend=\"gloo\"\n"
        "                    )\n"
        "                )\n",
        "isolate shared prefetch progress collectives",
    )
    source = replace_once(
        source,
        "    def terminate_prefetch(self, operation):\n"
        "        operation.mark_terminate()\n"
        "        return operation.completed_tokens, operation.hash_value\n",
        "    def terminate_prefetch(self, operation):\n"
        "        operation.mark_terminate()\n"
        "        return operation.completed_tokens, operation.hash_value\n"
        "\n"
        "    def shared_prefetch_can_terminate(\n"
        "        self, operation, stop_policy=\"wait_complete\",\n"
        "        timeout_base=None, timeout_per_page=0.0\n"
        "    ) -> bool:\n"
        "        \"\"\"Use Prefill's async prefetch completion consensus.\"\"\"\n"
        "        has_host_indices = operation.host_indices is not None\n"
        "        query_complete = (\n"
        "            not has_host_indices or operation.query_complete\n"
        "        )\n"
        "        if (\n"
        "            not has_host_indices\n"
        "            or not query_complete\n"
        "            or len(operation.hash_value) == 0\n"
        "        ):\n"
        "            completed = False\n"
        "        else:\n"
        "            completed = (\n"
        "                operation.completed_tokens\n"
        "                == len(operation.hash_value) * self.page_size\n"
        "            )\n"
        "        if not has_host_indices:\n"
        "            can_terminate = True\n"
        "        elif stop_policy == \"best_effort\":\n"
        "            can_terminate = query_complete\n"
        "        elif stop_policy == \"wait_complete\":\n"
        "            can_terminate = completed\n"
        "        elif stop_policy == \"timeout\":\n"
        "            can_terminate = query_complete and (\n"
        "                completed\n"
        "                or (\n"
        "                    timeout_base is not None\n"
        "                    and time.monotonic() - operation.start_time\n"
        "                    > timeout_base\n"
        "                    + len(operation.hash_value) * timeout_per_page\n"
        "                )\n"
        "            )\n"
        "        else:\n"
        "            can_terminate = query_complete\n"
        "        operation_terminated = operation.is_terminated()\n"
        "        if self.tp_world_size > 1:\n"
        "            states = torch.tensor(\n"
        "                [\n"
        "                    1 - int(has_host_indices),\n"
        "                    1 - int(query_complete),\n"
        "                    1 - int(can_terminate),\n"
        "                    int(operation_terminated),\n"
        "                ],\n"
        "                dtype=torch.int,\n"
        "            )\n"
        "            torch.distributed.all_reduce(\n"
        "                states,\n"
        "                op=torch.distributed.ReduceOp.MAX,\n"
        "                group=self.prefetch_progress_tp_group,\n"
        "            )\n"
        "            missing_host_indices = states[0].item() == 1\n"
        "            query_complete = (\n"
        "                missing_host_indices or states[1].item() == 0\n"
        "            )\n"
        "            can_terminate = (\n"
        "                missing_host_indices\n"
        "                or (query_complete and states[2].item() == 0)\n"
        "            )\n"
        "            operation_terminated = (\n"
        "                missing_host_indices\n"
        "                or (query_complete and states[3].item() == 1)\n"
        "            )\n"
        "        return can_terminate or operation_terminated\n"
        "\n"
        "    def shared_prefetch_finish(self, operation):\n"
        "        \"\"\"Finish a shared prefetch with Prefill's MIN consensus.\"\"\"\n"
        "        completed_tokens, hash_value = self.terminate_prefetch(operation)\n"
        "        if self.tp_world_size > 1:\n"
        "            completed_tensor = torch.tensor(\n"
        "                completed_tokens, dtype=torch.int\n"
        "            )\n"
        "            torch.distributed.all_reduce(\n"
        "                completed_tensor,\n"
        "                op=torch.distributed.ReduceOp.MIN,\n"
        "                group=self.prefetch_progress_tp_group,\n"
        "            )\n"
        "            completed_tokens = int(completed_tensor.item())\n"
        "        return completed_tokens, hash_value\n"
        "\n"
        "    def shared_prefetch_cancel(self, operation, timeout=60.0) -> bool:\n"
        "        \"\"\"Cancel without freeing a buffer used by async I/O.\"\"\"\n"
        "        if operation is None:\n"
        "            return True\n"
        "        operation.mark_terminate()\n"
        "        completed = operation.io_complete.wait(timeout=timeout)\n"
        "        if completed:\n"
        "            self.drain_shared_prefetch_releases()\n"
        "        else:\n"
        "            logger.error(\n"
        "                \"Timed out waiting for shared prefetch cancellation: %s\",\n"
        "                operation.request_id,\n"
        "            )\n"
        "        return completed\n"
        "\n"
        "    def drain_shared_prefetch_releases(self):\n"
        "        \"\"\"Release pages revoked by the shared prefetch worker.\"\"\"\n"
        "        while True:\n"
        "            try:\n"
        "                indices = self.host_mem_release_queue.get_nowait()\n"
        "            except Empty:\n"
        "                return\n"
        "            self.mem_pool_host.free(indices)\n",
        "share Prefill prefetch completion and cleanup protocol",
    )

    source = replace_once(
        source,
        "                hash_value, storage_hit_count = self._storage_hit_query(operation)\n"
        "                if self.tp_world_size > 1:\n",
        "                hash_value, storage_hit_count = self._storage_hit_query(operation)\n"
        "                if __import__(\"os\").environ.get(\n"
        "                    \"DUALPATH_L3_HIT_DIAGNOSTIC\", \"1\"\n"
        "                ) == \"1\":\n"
        "                    logger.info(\n"
        "                        \"DUALPATH_L3_HIT_QUERY rid=%s tp_rank=%d \"\n"
        "                        \"requested_pages=%d local_hit_tokens=%d first_hash=%s\",\n"
        "                        operation.request_id,\n"
        "                        self.tp_rank,\n"
        "                        len(hash_value),\n"
        "                        storage_hit_count,\n"
        "                        hash_value[0] if hash_value else \"none\",\n"
        "                    )\n"
        "                if self.tp_world_size > 1:\n",
        "diagnose the first durable L3 hash before TP consensus",
    )
    source = replace_once(
        source,
        "                    storage_hit_count = storage_hit_count_tensor.item()\n"
        "\n"
        "                if storage_hit_count < self.prefetch_threshold:\n",
        "                    storage_hit_count = storage_hit_count_tensor.item()\n"
        "                if __import__(\"os\").environ.get(\n"
        "                    \"DUALPATH_L3_HIT_DIAGNOSTIC\", \"1\"\n"
        "                ) == \"1\":\n"
        "                    logger.info(\n"
        "                        \"DUALPATH_L3_HIT_CONSENSUS rid=%s tp_rank=%d \"\n"
        "                        \"hit_tokens=%d threshold=%d\",\n"
        "                        operation.request_id,\n"
        "                        self.tp_rank,\n"
        "                        storage_hit_count,\n"
        "                        self.prefetch_threshold,\n"
        "                    )\n"
        "                operation.query_complete = True\n"
        "\n"
        "                if storage_hit_count < self.prefetch_threshold:\n",
        "publish shared hash query completion after TP consensus",
    )
    source = replace_once(
        source,
        "                if storage_hit_count < self.prefetch_threshold:\n"
        "                    # not to prefetch if not enough benefits\n"
        "                    self.prefetch_revoke_queue.put(operation.request_id)\n"
        "                    self.append_host_mem_release(operation.host_indices)\n",
        "                if storage_hit_count < self.prefetch_threshold:\n"
        "                    # not to prefetch if not enough benefits\n"
        "                    operation.revoked = True\n"
        "                    operation.mark_terminate()\n"
        "                    self.prefetch_revoke_queue.put(operation.request_id)\n"
        "                    if not operation.retain_host_indices:\n"
        "                        self.append_host_mem_release(operation.host_indices)\n"
        "                    operation.io_complete.set()\n",
        "mark shared prefetch revocation for non-hit requests",
    )
    source = replace_once(
        source,
        "                else:\n"
        "                    operation.hash_value = hash_value[\n"
        "                        : (storage_hit_count // self.page_size)\n"
        "                    ]\n"
        "                    # free the pre-allocated memory for pages that are not hit\n"
        "                    self.append_host_mem_release(\n"
        "                        operation.host_indices[storage_hit_count:]\n"
        "                    )\n"
        "                    operation.host_indices = operation.host_indices[:storage_hit_count]\n",
        "                else:\n"
        "                    operation.hash_value = hash_value[\n"
        "                        : (storage_hit_count // self.page_size)\n"
        "                    ]\n"
        "                    if not operation.retain_host_indices:\n"
        "                        # free the pre-allocated memory for pages that are not hit\n"
        "                        self.append_host_mem_release(\n"
        "                            operation.host_indices[storage_hit_count:]\n"
        "                        )\n"
        "                        operation.host_indices = operation.host_indices[:storage_hit_count]\n",
        "retain Decode staging pages after shared hit query",
    )
    source = replace_once(
        source,
        "                self._page_transfer(operation)\n"
        "                # operation terminated by controller, release pre-allocated memory\n"
        "                self.append_host_mem_release(\n"
        "                    operation.host_indices[operation.completed_tokens :]\n"
        "                )\n",
        "                try:\n"
        "                    self._page_transfer(operation)\n"
        "                except Exception:\n"
        "                    operation.mark_terminate()\n"
        "                    logger.exception(\n"
        "                        \"Shared prefetch I/O failed for %s\",\n"
        "                        operation.request_id,\n"
        "                    )\n"
        "                finally:\n"
        "                    # operation terminated by controller, release\n"
        "                    # pre-allocated memory and publish safe cleanup.\n"
        "                    if operation.retain_host_indices:\n"
        "                        decode_read_end = (\n"
        "                            operation.dualpath_read_end\n"
        "                            or __import__(\"time\").perf_counter()\n"
        "                        )\n"
        "                        decode_read_start = (\n"
        "                            operation.dualpath_read_start\n"
        "                            or decode_read_end\n"
        "                        )\n"
        "                        logger.info(\n"
        "                            \"DUALPATH_DECODE_L3_READ rid=%s \"\n"
        "                            \"decode_engine=%s node=%s tp_rank=%d \"\n"
        "                            \"read_start=%.9f read_end=%.9f read_s=%.9f \"\n"
        "                            \"lustre_read_s=%.9f scatter_s=%.9f tokens=%d\",\n"
        "                            operation.request_id,\n"
        "                            __import__(\"os\").environ.get(\n"
        "                                \"DECODE_ID\", \"0\"\n"
        "                            ),\n"
        "                            __import__(\"os\").environ.get(\n"
        "                                \"DECODE_HOST\",\n"
        "                                __import__(\"os\").environ.get(\n"
        "                                    \"HOSTNAME\", \"unknown\"\n"
        "                                ),\n"
        "                            ),\n"
        "                            self.tp_rank,\n"
        "                            decode_read_start,\n"
        "                            decode_read_end,\n"
        "                            decode_read_end - decode_read_start,\n"
        "                            operation.dualpath_lustre_read_s,\n"
        "                            operation.dualpath_scatter_s,\n"
        "                            operation.completed_tokens,\n"
        "                        )\n"
        "                    if not operation.retain_host_indices:\n"
        "                        self.append_host_mem_release(\n"
        "                            operation.host_indices[operation.completed_tokens :]\n"
        "                        )\n"
        "                    operation.io_complete.set()\n",
        "make shared prefetch I/O cancellation-safe",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_hiradix_prefetch_skips(source: str) -> str:
    """Expose the silent Prefill prefetch skips and unpack aborted requests.

    prefetch_from_storage() abandons a request on three separate conditions
    without emitting anything, so a starved prefetch budget looks exactly like
    a durable-storage miss in the logs. The budget is only
    0.8 * (host pool - device pool) tokens, which a single long-context prefix
    can exhaust on its own.
    """
    if "DUALPATH_L3_PREFETCH_SKIP" not in source:
        source = replace_once(
            source,
            "        if (\n"
            "            not self.enable_storage\n"
            "            or prefetch_length < self.prefetch_threshold\n"
            "            or self.cache_controller.prefetch_rate_limited()\n"
            "        ):\n"
            "            return\n",
            "        if not self.enable_storage:\n"
            "            return\n"
            "        dualpath_skip_reason = None\n"
            "        if prefetch_length < self.prefetch_threshold:\n"
            "            dualpath_skip_reason = \"below_threshold\"\n"
            "        elif self.cache_controller.prefetch_rate_limited():\n"
            "            dualpath_skip_reason = \"rate_limited\"\n"
            "        if dualpath_skip_reason is not None:\n"
            "            if __import__(\"os\").environ.get(\n"
            "                \"DUALPATH_L3_HIT_DIAGNOSTIC\", \"1\"\n"
            "            ) == \"1\":\n"
            "                logger.info(\n"
            "                    \"DUALPATH_L3_PREFETCH_SKIP rid=%s reason=%s \"\n"
            "                    \"requested_tokens=%d threshold=%d \"\n"
            "                    \"occupied=%d limit=%d\",\n"
            "                    req_id,\n"
            "                    dualpath_skip_reason,\n"
            "                    prefetch_length,\n"
            "                    self.prefetch_threshold,\n"
            "                    self.cache_controller.prefetch_tokens_occupied,\n"
            "                    self.cache_controller.prefetch_capacity_limit,\n"
            "                )\n"
            "            return\n",
            "log rate-limited and below-threshold Prefill prefetch skips",
        )
        source = replace_once(
            source,
            "        if host_indices is None:\n"
            "            last_host_node.release_host()\n"
            "            # no sufficient host memory for prefetch\n"
            "            return\n",
            "        if host_indices is None:\n"
            "            last_host_node.release_host()\n"
            "            # no sufficient host memory for prefetch\n"
            "            if __import__(\"os\").environ.get(\n"
            "                \"DUALPATH_L3_HIT_DIAGNOSTIC\", \"1\"\n"
            "            ) == \"1\":\n"
            "                logger.info(\n"
            "                    \"DUALPATH_L3_PREFETCH_SKIP rid=%s \"\n"
            "                    \"reason=no_host_mem requested_tokens=%d \"\n"
            "                    \"host_free=%d\",\n"
            "                    req_id,\n"
            "                    prefetch_length,\n"
            "                    self.cache_controller.mem_pool_host.available_size(),\n"
            "                )\n"
            "            return\n",
            "log host-memory-starved Prefill prefetch skips",
        )
    # ongoing_prefetch gained the request radix namespace as a fifth element.
    return source.replace(
        "        last_host_node, token_ids, host_indices, operation = "
        "self.ongoing_prefetch[rid]\n",
        "        (\n"
        "            last_host_node,\n"
        "            token_ids,\n"
        "            host_indices,\n"
        "            operation,\n"
        "            _,\n"
        "        ) = self.ongoing_prefetch[rid]\n",
    )


def patch_hiradix(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_L3_TIMING"
    if marker in source:
        source = source.replace("extra_key=req.extra_key,\n", "")
        source = source.replace(
            "extra_key=last_host_node.key.extra_key",
            "extra_key=extra_key",
        )
        source = patch_hiradix_prefetch_skips(source)
        path.write_text(source)
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "        prefix_keys: Optional[List[str]] = None,\n"
        "    ):\n"
        "        # align the number of fetching tokens to the page size\n",
        "        prefix_keys: Optional[List[str]] = None,\n"
        "        extra_key: Optional[str] = None,\n"
        "    ):\n"
        "        # Keep the request-local radix namespace across this async\n"
        "        # operation. last_host_node may be the root, whose extra_key\n"
        "        # is always None.\n"
        "        # align the number of fetching tokens to the page size\n",
        "carry the request radix namespace through Prefill prefetch",
    )
    source = replace_once(
        source,
        "        self.ongoing_prefetch[req_id] = (\n"
        "            last_host_node,\n"
        "            new_input_tokens,\n"
        "            host_indices,\n"
        "            operation,\n"
        "        )\n",
        "        self.ongoing_prefetch[req_id] = (\n"
        "            last_host_node,\n"
        "            new_input_tokens,\n"
        "            host_indices,\n"
        "            operation,\n"
        "            extra_key,\n"
        "        )\n",
        "retain the request radix namespace with Prefill prefetch state",
    )
    source = replace_once(
        source,
        "                last_host_node, token_ids, _, _ = info\n",
        "                last_host_node, token_ids, _, _, _ = info\n",
        "unpack request radix namespace on Prefill prefetch revoke",
    )
    source = replace_once(
        source,
        "        last_host_node, token_ids, host_indices, operation = self.ongoing_prefetch[\n"
        "            req_id\n"
        "        ]\n",
        "        (\n"
        "            last_host_node,\n"
        "            token_ids,\n"
        "            host_indices,\n"
        "            operation,\n"
        "            extra_key,\n"
        "        ) = self.ongoing_prefetch[req_id]\n",
        "restore the request radix namespace on Prefill prefetch completion",
    )
    source = replace_once(
        source,
        "                token_ids=fetched_token_ids, extra_key=last_host_node.key.extra_key\n",
        "                token_ids=fetched_token_ids, extra_key=extra_key\n",
        "insert Prefill storage hits into the requesting radix namespace",
    )
    # Some SGLang revisions contain the same insertion in a second completion
    # path. Keep both paths in the request namespace; neither should fall back
    # to the source node's namespace when a request uses a durable L3 lookup.
    source = source.replace(
        "extra_key=last_host_node.key.extra_key",
        "extra_key=extra_key",
    )

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
    can_terminate_start = source.index(
        "    def can_terminate_prefetch(", source.index("class HiRadixCache")
    )
    check_progress_start = source.index(
        "    def check_prefetch_progress(", can_terminate_start
    )
    shared_can_terminate = (
        "    def can_terminate_prefetch(self, operation: PrefetchOperation):\n"
        "        if (\n"
        "            operation.completed_tokens == len(operation.hash_value) * self.page_size\n"
        "            and operation.request_id not in self.dualpath_prefetch_local_end\n"
        "        ):\n"
        "            # Capture this rank's storage IO completion before the\n"
        "            # shared TP completion-consensus all_reduce.\n"
        "            self.dualpath_prefetch_local_end[operation.request_id] = (\n"
        "                time.perf_counter()\n"
        "            )\n"
        "        return self.cache_controller.shared_prefetch_can_terminate(\n"
        "            operation,\n"
        "            stop_policy=self.prefetch_stop_policy,\n"
        "            timeout_base=self.prefetch_timeout_base,\n"
        "            timeout_per_page=self.prefetch_timeout_per_page,\n"
        "        )\n\n"
    )
    source = source[:can_terminate_start] + shared_can_terminate + source[check_progress_start:]
    source = replace_once(
        source,
        "        if operation.host_indices is None:\n"
        "            # prefetch has not been issued due to insufficient host memory\n"
        "            return True\n"
        "\n"
        "        if not self.can_terminate_prefetch(operation):\n",
        "        if not self.can_terminate_prefetch(operation):\n",
        "keep all TP ranks in shared prefetch progress consensus",
    )
    source = replace_once(
        source,
        "        completed_tokens, hash_value = self.cache_controller.terminate_prefetch(\n"
        "            operation\n"
        "        )\n",
        "        completed_tokens, hash_value = self.cache_controller.shared_prefetch_finish(\n"
        "            operation\n"
        "        )\n",
        "reuse shared Prefill prefetch completion consensus",
    )
    source = replace_once(
        source,
        "        min_completed_tokens = completed_tokens\n"
        "        if self.tp_world_size > 1:\n"
        "            # synchrnoize TP workers to make the same update to hiradix cache\n"
        "            completed_tokens_tensor = torch.tensor(\n"
        "                min_completed_tokens, dtype=torch.int\n"
        "            )\n"
        "            torch.distributed.all_reduce(\n"
        "                completed_tokens_tensor,\n"
        "                op=torch.distributed.ReduceOp.MIN,\n"
        "                group=self.tp_group,\n"
        "            )\n"
        "            min_completed_tokens = completed_tokens_tensor.item()\n",
        "        # shared_prefetch_finish already reduced this count across TP.\n"
        "        min_completed_tokens = completed_tokens\n",
        "remove duplicate Prefill prefetch completion reduction",
    )
    source = replace_once(
        source,
        "    def load_back(\n"
        "        self, node: TreeNode, mem_quota: Optional[int] = None\n"
        "    ) -> Optional[torch.Tensor]:\n",
        "    def dualpath_drop_request_cache(self, last_node) -> int:\n"
        "        \"\"\"Drop one request's radix metadata after durable backup.\"\"\"\n"
        "        if last_node is None:\n"
        "            return 0\n"
        "        removed_tokens = 0\n"
        "        if last_node.lock_ref > 0:\n"
        "            # dec_lock_ref already walks to the root; calling it again\n"
        "            # for each ancestor would consume locks owned by other\n"
        "            # concurrent requests that share the same prefix.\n"
        "            self.dec_lock_ref(last_node)\n"
        "        node = last_node\n"
        "        while node is not self.root_node:\n"
        "            parent = node.parent\n"
        "            if (\n"
        "                node.lock_ref != 0\n"
        "                or node.host_ref_counter != 0\n"
        "                or node.children\n"
        "            ):\n"
        "                break\n"
        "            if node.value is not None:\n"
        "                removed_tokens += len(node.value)\n"
        "                node.value = None\n"
        "            if node.host_value is not None:\n"
        "                self.cache_controller.evict_host(node.host_value)\n"
        "                node.host_value = None\n"
        "            self._delete_leaf(node)\n"
        "            node = parent\n"
        "        return removed_tokens\n"
        "\n"
        "    def load_back(\n"
        "        self, node: TreeNode, mem_quota: Optional[int] = None\n"
        "    ) -> Optional[torch.Tensor]:\n",
        "drop request radix metadata after Decode backup",
    )
    source = patch_hiradix_prefetch_skips(source)
    path.write_text(source)
    print(f"Patched: {path}")


def patch_scheduler_prefetch_namespace(source: str) -> str:
    """Carry the request radix namespace into the Prefill prefetch.

    The durable L3 hash lookup must stay namespace-free so turns can share
    pages, but the restored pages are inserted back into the request-local
    radix namespace. prefetch_from_storage() only stores extra_key in
    ongoing_prefetch for that insert; it never reaches cache_controller's
    hash query. Without it the pages land under extra_key=None while
    init_next_round_input() looks them up under the request's
    "dualpath:<rid>:" namespace, so the prefix never matches and Prefill
    recomputes the whole prompt.
    """
    if "extra_key=req.extra_key" in source:
        return source
    return replace_once(
        source,
        "                self.tree_cache.prefetch_from_storage(\n"
        "                    req.rid,\n"
        "                    req.last_host_node,\n"
        "                    new_input_tokens,\n"
        "                    last_hash,\n"
        "                    prefix_keys,\n"
        "                )\n",
        "                self.tree_cache.prefetch_from_storage(\n"
        "                    req.rid,\n"
        "                    req.last_host_node,\n"
        "                    new_input_tokens,\n"
        "                    last_hash,\n"
        "                    prefix_keys,\n"
        "                    extra_key=req.extra_key,\n"
        "                )\n",
        "restore the request radix namespace for the local prefetch insert",
    )


def patch_decode_l3_scheduler(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_DECODE_L3_READ_PATH"
    if marker in source:
        source = patch_scheduler_prefetch_namespace(source)
        path.write_text(source)
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
    source = patch_scheduler_prefetch_namespace(source)
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
        "        if DUALPATH_TTFT_DETAIL:\n"
        "            stats = self.time_stats\n"
        "            prefill_s = stats.get_prefill_launch_latency()\n"
        "            logger.info(\n"
        "                \"DUALPATH_REQ_TIMING rid=%s type=%s queue_s=%.9f \"\n"
        "                \"alloc_s=%.9f prefill_s=%.9f forward_s=%.9f \"\n"
        "                \"queue_start=%.9f queue_end=%.9f prefill_start=%.9f \"\n"
        "                \"prefill_end=%.9f prefill_bootstrap_start=%.9f \"\n"
        "                \"decode_prealloc_start=%.9f decode_receive_start=%.9f\",\n"
        "                self.rid,\n"
        "                stats.disagg_mode_str(),\n"
        "                max(0.0, stats.get_queueing_time()),\n"
        "                max(0.0, stats.alloc_waiting_duration),\n"
        "                max(0.0, prefill_s or 0.0),\n"
        "                max(0.0, stats.completion_time - stats.forward_entry_time),\n"
        "                stats.wait_queue_entry_time,\n"
        "                stats.forward_entry_time,\n"
        "                stats.prefill_start_time_host,\n"
        "                stats.prefill_end_time_host,\n"
        "                stats.prefill_bootstrap_queue_entry_time,\n"
        "                stats.decode_prealloc_queue_entry_time,\n"
        "                stats.decode_transfer_queue_entry_time,\n"
        "            )\n"
        "        self.has_log_time_stats = True\n",
        "record request stage durations",
    )
    source = replace_once(
        source,
        "        self.extra_key = extra_key\n"
        "        self.lora_id = lora_id\n",
        "        # DUALPATH_NO_CROSS_TURN_KV_REUSE: each replay turn has a\n"
        "        # distinct request id. Isolate completed turns from L1/L2\n"
        "        # radix reuse while preserving reuse within one request.\n"
        "        if __import__(\"os\").environ.get(\n"
        "            \"DUALPATH_NO_CROSS_TURN_KV_REUSE\", \"1\"\n"
        "        ) == \"1\":\n"
        "            self.extra_key = f\"dualpath:{self.rid}:{extra_key or ''}\"\n"
        "        else:\n"
        "            self.extra_key = extra_key\n"
        "        self.lora_id = lora_id\n",
        "isolate each DualPath request from cross-turn radix reuse",
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
        "        # Leave cached_tokens at 0 here. These pages stay in\n"
        "        # prefix_indices, so ScheduleBatch.prepare_for_extend adds\n"
        "        # `pre_len - already_computed` for them right after; stamping\n"
        "        # hit_tokens as well reported the same prefix twice. Decode\n"
        "        # learns the durable boundary from\n"
        "        # dualpath_decode_l3_hit_tokens, not from this counter.\n"
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
        "            kv_args.dualpath_decode_device_allocator = self.token_to_kv_pool_allocator\n"
        "            kv_args.dualpath_decode_tp_group = self.gloo_group\n"
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
    marker = "DUALPATH_DECODE_SCHEDULER_PREFETCH"
    if marker in source:
        print(f"Already patched: {path}")
        return

    source = replace_once(
        source,
        "    metadata_buffer_index: int = -1\n",
        "    metadata_buffer_index: int = -1\n"
        "    # DUALPATH_DECODE_SCHEDULER_PREFETCH: Decode mirrors Prefill's\n"
        "    # scheduler-owned prefetch lifecycle. The receiver remains passive.\n"
        "    dualpath_l3_prefetch_operation: Optional[object] = None\n"
        "    dualpath_l3_page_indices: Optional[object] = None\n"
        "    dualpath_l3_state_indices: Optional[object] = None\n"
        "    dualpath_l3_host_token_indices: Optional[object] = None\n"
        "    dualpath_l3_host_page_indices: Optional[object] = None\n"
        "    dualpath_l3_hit_tokens: int = 0\n"
        "    dualpath_l3_prefetch_timing: Optional[dict] = None\n",
        "store Decode prefetch state on the scheduler request",
    )
    source = replace_once(
        source,
        "    def pop_preallocated(\n"
        "        self, rids_to_check: Optional[List[str]] = None\n"
        "    ) -> Tuple[List[DecodeRequest], List[DecodeRequest]]:\n",
        "    def _dualpath_init_receiver(self, decode_req: DecodeRequest) -> None:\n"
        "        \"\"\"Commit scheduler-owned prefetch state to the passive receiver.\"\"\"\n"
        "        if getattr(\n"
        "            decode_req.kv_receiver,\n"
        "            \"dualpath_scheduler_prefetch_supported\",\n"
        "            False,\n"
        "        ):\n"
        "            decode_req.kv_receiver.init(\n"
        "                decode_req.dualpath_l3_page_indices,\n"
        "                decode_req.metadata_buffer_index,\n"
        "                decode_req.dualpath_l3_state_indices,\n"
        "                request_id=decode_req.req.rid,\n"
        "                host_staging_token_indices=(\n"
        "                    decode_req.dualpath_l3_host_token_indices\n"
        "                ),\n"
        "                host_staging_page_indices=(\n"
        "                    decode_req.dualpath_l3_host_page_indices\n"
        "                ),\n"
        "                decode_l3_hit_tokens=decode_req.dualpath_l3_hit_tokens,\n"
        "                decode_l3_prefetch_timing=(\n"
        "                    decode_req.dualpath_l3_prefetch_timing or {}\n"
        "                ),\n"
        "            )\n"
        "        else:\n"
        "            decode_req.kv_receiver.init(\n"
        "                decode_req.dualpath_l3_page_indices,\n"
        "                decode_req.metadata_buffer_index,\n"
        "                decode_req.dualpath_l3_state_indices,\n"
        "            )\n"
        "        decode_req.req.time_stats.decode_transfer_queue_entry_time = (\n"
        "            time.perf_counter()\n"
        "        )\n"
        "        decode_req.req.add_latency(RequestStage.DECODE_BOOTSTRAP)\n"
        "        trace_slice_end(\n"
        "            RequestStage.DECODE_BOOTSTRAP,\n"
        "            decode_req.req.rid,\n"
        "            auto_next_anon=True,\n"
        "        )\n"
        "\n"
        "    def _dualpath_progress_l3_prefetch(\n"
        "        self, rids_to_check: Optional[List[str]] = None\n"
        "    ) -> Tuple[List[DecodeRequest], set[int]]:\n"
        "        \"\"\"Advance Decode prefetch in FIFO order, as Prefill does.\"\"\"\n"
        "        ready = []\n"
        "        indices_to_remove = set()\n"
        "        for i, decode_req in enumerate(self.queue):\n"
        "            operation = decode_req.dualpath_l3_prefetch_operation\n"
        "            if operation is None:\n"
        "                continue\n"
        "            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:\n"
        "                continue\n"
        "            done, hit_tokens = self.kv_manager.poll_decode_l3_prefetch(\n"
        "                operation\n"
        "            )\n"
        "            if not done:\n"
        "                # Keep collective order identical on every TP rank. Later\n"
        "                # operations may perform I/O, but cannot commit first.\n"
        "                break\n"
        "            decode_req.dualpath_l3_hit_tokens = hit_tokens\n"
        "            decode_req.dualpath_l3_prefetch_timing = (\n"
        "                operation.dualpath_timing_snapshot()\n"
        "                if hasattr(operation, \"dualpath_timing_snapshot\")\n"
        "                else {}\n"
        "            )\n"
        "            decode_req.dualpath_l3_prefetch_operation = None\n"
        "            self._dualpath_init_receiver(decode_req)\n"
        "            ready.append(decode_req)\n"
        "            indices_to_remove.add(i)\n"
        "        return ready, indices_to_remove\n"
        "\n"
        "    def pop_preallocated(\n"
        "        self, rids_to_check: Optional[List[str]] = None\n"
        "    ) -> Tuple[List[DecodeRequest], List[DecodeRequest]]:\n",
        "add Prefill-style Decode scheduler prefetch progress",
    )
    source = replace_once(
        source,
        "        failed_reqs = []\n"
        "        preallocated_reqs = []\n"
        "        indices_to_remove = set()\n",
        "        failed_reqs = []\n"
        "        (\n"
        "            preallocated_reqs,\n"
        "            indices_to_remove,\n"
        "        ) = self._dualpath_progress_l3_prefetch(rids_to_check)\n",
        "advance Decode L3 prefetch before new allocations",
    )
    source = replace_once(
        source,
        "        self._update_handshake_waiters(rids_to_check)\n"
        "\n"
        "        failed_reqs = []\n",
        "        self._update_handshake_waiters(rids_to_check)\n"
        "        if hasattr(\n"
        "            self.kv_manager, \"drain_decode_l3_prefetch_cancellations\"\n"
        "        ):\n"
        "            self.kv_manager.drain_decode_l3_prefetch_cancellations()\n"
        "\n"
        "        failed_reqs = []\n",
        "drain cancelled Decode prefetch operations without blocking",
    )
    source = replace_once(
        source,
        "        # First, remove all failed requests from the queue\n"
        "        for i, decode_req in enumerate(self.queue):\n"
        "            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:\n"
        "                continue\n"
        "            if isinstance(decode_req.req.finished_reason, FINISH_ABORT):\n",
        "        # First, remove all failed requests from the queue\n"
        "        for i, decode_req in enumerate(self.queue):\n"
        "            if rids_to_check is not None and decode_req.req.rid not in rids_to_check:\n"
        "                continue\n"
        "            if i in indices_to_remove:\n"
        "                continue\n"
        "            if isinstance(decode_req.req.finished_reason, FINISH_ABORT):\n"
        "                operation = decode_req.dualpath_l3_prefetch_operation\n"
        "                host_token_indices = (\n"
        "                    decode_req.dualpath_l3_host_token_indices\n"
        "                )\n"
        "                if operation is not None and hasattr(\n"
        "                    self.kv_manager, \"cancel_decode_l3_prefetch\"\n"
        "                ):\n"
        "                    self.kv_manager.cancel_decode_l3_prefetch(\n"
        "                        operation, host_token_indices\n"
        "                    )\n"
        "                    decode_req.dualpath_l3_prefetch_operation = None\n"
        "                    decode_req.dualpath_l3_host_token_indices = None\n"
        "                elif host_token_indices is not None and hasattr(\n"
        "                    self.kv_manager, \"release_decode_host_staging\"\n"
        "                ):\n"
        "                    self.kv_manager.release_decode_host_staging(\n"
        "                        host_token_indices\n"
        "                    )\n"
        "                    decode_req.dualpath_l3_host_token_indices = None\n"
        "                if decode_req.dualpath_l3_page_indices is not None:\n"
        "                    release_kv_cache(\n"
        "                        decode_req.req, self.tree_cache, is_insert=False\n"
        "                    )\n"
        "                    decode_req.dualpath_l3_page_indices = None\n"
        "                if decode_req.metadata_buffer_index != -1:\n"
        "                    self.req_to_metadata_buffer_idx_allocator.free(\n"
        "                        decode_req.metadata_buffer_index\n"
        "                    )\n"
        "                    decode_req.metadata_buffer_index = -1\n",
        "release scheduler-owned Decode prefetch state on abort",
    )
    source = replace_once(
        source,
        "        # Then, preallocate the remaining requests if possible\n"
        "        for i, decode_req in enumerate(self.queue):\n",
        "        # Then, preallocate the remaining requests if possible\n"
        "        dualpath_prefetch_inflight = any(\n"
        "            item.dualpath_l3_prefetch_operation is not None\n"
        "            for item in self.queue\n"
        "        )\n"
        "        dualpath_cancel_pending = (\n"
        "            hasattr(self.kv_manager, \"decode_l3_prefetch_cancel_pending\")\n"
        "            and self.kv_manager.decode_l3_prefetch_cancel_pending()\n"
        "        )\n"
        "        for i, decode_req in enumerate(self.queue):\n",
        "serialize Decode prefetch query collectives",
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
        "            if dualpath_prefetch_inflight or dualpath_cancel_pending:\n"
        "                # The shared Prefill controller has one ordered query\n"
        "                # stream. Do not interleave another Decode TP query.\n"
        "                break\n"
        "\n"
        "            if self.req_to_token_pool.available_size() <= 0:\n",
        "avoid duplicate Decode allocation while prefetch is in flight",
    )
    source = replace_once(
        source,
        "            page_indices = kv_to_page_indices(kv_indices, page_size)\n"
        "            decode_req.kv_receiver.init(\n"
        "                page_indices, decode_req.metadata_buffer_index, state_indices\n"
        "            )\n"
        "            preallocated_reqs.append(decode_req)\n"
        "            indices_to_remove.add(i)\n"
        "            decode_req.req.time_stats.decode_transfer_queue_entry_time = (\n"
        "                time.perf_counter()\n"
        "            )\n"
        "            decode_req.req.add_latency(RequestStage.DECODE_BOOTSTRAP)\n"
        "            trace_slice_end(\n"
        "                RequestStage.DECODE_BOOTSTRAP, decode_req.req.rid, auto_next_anon=True\n"
        "            )\n",
        "            page_indices = kv_to_page_indices(kv_indices, page_size)\n"
        "            decode_req.dualpath_l3_page_indices = page_indices\n"
        "            decode_req.dualpath_l3_state_indices = state_indices\n"
        "            if getattr(\n"
        "                self.kv_manager, \"decode_host_staging_enabled\", False\n"
        "            ):\n"
        "                host_alloc_start = time.perf_counter()\n"
        "                (\n"
        "                    decode_req.dualpath_l3_host_token_indices,\n"
        "                    decode_req.dualpath_l3_host_page_indices,\n"
        "                ) = self.kv_manager.allocate_decode_host_staging(page_indices)\n"
        "                decode_req.kv_receiver.dualpath_host_alloc_start = (\n"
        "                    host_alloc_start\n"
        "                )\n"
        "                decode_req.kv_receiver.dualpath_host_alloc_end = (\n"
        "                    time.perf_counter()\n"
        "                )\n"
        "\n"
        "            decode_l3_read = (\n"
        "                __import__(\"os\").environ.get(\n"
        "                    \"DUALPATH_L3_READ_PATH\", \"prefill\"\n"
        "                )\n"
        "                == \"decode\"\n"
        "            )\n"
        "            if decode_l3_read and hasattr(\n"
        "                self.kv_manager, \"start_decode_l3_prefetch\"\n"
        "            ):\n"
        "                decode_req.dualpath_l3_prefetch_operation = (\n"
        "                    self.kv_manager.start_decode_l3_prefetch(\n"
        "                        decode_req.req.rid,\n"
        "                        decode_req.req.origin_input_ids,\n"
        "                        decode_req.dualpath_l3_host_token_indices,\n"
        "                    )\n"
        "                )\n"
        "                if decode_req.dualpath_l3_prefetch_operation is not None:\n"
        "                    # Leave the request in the preallocation queue. The\n"
        "                    # next scheduler iterations mirror Prefill's\n"
        "                    # check_prefetch_progress lifecycle.\n"
        "                    dualpath_prefetch_inflight = True\n"
        "                    break\n"
        "\n"
        "            self._dualpath_init_receiver(decode_req)\n"
        "            preallocated_reqs.append(decode_req)\n"
        "            indices_to_remove.add(i)\n",
        "move Decode L3 operation ownership into DecodePreallocQueue",
    )
    source = replace_once(
        source,
        "                self.scheduler.stream_output(\n"
        "                    [decode_req.req], decode_req.req.return_logprob\n"
        "                )\n"
        "                # release pre-allocated kv cache, but don't insert into the tree since it's failed\n",
        "                self.scheduler.stream_output(\n"
        "                    [decode_req.req], decode_req.req.return_logprob\n"
        "                )\n"
        "                decode_req.kv_receiver.clear()\n"
        "                decode_req.kv_receiver = None\n"
        "                # release pre-allocated kv cache, but don't insert into the tree since it's failed\n",
        "release Decode staging after transfer failure",
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
        "            # DUALPATH_DECODE_L3_PREFETCH_WIRING\n"
        "            if (\n",
        "            page_indices = kv_to_page_indices(kv_indices, page_size)\n"
        "            # DUALPATH_DECODE_L3_PREFETCH_WIRING\n"
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
    """Decode receiver timing is emitted by the shared async receiver init."""
    source = path.read_text()
    if "dualpath_host_alloc_start" in source:
        print(f"Already patched: {path}")
        return
    # The async Decode L3 implementation records allocation and prefetch
    # timestamps in MooncakeKVReceiver.init/poll. Keep this hook for older
    # patch ordering without reintroducing the synchronous reader.
    path.write_text(source)
    print(f"Decode receiver timing is provided by async L3 patch: {path}")


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
        "        self.decode_l3_bootstrap_sent = False\n"
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
    marker = "DUALPATH_DECODE_L3_REVERSE"
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
        "        self.decode_l3_prefetch_controller = None\n"
        "        self.decode_l3_prefetch_tokens = 0\n"
        "        self.decode_l3_cancelled_prefetch = []\n"
        "        if (\n"
        "            self.disaggregation_mode == DisaggregationMode.DECODE\n"
        "            and self.decode_l3_read_path\n"
        "            and getattr(args, \"dualpath_decode_host_staging_pool\", None)\n"
        "            is not None\n"
        "        ):\n"
        "            from sglang.srt.managers.cache_controller import HiCacheController\n"
        "            extra_config = server_args.hicache_storage_backend_extra_config\n"
        "            if isinstance(extra_config, str):\n"
        "                try:\n"
        "                    extra_config = __import__(\"json\").loads(extra_config)\n"
        "                except Exception as error:\n"
        "                    raise ValueError(\n"
        "                        f\"Invalid HiCache backend config: {error}\"\n"
        "                    ) from error\n"
        "            extra_config = extra_config or {}\n"
        "            self.decode_l3_prefetch_controller = HiCacheController(\n"
        "                token_to_kv_pool_allocator=getattr(\n"
        "                    args, \"dualpath_decode_device_allocator\"\n"
        "                ),\n"
        "                mem_pool_host=args.dualpath_decode_host_staging_pool,\n"
        "                page_size=args.dualpath_decode_host_staging_pool.page_size,\n"
        "                tp_group=args.dualpath_decode_tp_group,\n"
        "                load_cache_event=threading.Event(),\n"
        "                write_policy=\"write_through\",\n"
        "                io_backend=server_args.hicache_io_backend,\n"
        "                storage_backend=server_args.hicache_storage_backend,\n"
        "                prefetch_threshold=int(\n"
        "                    extra_config.get(\"prefetch_threshold\", 256)\n"
        "                ),\n"
        "                model_name=server_args.served_model_name,\n"
        "                storage_backend_extra_config=extra_config,\n"
        "            )\n"
        "            # The dedicated staging pool is intentionally independent\n"
        "            # of the Decode page-first HiCache pool.\n"
        "            self.decode_l3_prefetch_controller.prefetch_capacity_limit = int(\n"
        "                0.8 * args.dualpath_decode_host_staging_pool.size\n"
        "            )\n"
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
        "        lustre_read_s = 0.0\n"
        "        scatter_s = 0.0\n"
        "        hit_tokens = 0\n"
        "        prior_hash = None\n"
        "        target = self.decode_host_staging_pool.get_dummy_flat_data_page()\n"
        "        for offset in range(0, candidate_tokens, page_size):\n"
        "            page_tokens = token_ids[offset : offset + page_size]\n"
        "            page_hash = get_hash_str(page_tokens, prior_hash=prior_hash)\n"
        "            lustre_start = time.perf_counter()\n"
        "            data_page = self.decode_l3_storage_backend.get(page_hash, target)\n"
        "            lustre_read_s += time.perf_counter() - lustre_start\n"
        "            if data_page is None:\n"
        "                break\n"
        "            scatter_start = time.perf_counter()\n"
        "            self.decode_host_staging_pool.set_from_flat_data_page(\n"
        "                int(host_token_indices[offset]), data_page\n"
        "            )\n"
        "            scatter_s += time.perf_counter() - scatter_start\n"
        "            hit_tokens += page_size\n"
        "            prior_hash = page_hash\n"
        "        read_end = time.perf_counter()\n"
        "        logger.info(\n"
        "            \"DUALPATH_DECODE_L3_READ rid=%s decode_engine=%s \"\n"
        "            \"node=%s tp_rank=%d read_start=%.9f read_end=%.9f \"\n"
        "            \"read_s=%.9f lustre_read_s=%.9f scatter_s=%.9f tokens=%d\",\n"
        "            request_id,\n"
        "            os.environ.get(\"DECODE_ID\", \"0\"),\n"
        "            os.environ.get(\n"
        "                \"DECODE_HOST\", os.environ.get(\"HOSTNAME\", \"unknown\")\n"
        "            ),\n"
        "            self.kv_args.engine_rank,\n"
        "            read_start,\n"
        "            read_end,\n"
        "            read_end - read_start,\n"
        "            lustre_read_s,\n"
        "            scatter_s,\n"
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
    # Replace the old synchronous Decode reader with the shared Prefill
    # CacheController pipeline. The preceding replacement is retained as the
    # source anchor for versions of SGLang that have already received the
    # historical DualPath patch.
    legacy_start = source.index(
        "    def load_decode_l3_prefix("
    )
    legacy_end = source.index(
        "    def register_decode_l3_receiver", legacy_start
    )
    source = source[:legacy_start] + source[legacy_end:]
    source = replace_once(
        source,
        "    def _transfer_data(self, mooncake_session_id, transfer_blocks):\n",
        "    def start_decode_l3_prefetch(\n"
        "        self, request_id, token_ids, host_token_indices\n"
        "    ):\n"
        "        if (\n"
        "            not self.decode_l3_read_path\n"
        "            or self.decode_l3_prefetch_controller is None\n"
        "            or not self.decode_host_staging_enabled\n"
        "        ):\n"
        "            return None\n"
        "        page_size = self.decode_host_staging_pool.page_size\n"
        "        candidate_tokens = max(0, len(token_ids) - 1)\n"
        "        candidate_tokens -= candidate_tokens % page_size\n"
        "        if candidate_tokens < page_size:\n"
        "            return None\n"
        "        controller = self.decode_l3_prefetch_controller\n"
        "        if controller.prefetch_rate_limited():\n"
        "            return None\n"
        "        operation = controller.prefetch(\n"
        "            request_id,\n"
        "            host_token_indices[:candidate_tokens],\n"
        "            token_ids[:candidate_tokens],\n"
        "            last_hash=None,\n"
        "            prefix_keys=None,\n"
        "            retain_host_indices=True,\n"
        "        )\n"
        "        controller.prefetch_tokens_occupied += candidate_tokens\n"
        "        return operation\n"
        "\n"
        "    def poll_decode_l3_prefetch(self, operation):\n"
        "        controller = self.decode_l3_prefetch_controller\n"
        "        if operation is None or controller is None:\n"
        "            return True, 0\n"
        "        if not controller.shared_prefetch_can_terminate(\n"
        "            operation, stop_policy=\"wait_complete\"\n"
        "        ):\n"
        "            return False, 0\n"
        "        completed_tokens, _ = controller.shared_prefetch_finish(operation)\n"
        "        if operation.revoked:\n"
        "            try:\n"
        "                controller.prefetch_revoke_queue.get_nowait()\n"
        "            except Exception:\n"
        "                pass\n"
        "        controller.drain_shared_prefetch_releases()\n"
        "        controller.prefetch_tokens_occupied = max(\n"
        "            0, controller.prefetch_tokens_occupied - len(operation.token_ids)\n"
        "        )\n"
        "        return True, completed_tokens\n"
        "\n"
        "    def cancel_decode_l3_prefetch(\n"
        "        self, operation, host_token_indices\n"
        "    ):\n"
        "        \"\"\"Cancel like Prefill without blocking the scheduler thread.\"\"\"\n"
        "        if operation is None or self.decode_l3_prefetch_controller is None:\n"
        "            if host_token_indices is not None:\n"
        "                self.release_decode_host_staging(host_token_indices)\n"
        "            return\n"
        "        operation.mark_terminate()\n"
        "        self.decode_l3_cancelled_prefetch.append(\n"
        "            (operation, host_token_indices)\n"
        "        )\n"
        "\n"
        "    def drain_decode_l3_prefetch_cancellations(self):\n"
        "        if not self.decode_l3_cancelled_prefetch:\n"
        "            return\n"
        "        controller = self.decode_l3_prefetch_controller\n"
        "        pending = []\n"
        "        for operation, host_token_indices in (\n"
        "            self.decode_l3_cancelled_prefetch\n"
        "        ):\n"
        "            if not operation.io_complete.is_set():\n"
        "                pending.append((operation, host_token_indices))\n"
        "                continue\n"
        "            controller.prefetch_tokens_occupied = max(\n"
        "                0,\n"
        "                controller.prefetch_tokens_occupied\n"
        "                - len(operation.token_ids),\n"
        "            )\n"
        "            if operation.revoked:\n"
        "                try:\n"
        "                    controller.prefetch_revoke_queue.get_nowait()\n"
        "                except Exception:\n"
        "                    pass\n"
        "            if host_token_indices is not None:\n"
        "                self.release_decode_host_staging(host_token_indices)\n"
        "        self.decode_l3_cancelled_prefetch = pending\n"
        "\n"
        "    def decode_l3_prefetch_cancel_pending(self):\n"
        "        return bool(self.decode_l3_cancelled_prefetch)\n"
        "\n"
        "    def _transfer_data(self, mooncake_session_id, transfer_blocks):\n",
        "use shared Prefill async prefetch for Decode L3 staging",
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
        "        self.decode_l3_bootstrap_sent = False\n"
        "        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)\n",
        "        self.host_staging_h2d_start = None\n"
        "        self.host_staging_h2d_end = None\n"
        "        self.dualpath_scheduler_prefetch_supported = True\n"
        "        self.host_staging_page_indices = np.array([], dtype=np.int32)\n"
        "        self.decode_l3_bootstrap_sent = False\n"
        "        self.decode_l3_hit_tokens = 0\n"
        "        self.dualpath_prefetch_timing = {}\n"
        "        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)\n",
        "initialize Decode L3 receiver state",
    )
    receiver_init_start = source.index(
        "    def init(", source.index("class MooncakeKVReceiver")
    )
    receiver_init_end = source.index("\n    def poll(", receiver_init_start)
    receiver_init = (
        "    def init(\n"
        "        self,\n"
        "        kv_indices: npt.NDArray[np.int32],\n"
        "        aux_index: Optional[int] = None,\n"
        "        state_indices: Optional[List[int]] = None,\n"
        "        request_id: Optional[str] = None,\n"
        "        host_staging_token_indices=None,\n"
        "        host_staging_page_indices=None,\n"
        "        decode_l3_hit_tokens: int = 0,\n"
        "        decode_l3_prefetch_timing=None,\n"
        "    ):\n"
        "        if self.bootstrap_infos is None:\n"
        "            self.kv_mgr.record_failure(\n"
        "                self.bootstrap_room,\n"
        "                f\"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}\",\n"
        "            )\n"
        "            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)\n"
        "            return\n"
        "        self.decode_aux_index = aux_index\n"
        "        self.decode_state_indices = state_indices\n"
        "        self.kv_indices = np.array(kv_indices, dtype=np.int64, copy=True)\n"
        "        if self.kv_mgr.decode_host_staging_enabled:\n"
        "            if host_staging_token_indices is None:\n"
        "                raise RuntimeError(\n"
        "                    \"Decode scheduler did not provide Host staging\"\n"
        "                )\n"
        "            self.host_staging_token_indices = host_staging_token_indices\n"
        "            self.host_staging_page_indices = np.array(\n"
        "                host_staging_page_indices, dtype=np.int32, copy=True\n"
        "            )\n"
        "        self.decode_l3_hit_tokens = int(decode_l3_hit_tokens)\n"
        "        self.dualpath_prefetch_timing = decode_l3_prefetch_timing or {}\n"
        "        self.init_time = time.time()\n"
        "        if request_id is not None:\n"
        "            self.kv_mgr.register_decode_l3_receiver(\n"
        "                self.bootstrap_room, self\n"
        "            )\n"
        "        self._send_decode_l3_bootstrap()\n"
        "\n"
        "    def _send_decode_l3_bootstrap(self):\n"
        "        if self.decode_l3_bootstrap_sent:\n"
        "            return\n"
        "        for bootstrap_info in self.bootstrap_infos:\n"
        "            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)\n"
        "            is_dummy = bootstrap_info[\"is_dummy\"]\n"
        "            with lock:\n"
        "                sock.send_multipart(\n"
        "                    [\n"
        "                        str(self.bootstrap_room).encode(\"ascii\"),\n"
        "                        self.kv_mgr.local_ip.encode(\"ascii\"),\n"
        "                        str(self.kv_mgr.rank_port).encode(\"ascii\"),\n"
        "                        self.session_id.encode(\"ascii\"),\n"
        "                        (\n"
        "                            self.host_staging_page_indices.tobytes()\n"
        "                            if self.kv_mgr.decode_host_staging_enabled\n"
        "                            else self.kv_indices.tobytes()\n"
        "                        ) if not is_dummy else b\"\",\n"
        "                        str(self.decode_aux_index).encode(\"ascii\")\n"
        "                        if not is_dummy\n"
        "                        else b\"\",\n"
        "                        (\n"
        "                            np.asarray(\n"
        "                                self.decode_state_indices, dtype=np.int32\n"
        "                            ).tobytes()\n"
        "                            if not is_dummy and self.decode_state_indices is not None\n"
        "                            else b\"\"\n"
        "                        ),\n"
        "                        str(self.required_dst_info_num).encode(\"ascii\"),\n"
        "                        str(self.decode_l3_hit_tokens).encode(\"ascii\"),\n"
        "                    ]\n"
        "                )\n"
        "        self.decode_l3_bootstrap_sent = True\n"
    )
    source = (
        source[:receiver_init_start]
        + receiver_init
        + source[receiver_init_end:]
    )
    poll_start = source.index(
        "    def poll(self) -> KVPoll:",
        source.index("class MooncakeKVReceiver"),
    )
    poll_end = source.index("\n    def clear(self) -> None:", poll_start)
    source = (
        source[:poll_start]
        + "    def poll(self) -> KVPoll:\n"
        + "        if self.conclude_state is None:\n"
        + "            status = self.kv_mgr.check_status(self.bootstrap_room)\n"
        + "            if status in (KVPoll.Success, KVPoll.Failed):\n"
        + "                if (\n"
        + "                    status == KVPoll.Success\n"
        + "                    and self.kv_mgr.decode_host_staging_enabled\n"
        + "                    and self.host_staging_h2d_end is None\n"
        + "                ):\n"
        + "                    (\n"
        + "                        self.host_staging_h2d_start,\n"
        + "                        self.host_staging_h2d_end,\n"
        + "                    ) = self.kv_mgr.stage_received_kv_to_device(\n"
        + "                        self.host_staging_token_indices,\n"
        + "                        self.kv_indices,\n"
        + "                    )\n"
        + "                    self.kv_mgr.release_decode_host_staging(\n"
        + "                        self.host_staging_token_indices\n"
        + "                    )\n"
        + "                    self.host_staging_token_indices = None\n"
        + "                self.conclude_state = status\n"
        + "            elif status == KVPoll.WaitingForInput:\n"
        + "                if self.init_time is not None:\n"
        + "                    now = time.time()\n"
        + "                    elapsed = now - self.init_time\n"
        + "                    if elapsed >= self.kv_mgr.waiting_timeout:\n"
        + "                        logger.warning_once(\n"
        + "                            \"Some requests fail to receive KV Cache transfer done signal after bootstrapping. \"\n"
        + "                            \"If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600' (10 minutes) to relax the timeout condition. \"\n"
        + "                        )\n"
        + "                        self.kv_mgr.record_failure(\n"
        + "                            self.bootstrap_room,\n"
        + "                            f\"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput\",\n"
        + "                        )\n"
        + "                        self.conclude_state = KVPoll.Failed\n"
        + "                        return KVPoll.Failed\n"
        + "            return status\n"
        + "        return self.conclude_state\n"
        + source[poll_end:]
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


DUALPATH_READ_PATH_IMPORT = (
    "from sglang.srt.utils.dualpath_read_path import (\n"
    "    is_decode_read as dualpath_is_decode_read,\n"
    ")\n"
)


def patch_dynamic_read_path(source: str, sites: tuple[tuple[str, str], ...],
                            anchor: str, label: str) -> str:
    """Turn process-level read-path switches into per-request decisions.

    Every one of these call sites already has the request in scope, so the
    only thing the process-level environment lookup bought was uniformity.
    Replacing it with a pure function of the rid keeps static `prefill` and
    `decode` behaviour byte-identical while letting `dynamic` choose per
    request.
    """
    if "dualpath_is_decode_read" in source:
        return source
    for old, new in sites:
        source = replace_once(source, old, new, label)
    return replace_once(
        source, anchor, DUALPATH_READ_PATH_IMPORT + anchor,
        f"import the read-path resolver for {label}",
    )


_ENV_DECODE_READ = (
    "__import__(\"os\").environ.get(\n"
    "                    \"DUALPATH_L3_READ_PATH\", \"prefill\"\n"
    "                )\n"
)

DYNAMIC_READ_PATH_SITES = {
    "managers/scheduler.py": (
        (
            "            dualpath_decode_l3_read = (\n"
            "                " + _ENV_DECODE_READ +
            "                == \"decode\"\n"
            "            )\n",
            "            dualpath_decode_l3_read = dualpath_is_decode_read(req.rid)\n",
        ),
    ),
    "disaggregation/prefill.py": (
        (
            "        if (\n"
            "            __import__(\"os\").environ.get(\n"
            "                \"DUALPATH_L3_READ_PATH\", \"prefill\"\n"
            "            )\n"
            "            != \"decode\"\n"
            "        ):\n"
            "            return True\n",
            "        if not dualpath_is_decode_read(req.rid):\n"
            "            return True\n",
        ),
    ),
    "disaggregation/decode.py": (
        (
            "            decode_l3_read = (\n"
            "                " + _ENV_DECODE_READ +
            "                == \"decode\"\n"
            "            )\n",
            "            decode_l3_read = dualpath_is_decode_read(\n"
            "                decode_req.req.rid\n"
            "            )\n",
        ),
    ),
    "disaggregation/decode_kvcache_offload_manager.py": (
        (
            "        decode_l3_read = (\n"
            "            __import__(\"os\").environ.get(\n"
            "                \"DUALPATH_L3_READ_PATH\", \"prefill\"\n"
            "            )\n"
            "            == \"decode\"\n"
            "        )\n",
            "        decode_l3_read = dualpath_is_decode_read(req.rid)\n",
        ),
    ),
    # Startup-time capability, not a per-request choice: dynamic still needs
    # the Decode-side L3 machinery constructed.
    "disaggregation/mooncake/conn.py": (
        (
            "        self.decode_l3_read_path = (\n"
            "            os.environ.get(\"DUALPATH_L3_READ_PATH\", \"prefill\")\n"
            "            == \"decode\"\n"
            "        )\n",
            "        self.decode_l3_read_path = dualpath_decode_capable()\n",
        ),
    ),
}


def patch_dynamic_read_path_file(path: Path, rel: str) -> None:
    source = path.read_text()
    if "dualpath_read_path import" in source:
        print(f"Already patched: {path}")
        return
    for old, new in DYNAMIC_READ_PATH_SITES[rel]:
        source = replace_once(source, old, new, f"resolve read path per request in {rel}")
    names = ["is_decode_read as dualpath_is_decode_read"]
    if "dualpath_decode_capable()" in source:
        names.append("decode_capable as dualpath_decode_capable")
    imports = (
        "# DUALPATH_DYNAMIC_READ_PATH\n"
        "from sglang.srt.utils.dualpath_read_path import (\n"
        + "".join(f"    {n},\n" for n in names)
        + ")\n"
    )
    source = replace_once(
        source, "logger = logging.getLogger(__name__)\n",
        "logger = logging.getLogger(__name__)\n\n" + imports,
        f"import the read-path resolver into {rel}",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_decode_offload_prefill_path_boundary(source: str) -> str:
    """Skip the durable prefix on the Prefill read path too.

    The Prefill read path used to republish the whole aligned context every
    turn because Router's cached_tokens aggregates independent cache layers.
    With DUALPATH_NO_CROSS_TURN_KV_REUSE the Prefill radix namespace is
    request-private, so its only possible prefix hit is the L3 prefetch
    insert. req.cached_tokens (carried to Decode in the P-to-D metadata
    buffer) is then exactly the durable L3 prefix, and republishing it is
    pure write amplification.

    This runs in the Decode process, so the guard may only read variables
    that Decode actually has. DUALPATH_HICACHE_READ_ONLY is Prefill-scoped
    and pd_decode.sh unsets it; gating on it here silently disabled the
    whole branch. DUALPATH_NO_CROSS_TURN_KV_REUSE is exported to both sides
    and is the load-bearing condition on its own: a request-private radix
    namespace cannot produce a cross-turn local hit regardless of whether
    Prefill also writes back.
    """
    if "dualpath_prefill_l3_prefix_trusted" in source:
        return source
    return replace_once(
        source,
        "        cached_tokens = min(\n"
        "            max(\n"
        "                int(\n"
        "                    getattr(req, \"dualpath_decode_l3_hit_tokens\", 0)\n"
        "                    if decode_l3_read\n"
        "                    else 0\n"
        "                ),\n"
        "                0,\n"
        "            ),\n"
        "            len(req.origin_input_ids),\n"
        "        )\n",
        "        dualpath_prefill_l3_prefix_trusted = (\n"
        "            __import__(\"os\").environ.get(\n"
        "                \"DUALPATH_NO_CROSS_TURN_KV_REUSE\", \"1\"\n"
        "            ) == \"1\"\n"
        "        )\n"
        "        if decode_l3_read:\n"
        "            dualpath_durable_prefix = getattr(\n"
        "                req, \"dualpath_decode_l3_hit_tokens\", 0\n"
        "            )\n"
        "        elif dualpath_prefill_l3_prefix_trusted:\n"
        "            # Prefill's prefix hit can only come from its L3 restore.\n"
        "            dualpath_durable_prefix = getattr(req, \"cached_tokens\", 0)\n"
        "        else:\n"
        "            dualpath_durable_prefix = 0\n"
        "        cached_tokens = min(\n"
        "            max(int(dualpath_durable_prefix), 0),\n"
        "            len(req.origin_input_ids),\n"
        "        )\n",
        "trust the Prefill L3 prefix as a durable offload boundary",
    )


def patch_decode_owns_l3_offload(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_DECODE_OFFLOAD_START"
    if marker in source:
        source = patch_decode_offload_prefill_path_boundary(source)
        path.write_text(source)
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
        "        # Prefill is read-only and never publishes KV to L3. Router's\n"
        "        # cached_tokens aggregates independent cache layers and is not a\n"
        "        # durable-prefix boundary. On the Decode read path, skip only the\n"
        "        # prefix that Decode itself restored from L3. On the Prefill read\n"
        "        # path, conservatively republish the full aligned context.\n"
        "        all_tokens = req.origin_input_ids + req.output_ids[:-1]\n"
        "        decode_l3_read = (\n"
        "            __import__(\"os\").environ.get(\n"
        "                \"DUALPATH_L3_READ_PATH\", \"prefill\"\n"
        "            )\n"
        "            == \"decode\"\n"
        "        )\n"
        "        cached_tokens = min(\n"
        "            max(\n"
        "                int(\n"
        "                    getattr(req, \"dualpath_decode_l3_hit_tokens\", 0)\n"
        "                    if decode_l3_read\n"
        "                    else 0\n"
        "                ),\n"
        "                0,\n"
        "            ),\n"
        "            len(req.origin_input_ids),\n"
        "        )\n"
        "        prefill_offloaded_len = (\n"
        "            cached_tokens // self.page_size * self.page_size\n"
        "        )\n",
        "make Decode own newly computed KV persistence",
    )
    source = patch_decode_offload_prefill_path_boundary(source)
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
        "            if hasattr(self.tree_cache, \"dualpath_drop_request_cache\"):\n"
        "                self.tree_cache.dualpath_drop_request_cache(\n"
        "                    getattr(req, \"last_node\", None)\n"
        "                )\n"
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
        "                cache_node,\n"
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
        "            if hasattr(self.tree_cache, \"dualpath_drop_request_cache\"):\n"
        "                self.tree_cache.dualpath_drop_request_cache(cache_node)\n"
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
        "            getattr(req, \"last_node\", None),\n"
        "        )\n",
        "retain Decode offload range until L3 completion",
    )
    source = replace_once(
        source,
        "        self.tree_cache.protected_size_ -= len(req.prefix_indices)\n",
        "        # Keep this request's radix node protected until the Lustre\n"
        "        # backup ACK; the ACK handler removes it and decrements the\n"
        "        # lock exactly once.\n",
        "defer Decode radix release until durable backup",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_ttft_v3_cache_controller(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_TTFT_V3_CACHE_CONTROLLER"
    if marker in source:
        print(f"Already patched: {path}")
        return
    source = replace_once(
        source,
        "device_module = get_device_module()\n",
        "device_module = get_device_module()\n"
        "\n"
        "# DUALPATH_TTFT_V3_CACHE_CONTROLLER\n"
        "from sglang.srt.utils.dualpath_ttft import (\n"
        "    DETAIL as DUALPATH_TTFT_DETAIL,\n"
        "    ENABLED as DUALPATH_TTFT_ENABLED,\n"
        "    new_cuda_event as dualpath_new_cuda_event,\n"
        "    now_ns as dualpath_now_ns,\n"
        ")\n",
        "import TTFT V3 cache helpers",
    )
    source = replace_once(
        source,
        "        self.start_event = device_module.Event()  # start event on controller stream\n",
        "        self.start_event = device_module.Event()  # start event on controller stream\n"
        "        if DUALPATH_TTFT_ENABLED:\n"
        "            self.dualpath_start_event = dualpath_new_cuda_event(device_module)\n"
        "            self.dualpath_finish_event = dualpath_new_cuda_event(device_module)\n",
        "create constant-count H2D timing events",
    )
    source = replace_once(
        source,
        "    def complete(self, layer_index: int):\n"
        "        assert 0 <= layer_index < self._num_layers\n"
        "        self.load_events[layer_index].record()\n",
        "    def complete(self, layer_index: int):\n"
        "        assert 0 <= layer_index < self._num_layers\n"
        "        self.load_events[layer_index].record()\n"
        "        if (\n"
        "            layer_index == self._num_layers - 1\n"
        "            and getattr(self, \"dualpath_finish_event\", None) is not None\n"
        "        ):\n"
        "            self.dualpath_finish_event.record()\n",
        "finish aggregate H2D timing on the existing load stream",
    )
    source = replace_once(
        source,
        "            producer_event.start_event.wait(self.load_stream)\n"
        "            for i in range(self.layer_num):\n",
        "            producer_event.start_event.wait(self.load_stream)\n"
        "            if getattr(producer_event, \"dualpath_start_event\", None) is not None:\n"
        "                producer_event.dualpath_start_event.record(self.load_stream)\n"
        "            for i in range(self.layer_num):\n",
        "record aggregate H2D start on the existing stream",
    )
    source = replace_once(
        source,
        "        return producer_id\n"
        "\n"
        "    def evict_device",
        "        return producer_id\n"
        "\n"
        "    def dualpath_loading_timing_events(self, producer_id: int):\n"
        "        if not DUALPATH_TTFT_ENABLED or producer_id < 0:\n"
        "            return None, None\n"
        "        event = self.layer_done_counter.events[producer_id]\n"
        "        return event.dualpath_start_event, event.dualpath_finish_event\n"
        "\n"
        "    def evict_device",
        "expose aggregate H2D events without synchronization",
    )
    source = replace_once(
        source,
        "        self.dualpath_read_start = None\n"
        "        self.dualpath_read_end = None\n"
        "        self.dualpath_lustre_read_s = 0.0\n"
        "        self.dualpath_scatter_s = 0.0\n"
        "\n"
        "        super().__init__(host_indices, token_ids, last_hash, prefix_keys=prefix_keys)\n",
        "        self.dualpath_read_start = None\n"
        "        self.dualpath_read_end = None\n"
        "        self.dualpath_lustre_read_s = 0.0\n"
        "        self.dualpath_scatter_s = 0.0\n"
        "        if DUALPATH_TTFT_ENABLED:\n"
        "            self.dualpath_before_read_start_ns = 0\n"
        "            self.dualpath_before_read_end_ns = 0\n"
        "            self.dualpath_lustre_read_ns = 0\n"
        "            self.dualpath_scatter_ns = 0\n"
        "            self.dualpath_read_pages = 0\n"
        "\n"
        "        super().__init__(host_indices, token_ids, last_hash, prefix_keys=prefix_keys)\n"
        "\n"
        "    def dualpath_timing_snapshot(self):\n"
        "        if not DUALPATH_TTFT_ENABLED:\n"
        "            return {}\n"
        "        return {\n"
        "            \"before_read_start_ns\": self.dualpath_before_read_start_ns,\n"
        "            \"before_read_end_ns\": self.dualpath_before_read_end_ns,\n"
        "            \"before_read_ns\": max(\n"
        "                0,\n"
        "                self.dualpath_before_read_end_ns\n"
        "                - self.dualpath_before_read_start_ns,\n"
        "            ),\n"
        "            \"lustre_read_ns\": self.dualpath_lustre_read_ns,\n"
        "            \"scatter_ns\": self.dualpath_scatter_ns,\n"
        "            \"pages\": self.dualpath_read_pages,\n"
        "            \"tokens\": self.completed_tokens,\n"
        "            \"revoked\": self.revoked,\n"
        "        }\n",
        "add request-scoped prefetch timing snapshot",
    )
    generic_body = '''    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):
        # DUALPATH_O_DIRECT_REUSABLE_PAGE and TTFT V3: file reads remain
        # synchronous. Timing uses integer nanosecond accumulators and never
        # logs in the page or storage-batch loop.
        if DUALPATH_TTFT_DETAIL and operation.dualpath_read_start is None:
            operation.dualpath_read_start = time.perf_counter()
        if getattr(self.storage_backend, "dualpath_o_direct_read", False):
            data_page = getattr(self, "_dualpath_o_direct_page_buffer", None)
            if data_page is None:
                data_page = self.mem_pool_host.get_dummy_flat_data_page()
                alignment = getattr(
                    self.storage_backend, "dualpath_o_direct_alignment", 4096
                )
                if data_page.data_ptr() % alignment:
                    element_size = data_page.element_size()
                    extra_elements = (alignment + element_size - 1) // element_size
                    owner = torch.empty(
                        data_page.numel() + extra_elements,
                        dtype=data_page.dtype,
                        device=data_page.device,
                        pin_memory=data_page.is_pinned(),
                    )
                    offset_bytes = (-owner.data_ptr()) % alignment
                    if offset_bytes % element_size:
                        raise RuntimeError("Cannot align O_DIRECT Prefill page buffer")
                    offset = offset_bytes // element_size
                    data_page = owner[offset : offset + data_page.numel()]
                    self._dualpath_o_direct_page_buffer_owner = owner
                self._dualpath_o_direct_page_buffer = data_page

            for i, hash_value in enumerate(hash_values):
                read_start_ns = dualpath_now_ns() if DUALPATH_TTFT_ENABLED else 0
                page_data = self.storage_backend.get(hash_value, data_page)
                if DUALPATH_TTFT_ENABLED:
                    operation.dualpath_lustre_read_ns += (
                        dualpath_now_ns() - read_start_ns
                    )
                if page_data is None:
                    logger.warning(
                        "Prefetch operation %s failed to retrieve page %s.",
                        operation.request_id,
                        hash_value,
                    )
                    break
                scatter_start_ns = dualpath_now_ns() if DUALPATH_TTFT_ENABLED else 0
                self.mem_pool_host.set_from_flat_data_page(
                    host_indices[i * self.page_size], page_data
                )
                if DUALPATH_TTFT_ENABLED:
                    operation.dualpath_scatter_ns += (
                        dualpath_now_ns() - scatter_start_ns
                    )
                    operation.dualpath_read_pages += 1
                if not operation.increment(self.page_size):
                    break
            if DUALPATH_TTFT_DETAIL:
                operation.dualpath_read_end = time.perf_counter()
                operation.dualpath_lustre_read_s = (
                    operation.dualpath_lustre_read_ns / 1_000_000_000.0
                    if DUALPATH_TTFT_ENABLED else 0.0
                )
                operation.dualpath_scatter_s = (
                    operation.dualpath_scatter_ns / 1_000_000_000.0
                    if DUALPATH_TTFT_ENABLED else 0.0
                )
            return

        dummy_page_dst = [
            self.mem_pool_host.get_dummy_flat_data_page() for _ in hash_values
        ]
        read_start_ns = dualpath_now_ns() if DUALPATH_TTFT_ENABLED else 0
        page_data = self.storage_backend.batch_get(hash_values, dummy_page_dst)
        if DUALPATH_TTFT_ENABLED:
            operation.dualpath_lustre_read_ns += dualpath_now_ns() - read_start_ns
        if page_data is None:
            if DUALPATH_TTFT_DETAIL:
                operation.dualpath_read_end = time.perf_counter()
            return
        for i in range(len(hash_values)):
            if page_data[i] is None:
                logger.warning(
                    "Prefetch operation %s failed to retrieve page %s.",
                    operation.request_id,
                    hash_values[i],
                )
                break
            scatter_start_ns = dualpath_now_ns() if DUALPATH_TTFT_ENABLED else 0
            self.mem_pool_host.set_from_flat_data_page(
                host_indices[i * self.page_size], page_data[i]
            )
            if DUALPATH_TTFT_ENABLED:
                operation.dualpath_scatter_ns += dualpath_now_ns() - scatter_start_ns
                operation.dualpath_read_pages += 1
            if not operation.increment(self.page_size):
                break
        if DUALPATH_TTFT_DETAIL:
            operation.dualpath_read_end = time.perf_counter()
            operation.dualpath_lustre_read_s = (
                operation.dualpath_lustre_read_ns / 1_000_000_000.0
                if DUALPATH_TTFT_ENABLED else 0.0
            )
            operation.dualpath_scatter_s = (
                operation.dualpath_scatter_ns / 1_000_000_000.0
                if DUALPATH_TTFT_ENABLED else 0.0
            )
'''
    source = replace_function(
        source,
        "    def _generic_page_get(self, operation, hash_values, host_indices, extra_info=None):\n",
        generic_body,
        "replace generic page reader with aggregate V3 timing",
    )
    source = replace_once(
        source,
        "                operation = self.prefetch_queue.get(block=True, timeout=1)\n"
        "                if operation is None:\n"
        "                    continue\n"
        "\n"
        "                hash_value, storage_hit_count = self._storage_hit_query(operation)\n",
        "                operation = self.prefetch_queue.get(block=True, timeout=1)\n"
        "                if operation is None:\n"
        "                    continue\n"
        "                if DUALPATH_TTFT_ENABLED:\n"
        "                    operation.dualpath_before_read_start_ns = dualpath_now_ns()\n"
        "\n"
        "                hash_value, storage_hit_count = self._storage_hit_query(operation)\n",
        "start pre-read timing after worker dequeue",
    )
    source = replace_once(
        source,
        "                    storage_hit_count = storage_hit_count_tensor.item()\n"
        "                if __import__(\"os\").environ.get(\n"
        "                    \"DUALPATH_L3_HIT_DIAGNOSTIC\", \"1\"\n"
        "                ) == \"1\":\n"
        "                    logger.info(\n"
        "                        \"DUALPATH_L3_HIT_CONSENSUS rid=%s tp_rank=%d \"\n"
        "                        \"hit_tokens=%d threshold=%d\",\n"
        "                        operation.request_id,\n"
        "                        self.tp_rank,\n"
        "                        storage_hit_count,\n"
        "                        self.prefetch_threshold,\n"
        "                    )\n"
        "                operation.query_complete = True\n",
        "                    storage_hit_count = storage_hit_count_tensor.item()\n"
        "                if __import__(\"os\").environ.get(\n"
        "                    \"DUALPATH_L3_HIT_DIAGNOSTIC\", \"1\"\n"
        "                ) == \"1\":\n"
        "                    logger.info(\n"
        "                        \"DUALPATH_L3_HIT_CONSENSUS rid=%s tp_rank=%d \"\n"
        "                        \"hit_tokens=%d threshold=%d\",\n"
        "                        operation.request_id,\n"
        "                        self.tp_rank,\n"
        "                        storage_hit_count,\n"
        "                        self.prefetch_threshold,\n"
        "                    )\n"
        "                if DUALPATH_TTFT_ENABLED:\n"
        "                    operation.dualpath_before_read_end_ns = dualpath_now_ns()\n"
        "                operation.query_complete = True\n",
        "finish pre-read timing after TP hit consensus",
    )
    source = gate_logger_call(source, "DUALPATH_DECODE_L3_READ rid=%s")
    path.write_text(source)
    print(f"Patched: {path}")


def patch_ttft_v3_hiradix(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_TTFT_V3_HIRADIX"
    if marker in source:
        print(f"Already patched: {path}")
        return
    source = replace_once(
        source,
        "logger = logging.getLogger(__name__)\n",
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "# DUALPATH_TTFT_V3_HIRADIX\n"
        "from sglang.srt.utils.dualpath_ttft import (\n"
        "    DETAIL as DUALPATH_TTFT_DETAIL,\n"
        "    ENABLED as DUALPATH_TTFT_ENABLED,\n"
        ")\n",
        "import TTFT V3 radix helpers",
    )
    source = replace_once(
        source,
        "        self.dualpath_prefetch_start = {}\n"
        "        self.dualpath_prefetch_local_end = {}\n"
        "        self.ongoing_backup = {}\n",
        "        if DUALPATH_TTFT_DETAIL:\n"
        "            self.dualpath_prefetch_start = {}\n"
        "            self.dualpath_prefetch_local_end = {}\n"
        "        if DUALPATH_TTFT_ENABLED:\n"
        "            self.dualpath_completed_prefetch_timing = {}\n"
        "        self.ongoing_backup = {}\n",
        "initialize Prefill timing handoff only when enabled",
    )
    source = replace_once(
        source,
        "        self.dualpath_prefetch_start[req_id] = time.perf_counter()\n"
        "        operation = self.cache_controller.prefetch(\n",
        "        if DUALPATH_TTFT_DETAIL:\n"
        "            self.dualpath_prefetch_start[req_id] = time.perf_counter()\n"
        "        operation = self.cache_controller.prefetch(\n",
        "avoid legacy Prefill timing state when detail is disabled",
    )
    source = replace_once(
        source,
        "            self.dualpath_prefetch_start.pop(req_id, None)\n"
        "            self.dualpath_prefetch_local_end.pop(req_id, None)\n",
        "            if DUALPATH_TTFT_DETAIL:\n"
        "                self.dualpath_prefetch_start.pop(req_id, None)\n"
        "                self.dualpath_prefetch_local_end.pop(req_id, None)\n",
        "avoid legacy revoke timing when detail is disabled",
    )
    source = replace_once(
        source,
        "        if (\n"
        "            operation.completed_tokens == len(operation.hash_value) * self.page_size\n",
        "        if (\n"
        "            DUALPATH_TTFT_DETAIL\n"
        "            and operation.completed_tokens == len(operation.hash_value) * self.page_size\n",
        "gate legacy local completion timing",
    )
    source = replace_once(
        source,
        "            info = self.ongoing_prefetch.pop(req_id, None)\n"
        "            if info is not None:\n"
        "                last_host_node, token_ids, _, _, _ = info\n",
        "            info = self.ongoing_prefetch.pop(req_id, None)\n"
        "            if info is not None:\n"
        "                last_host_node, token_ids, _, operation, _ = info\n"
        "                if DUALPATH_TTFT_ENABLED:\n"
        "                    self.dualpath_completed_prefetch_timing[req_id] = (\n"
        "                        operation.dualpath_timing_snapshot()\n"
        "                    )\n",
        "preserve revoked Prefill pre-read timing",
    )
    source = replace_once(
        source,
        "        logger.debug(f\"Prefetch {req_id} completed with {completed_tokens} tokens\")\n"
        "        dualpath_start = self.dualpath_prefetch_start.pop(req_id, None)\n",
        "        logger.debug(f\"Prefetch {req_id} completed with {completed_tokens} tokens\")\n"
        "        if DUALPATH_TTFT_ENABLED:\n"
        "            self.dualpath_completed_prefetch_timing[req_id] = (\n"
        "                operation.dualpath_timing_snapshot()\n"
        "            )\n"
        "        dualpath_start = (\n"
        "            self.dualpath_prefetch_start.pop(req_id, None)\n"
        "            if DUALPATH_TTFT_DETAIL\n"
        "            else None\n"
        "        )\n",
        "preserve completed Prefill prefetch timing",
    )
    source = replace_once(
        source,
        "    def load_back(\n"
        "        self, node: TreeNode, mem_quota: Optional[int] = None\n"
        "    ) -> Optional[torch.Tensor]:\n",
        "    def dualpath_pop_prefetch_timing(self, request_id: str):\n"
        "        if not DUALPATH_TTFT_ENABLED:\n"
        "            return {}\n"
        "        return self.dualpath_completed_prefetch_timing.pop(request_id, {})\n"
        "\n"
        "    def load_back(\n"
        "        self, node: TreeNode, mem_quota: Optional[int] = None\n"
        "    ) -> Optional[torch.Tensor]:\n",
        "expose one-shot Prefill timing handoff",
    )
    source = gate_logger_call(source, "DUALPATH_L3_TIMING rid=%s")
    source = source.replace(
        "extra_key=last_host_node.key.extra_key",
        "extra_key=extra_key",
    )
    path.write_text(source)
    print(f"Patched: {path}")


def patch_ttft_v3_scheduler(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_TTFT_V3_SCHEDULER"
    if marker in source:
        print(f"Already patched: {path}")
        return
    source = replace_once(
        source,
        "logger = logging.getLogger(__name__)\n",
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "# DUALPATH_TTFT_V3_SCHEDULER\n"
        "from sglang.srt.utils.dualpath_ttft import (\n"
        "    DETAIL as DUALPATH_TTFT_DETAIL,\n"
        "    ENABLED as DUALPATH_TTFT_ENABLED,\n"
        "    new_cuda_event as dualpath_new_cuda_event,\n"
        ")\n",
        "import TTFT V3 scheduler helpers",
    )
    source = replace_once(
        source,
        "        new_batch.prepare_for_extend()\n"
        "\n"
        "        # Mixed-style chunked prefill\n",
        "        if DUALPATH_TTFT_ENABLED:\n"
        "            if self.enable_hierarchical_cache:\n"
        "                new_batch.dualpath_h2d_events = (\n"
        "                    self.tree_cache.cache_controller.dualpath_loading_timing_events(\n"
        "                        new_batch.hicache_consumer_index\n"
        "                    )\n"
        "                )\n"
        "            else:\n"
        "                new_batch.dualpath_h2d_events = (None, None)\n"
        "            if hasattr(self.tree_cache, \"dualpath_pop_prefetch_timing\"):\n"
        "                for dualpath_req in can_run_list:\n"
        "                    dualpath_prefetch_timing = (\n"
        "                        self.tree_cache.dualpath_pop_prefetch_timing(\n"
        "                            dualpath_req.rid\n"
        "                        )\n"
        "                    )\n"
        "                    # A chunked request enters can_run_list once per\n"
        "                    # chunk. Preserve the request-level L3 timing after\n"
        "                    # the first chunk consumes it from the cache.\n"
        "                    if dualpath_prefetch_timing:\n"
        "                        dualpath_req.dualpath_prefetch_timing = (\n"
        "                            dualpath_prefetch_timing\n"
        "                        )\n"
        "        new_batch.prepare_for_extend()\n"
        "\n"
        "        # Mixed-style chunked prefill\n",
        "associate H2D and prefetch timing with the EXTEND batch",
    )
    source = replace_once(
        source,
        "        # Run forward\n"
        "        if self.is_generation:\n",
        "        # Run forward\n"
        "        if DUALPATH_TTFT_ENABLED and batch.forward_mode == ForwardMode.EXTEND:\n"
        "            batch.dualpath_compute_start_event = dualpath_new_cuda_event(\n"
        "                self.device_module\n"
        "            )\n"
        "            batch.dualpath_compute_end_event = dualpath_new_cuda_event(\n"
        "                self.device_module\n"
        "            )\n"
        "            dualpath_compute_stream = (\n"
        "                self.forward_stream if self.enable_overlap else self.default_stream\n"
        "            )\n"
        "            batch.dualpath_compute_start_event.record(dualpath_compute_stream)\n"
        "        if self.is_generation:\n",
        "record one Prefill computation start event",
    )
    source = replace_once(
        source,
        "        # Capture prefill end time for EXTEND mode\n"
        "        if batch.forward_mode == ForwardMode.EXTEND:\n",
        "        if DUALPATH_TTFT_ENABLED and batch.forward_mode == ForwardMode.EXTEND:\n"
        "            dualpath_compute_stream = (\n"
        "                self.forward_stream if self.enable_overlap else self.default_stream\n"
        "            )\n"
        "            if getattr(ret, \"delay_sample_func\", None) is None:\n"
        "                batch.dualpath_compute_end_event.record(dualpath_compute_stream)\n"
        "            else:\n"
        "                ret.dualpath_delayed_compute_end_event = (\n"
        "                    batch.dualpath_compute_end_event\n"
        "                )\n"
        "            ret.dualpath_compute_events = (\n"
        "                batch.dualpath_compute_start_event,\n"
        "                batch.dualpath_compute_end_event,\n"
        "            )\n"
        "            ret.dualpath_h2d_events = batch.dualpath_h2d_events\n"
        "\n"
        "        # Capture prefill end time for EXTEND mode\n"
        "        if batch.forward_mode == ForwardMode.EXTEND:\n",
        "record one Prefill computation end event",
    )
    source = replace_once(
        source,
        "            _batch_result = batch_result.delay_sample_func()\n"
        "            assert _batch_result is batch_result\n",
        "            _batch_result = batch_result.delay_sample_func()\n"
        "            assert _batch_result is batch_result\n"
        "            if DUALPATH_TTFT_ENABLED and hasattr(\n"
        "                batch_result, \"dualpath_delayed_compute_end_event\"\n"
        "            ):\n"
        "                batch_result.dualpath_delayed_compute_end_event.record(\n"
        "                    self.forward_stream\n"
        "                )\n",
        "include delayed first-token sampling in Prefill computation",
    )
    source = gate_logger_call(source, "DUALPATH_PREFILL_PREP_BREAKDOWN rid=%s")
    path.write_text(source)
    print(f"Patched: {path}")


def patch_ttft_v3_prefill(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_TTFT_V3_PREFILL"
    if marker in source:
        print(f"Already patched: {path}")
        return
    source = replace_once(
        source,
        "logger = logging.getLogger(__name__)\n",
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "# DUALPATH_TTFT_V3_PREFILL\n"
        "from sglang.srt.utils.dualpath_ttft import (\n"
        "    DETAIL as DUALPATH_TTFT_DETAIL,\n"
        "    ENABLED as DUALPATH_TTFT_ENABLED,\n"
        "    emit as dualpath_ttft_emit,\n"
        "    event_overlap_seconds as dualpath_event_overlap_seconds,\n"
        "    event_seconds as dualpath_event_seconds,\n"
        "    now_ns as dualpath_now_ns,\n"
        "    ns_seconds as dualpath_ns_seconds,\n"
        ")\n",
        "import TTFT V3 Prefill helpers",
    )
    source = replace_once(
        source,
        "        if copy_done is not None:\n"
        "            copy_done.synchronize()\n"
        "\n"
        "        logprob_pt = 0\n",
        "        if copy_done is not None:\n"
        "            copy_done.synchronize()\n"
        "\n"
        "        if DUALPATH_TTFT_ENABLED:\n"
        "            dualpath_pipeline_end_ns = dualpath_now_ns()\n"
        "            dualpath_compute_events = getattr(\n"
        "                result, \"dualpath_compute_events\", (None, None)\n"
        "            )\n"
        "            dualpath_h2d_events = getattr(\n"
        "                result, \"dualpath_h2d_events\", (None, None)\n"
        "            )\n"
        "            for dualpath_req in batch.reqs:\n"
        "                dualpath_req.dualpath_compute_event_pairs = getattr(\n"
        "                    dualpath_req, \"dualpath_compute_event_pairs\", []\n"
        "                )\n"
        "                dualpath_req.dualpath_compute_event_pairs.append(\n"
        "                    dualpath_compute_events\n"
        "                )\n"
        "                dualpath_req.dualpath_h2d_event_pairs = getattr(\n"
        "                    dualpath_req, \"dualpath_h2d_event_pairs\", []\n"
        "                )\n"
        "                dualpath_req.dualpath_h2d_event_pairs.append(\n"
        "                    dualpath_h2d_events\n"
        "                )\n"
        "                dualpath_req.dualpath_prefill_pipeline_end_ns = (\n"
        "                    dualpath_pipeline_end_ns\n"
        "                )\n"
        "\n"
        "        logprob_pt = 0\n",
        "retain completed CUDA event pairs after existing synchronization",
    )
    source = replace_once(
        source,
        "        if not hasattr(req, \"dualpath_transfer_start\"):\n"
        "            req.dualpath_transfer_start = time.perf_counter()\n"
        "            req.dualpath_transfer_logical_tokens = 0\n"
        "            req.dualpath_transfer_page_count = 0\n"
        "            req.dualpath_transfer_page_size = page_size\n"
        "        req.dualpath_transfer_logical_tokens += max(0, end_idx - start_idx)\n"
        "        req.dualpath_transfer_page_count += len(page_indices)\n",
        "        if DUALPATH_TTFT_ENABLED or DUALPATH_TTFT_DETAIL:\n"
        "            if not hasattr(req, \"dualpath_transfer_page_count\"):\n"
        "                if DUALPATH_TTFT_DETAIL:\n"
        "                    req.dualpath_transfer_start = time.perf_counter()\n"
        "                if DUALPATH_TTFT_ENABLED:\n"
        "                    req.dualpath_transfer_start_ns = dualpath_now_ns()\n"
        "                req.dualpath_transfer_logical_tokens = 0\n"
        "                req.dualpath_transfer_page_count = 0\n"
        "                req.dualpath_transfer_page_size = page_size\n"
        "            req.dualpath_transfer_logical_tokens += max(0, end_idx - start_idx)\n"
        "            req.dualpath_transfer_page_count += len(page_indices)\n",
        "record integer P-to-D transfer start",
    )
    source = replace_once(
        source,
        "            elif poll == KVPoll.Success:  # transfer done\n"
        "                dualpath_transfer_end = time.perf_counter()\n",
        "            elif poll == KVPoll.Success:  # transfer done\n"
        "                dualpath_transfer_end = (\n"
        "                    time.perf_counter() if DUALPATH_TTFT_DETAIL else 0.0\n"
        "                )\n"
        "                if DUALPATH_TTFT_ENABLED:\n"
        "                    dualpath_transfer_end_ns = dualpath_now_ns()\n"
        "                    dualpath_transfer_start_ns = getattr(\n"
        "                        req,\n"
        "                        \"dualpath_transfer_start_ns\",\n"
        "                        dualpath_transfer_end_ns,\n"
        "                    )\n"
        "                    dualpath_p2d_raw = dualpath_ns_seconds(\n"
        "                        dualpath_transfer_end_ns\n"
        "                        - dualpath_transfer_start_ns\n"
        "                    )\n"
        "                    dualpath_pipeline_end_ns = getattr(\n"
        "                        req,\n"
        "                        \"dualpath_prefill_pipeline_end_ns\",\n"
        "                        dualpath_transfer_start_ns,\n"
        "                    )\n"
        "                    dualpath_p2d_overlap = dualpath_ns_seconds(\n"
        "                        min(\n"
        "                            dualpath_transfer_end_ns,\n"
        "                            max(\n"
        "                                dualpath_transfer_start_ns,\n"
        "                                dualpath_pipeline_end_ns,\n"
        "                            ),\n"
        "                        )\n"
        "                        - dualpath_transfer_start_ns\n"
        "                    )\n"
        "                    dualpath_prefetch = getattr(\n"
        "                        req, \"dualpath_prefetch_timing\", {}\n"
        "                    )\n"
        "                    dualpath_h2d_raw = 0.0\n"
        "                    dualpath_compute_raw = 0.0\n"
        "                    dualpath_compute_overlap = 0.0\n"
        "                    dualpath_h2d_pairs = getattr(\n"
        "                        req, \"dualpath_h2d_event_pairs\", []\n"
        "                    )\n"
        "                    dualpath_compute_pairs = getattr(\n"
        "                        req, \"dualpath_compute_event_pairs\", []\n"
        "                    )\n"
        "                    for pair_index, dualpath_compute_pair in enumerate(\n"
        "                        dualpath_compute_pairs\n"
        "                    ):\n"
        "                        dualpath_compute_raw += dualpath_event_seconds(\n"
        "                            *dualpath_compute_pair\n"
        "                        )\n"
        "                        if pair_index < len(dualpath_h2d_pairs):\n"
        "                            dualpath_h2d_pair = dualpath_h2d_pairs[pair_index]\n"
        "                            dualpath_h2d_raw += dualpath_event_seconds(\n"
        "                                *dualpath_h2d_pair\n"
        "                            )\n"
        "                            dualpath_compute_overlap += (\n"
        "                                dualpath_event_overlap_seconds(\n"
        "                                    *dualpath_h2d_pair,\n"
        "                                    *dualpath_compute_pair,\n"
        "                                )\n"
        "                            )\n"
        "                    dualpath_ttft_emit(\n"
        "                        logger,\n"
        "                        rid=req.rid,\n"
        "                        side=\"prefill\",\n"
        "                        tp_rank=self.attn_tp_rank,\n"
        "                        values={\n"
        "                            \"prefill_before_lustre_read_raw_seconds\": dualpath_ns_seconds(dualpath_prefetch.get(\"before_read_ns\", 0)),\n"
        "                            \"prefill_lustre_read_raw_seconds\": dualpath_ns_seconds(dualpath_prefetch.get(\"lustre_read_ns\", 0)),\n"
        "                            \"prefill_scatter_raw_seconds\": dualpath_ns_seconds(dualpath_prefetch.get(\"scatter_ns\", 0)),\n"
        "                            \"prefill_host_to_device_raw_seconds\": dualpath_h2d_raw,\n"
        "                            # Keep the full CUDA event interval under\n"
        "                            # prefill_computation on both read paths.\n"
        "                            # Attribute H2D/compute overlap to H2D's\n"
        "                            # critical-stage value so accounting stays\n"
        "                            # additive without changing the computation\n"
        "                            # region being measured.\n"
        "                            \"prefill_host_to_device_seconds\": max(\n"
        "                                0.0, dualpath_h2d_raw - dualpath_compute_overlap\n"
        "                            ),\n"
        "                            \"prefill_host_to_device_overlap_removed_seconds\": dualpath_compute_overlap,\n"
        "                            \"prefill_computation_raw_seconds\": dualpath_compute_raw,\n"
        "                            \"prefill_computation_seconds\": dualpath_compute_raw,\n"
        "                            \"prefill_computation_overlap_removed_seconds\": 0.0,\n"
        "                            \"p_to_d_transfer_raw_seconds\": dualpath_p2d_raw,\n"
        "                            \"p_to_d_transfer_seconds\": max(0.0, dualpath_p2d_raw - dualpath_p2d_overlap),\n"
        "                            \"p_to_d_transfer_overlap_removed_seconds\": dualpath_p2d_overlap,\n"
        "                            \"prefetch_before_read_start_ns\": dualpath_prefetch.get(\"before_read_start_ns\", 0),\n"
        "                            \"prefetch_before_read_end_ns\": dualpath_prefetch.get(\"before_read_end_ns\", 0),\n"
        "                            \"p_to_d_start_ns\": dualpath_transfer_start_ns,\n"
        "                            \"p_to_d_end_ns\": dualpath_transfer_end_ns,\n"
        "                            \"prefill_pipeline_end_ns\": dualpath_pipeline_end_ns,\n"
        "                            \"read_tokens\": int(dualpath_prefetch.get(\"tokens\", 0)),\n"
        "                            \"read_pages\": int(dualpath_prefetch.get(\"pages\", 0)),\n"
        "                            \"p_to_d_logical_tokens\": int(getattr(req, \"dualpath_transfer_logical_tokens\", 0)),\n"
        "                            \"p_to_d_pages\": int(getattr(req, \"dualpath_transfer_page_count\", 0)),\n"
        "                            \"p_to_d_page_size\": int(getattr(req, \"dualpath_transfer_page_size\", 0)),\n"
        "                            \"p_to_d_page_aligned_tokens\": int(getattr(req, \"dualpath_transfer_page_count\", 0) * getattr(req, \"dualpath_transfer_page_size\", 0)),\n"
        "                            \"queue_wait_raw_seconds\": dualpath_ns_seconds(dualpath_queue_end_ns - dualpath_queue_start_ns),\n"
        "                            \"queue_wait_start_ns\": dualpath_queue_start_ns,\n"
        "                            \"queue_wait_end_ns\": dualpath_queue_end_ns,\n"
        "                        },\n"
        "                    )\n",
        "emit one Prefill TTFT V3 record per request and rank",
    )
    source = replace_once(
        source,
        "                    dualpath_ttft_emit(\n"
        "                        logger,\n"
        "                        rid=req.rid,\n"
        "                        side=\"prefill\",\n",
        "                    (\n"
        "                        dualpath_queue_start_ns,\n"
        "                        dualpath_queue_end_ns,\n"
        "                    ) = dualpath_queue_window_ns(\n"
        "                        req.time_stats,\n"
        "                        \"prefill_bootstrap_queue_entry_time\",\n"
        "                        \"forward_entry_time\",\n"
        "                    )\n"
        "                    dualpath_ttft_emit(\n"
        "                        logger,\n"
        "                        rid=req.rid,\n"
        "                        side=\"prefill\",\n",
        "compute the Prefill queue-wait window before emitting",
    )
    source = replace_once(
        source,
        "    emit as dualpath_ttft_emit,\n"
        "    event_overlap_seconds as dualpath_event_overlap_seconds,\n",
        "    emit as dualpath_ttft_emit,\n"
        "    event_overlap_seconds as dualpath_event_overlap_seconds,\n"
        "    queue_window_ns as dualpath_queue_window_ns,\n",
        "import the queue-wait window helper into Prefill",
    )
    for log_marker in (
        "DUALPATH_TTFT_P_TRANSFER_V2 rid=%s",
        "DUALPATH_D2P_REVERSE rid=%s",
    ):
        source = gate_logger_call(source, log_marker)
    path.write_text(source)
    print(f"Patched: {path}")


def patch_ttft_v3_mooncake(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_TTFT_V3_MOONCAKE"
    if marker in source:
        print(f"Already patched: {path}")
        return
    source = replace_once(
        source,
        "logger = logging.getLogger(__name__)\n",
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "# DUALPATH_TTFT_V3_MOONCAKE\n"
        "from sglang.srt.utils.dualpath_ttft import (\n"
        "    DETAIL as DUALPATH_TTFT_DETAIL,\n"
        "    ENABLED as DUALPATH_TTFT_ENABLED,\n"
        "    now_ns as dualpath_now_ns,\n"
        ")\n",
        "import TTFT V3 Mooncake helpers",
    )
    source = replace_once(
        source,
        "        self.decode_l3_hit_tokens = 0\n"
        "        self.dualpath_prefetch_timing = {}\n"
        "        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)\n",
        "        self.decode_l3_hit_tokens = 0\n"
        "        self.dualpath_prefetch_timing = {}\n"
        "        if DUALPATH_TTFT_ENABLED:\n"
        "            self.dualpath_request_id = None\n"
        "            self.dualpath_d2p_ns = 0\n"
        "            self.dualpath_d2p_pages = 0\n"
        "        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)\n",
        "initialize Decode receiver timing only when enabled",
    )
    source = replace_once(
        source,
        "        self.decode_aux_index = aux_index\n"
        "        self.decode_state_indices = state_indices\n",
        "        self.decode_aux_index = aux_index\n"
        "        self.decode_state_indices = state_indices\n"
        "        if DUALPATH_TTFT_ENABLED:\n"
        "            self.dualpath_request_id = request_id\n",
        "retain the real request id on Decode",
    )
    source = replace_once(
        source,
        "        start = time.perf_counter()\n"
        "        with torch.cuda.stream(self.decode_h2d_stream):\n",
        "        start = (\n"
        "            time.perf_counter()\n"
        "            if DUALPATH_TTFT_ENABLED or DUALPATH_TTFT_DETAIL\n"
        "            else 0.0\n"
        "        )\n"
        "        with torch.cuda.stream(self.decode_h2d_stream):\n",
        "gate Decode H2D start timing",
    )
    source = replace_once(
        source,
        "        end = time.perf_counter()\n"
        "        logger.info(\n"
        "            \"DUALPATH_DECODE_HOST_STAGING_COPY tp_rank=%d pages=%d \"\n",
        "        end = (\n"
        "            time.perf_counter()\n"
        "            if DUALPATH_TTFT_ENABLED or DUALPATH_TTFT_DETAIL\n"
        "            else 0.0\n"
        "        )\n"
        "        logger.info(\n"
        "            \"DUALPATH_DECODE_HOST_STAGING_COPY tp_rank=%d pages=%d \"\n",
        "gate Decode H2D end timing",
    )
    source = replace_once(
        source,
        "            reverse_start = time.perf_counter()\n"
        "            if not hasattr(self.kv_args, \"prefill_start_layer\"):\n",
        "            reverse_start = (\n"
        "                time.perf_counter() if DUALPATH_TTFT_DETAIL else 0.0\n"
        "            )\n"
        "            reverse_start_ns = (\n"
        "                dualpath_now_ns() if DUALPATH_TTFT_ENABLED else 0\n"
        "            )\n"
        "            if not hasattr(self.kv_args, \"prefill_start_layer\"):\n",
        "start Decode-to-Prefill timing",
    )
    source = replace_once(
        source,
        "            reverse_end = time.perf_counter()\n"
        "            status = KVPoll.Success if ret == 0 else KVPoll.Failed\n",
        "            reverse_end = (\n"
        "                time.perf_counter() if DUALPATH_TTFT_DETAIL else 0.0\n"
        "            )\n"
        "            if DUALPATH_TTFT_ENABLED:\n"
        "                receiver.dualpath_d2p_ns = dualpath_now_ns() - reverse_start_ns\n"
        "                receiver.dualpath_d2p_pages = len(dst_indices)\n"
        "            status = KVPoll.Success if ret == 0 else KVPoll.Failed\n",
        "finish Decode-to-Prefill timing on the Decode node",
    )
    source = replace_once(
        source,
        "                \"DUALPATH_DECODE_L3_REVERSE rid_room=%d \"\n"
        "                \"decode_engine=%s node=%s tp_rank=%d pages=%d \"\n",
        "                \"DUALPATH_DECODE_L3_REVERSE rid=%s rid_room=%d \"\n"
        "                \"decode_engine=%s node=%s tp_rank=%d pages=%d \"\n",
        "add the real rid to the legacy reverse diagnostic",
    )
    source = replace_once(
        source,
        "                room,\n"
        "                os.environ.get(\"DECODE_ID\", \"0\"),\n",
        "                getattr(receiver, \"dualpath_request_id\", None) or str(room),\n"
        "                room,\n"
        "                os.environ.get(\"DECODE_ID\", \"0\"),\n",
        "format the real reverse request id",
    )
    for log_marker in (
        "DUALPATH_DECODE_HOST_STAGING_COPY tp_rank=%d",
        "DUALPATH_DECODE_L3_REVERSE rid=%s",
    ):
        source = gate_logger_call(source, log_marker)
    path.write_text(source)
    print(f"Patched: {path}")


def patch_ttft_v3_decode(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_TTFT_V3_DECODE"
    if marker in source:
        print(f"Already patched: {path}")
        return
    source = replace_once(
        source,
        "logger = logging.getLogger(__name__)\n",
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "# DUALPATH_TTFT_V3_DECODE\n"
        "from sglang.srt.utils.dualpath_ttft import (\n"
        "    DETAIL as DUALPATH_TTFT_DETAIL,\n"
        "    ENABLED as DUALPATH_TTFT_ENABLED,\n"
        "    now_ns as dualpath_now_ns,\n"
        ")\n",
        "import TTFT V3 Decode helpers",
    )
    source = replace_once(
        source,
        "            self._pre_alloc(decode_req.req)\n"
        "\n"
        "            kv_indices = (\n",
        "            dualpath_prealloc_start_ns = (\n"
        "                dualpath_now_ns() if DUALPATH_TTFT_ENABLED else 0\n"
        "            )\n"
        "            self._pre_alloc(decode_req.req)\n"
        "            if DUALPATH_TTFT_ENABLED:\n"
        "                decode_req.req.dualpath_preallocation_ns = (\n"
        "                    dualpath_now_ns() - dualpath_prealloc_start_ns\n"
        "                )\n"
        "\n"
        "            kv_indices = (\n",
        "time exactly the Decode _pre_alloc call",
    )
    source = replace_once(
        source,
        "        decode_req.req.time_stats.wait_queue_entry_time = time.perf_counter()\n",
        "        decode_req.req.time_stats.wait_queue_entry_time = time.perf_counter()\n"
        "        if DUALPATH_TTFT_ENABLED:\n"
        "            decode_req.req.dualpath_decode_ready_ns = dualpath_now_ns()\n",
        "record Decode dispatch-ready boundary",
    )
    source = replace_once(
        source,
        "                self._commit_transfer_to_req(decode_req)\n"
        "                req = decode_req.req\n",
        "                self._commit_transfer_to_req(decode_req)\n"
        "                req = decode_req.req\n"
        "                # Keep the durable L3 prefix separate from Router's\n"
        "                # aggregate cached_tokens counter. Decode offload may\n"
        "                # skip only pages that this L3 prefetch actually read.\n"
        "                req.dualpath_decode_l3_hit_tokens = int(\n"
        "                    getattr(\n"
        "                        dualpath_host_staging_receiver,\n"
        "                        \"decode_l3_hit_tokens\",\n"
        "                        0,\n"
        "                    )\n"
        "                )\n"
        "                if DUALPATH_TTFT_ENABLED:\n"
        "                    dualpath_prefetch = getattr(\n"
        "                        dualpath_host_staging_receiver,\n"
        "                        \"dualpath_prefetch_timing\",\n"
        "                        {},\n"
        "                    )\n"
        "                    req.dualpath_decode_timing = {\n"
        "                        \"before_read_ns\": dualpath_prefetch.get(\"before_read_ns\", 0),\n"
        "                        \"before_read_start_ns\": dualpath_prefetch.get(\"before_read_start_ns\", 0),\n"
        "                        \"before_read_end_ns\": dualpath_prefetch.get(\"before_read_end_ns\", 0),\n"
        "                        \"lustre_read_ns\": dualpath_prefetch.get(\"lustre_read_ns\", 0),\n"
        "                        \"scatter_ns\": dualpath_prefetch.get(\"scatter_ns\", 0),\n"
        "                        \"read_tokens\": dualpath_prefetch.get(\"tokens\", 0),\n"
        "                        \"read_pages\": dualpath_prefetch.get(\"pages\", 0),\n"
        "                        \"d2p_ns\": getattr(dualpath_host_staging_receiver, \"dualpath_d2p_ns\", 0),\n"
        "                        \"d2p_pages\": getattr(dualpath_host_staging_receiver, \"dualpath_d2p_pages\", 0),\n"
        "                        \"h2d_seconds\": max(\n"
        "                            0.0,\n"
        "                            float(getattr(dualpath_host_staging_receiver, \"host_staging_h2d_end\", 0) or 0)\n"
        "                            - float(getattr(dualpath_host_staging_receiver, \"host_staging_h2d_start\", 0) or 0),\n"
        "                        ),\n"
        "                        \"host_allocation_seconds_diagnostic\": max(\n"
        "                            0.0,\n"
        "                            float(getattr(dualpath_host_staging_receiver, \"dualpath_host_alloc_end\", 0) or 0)\n"
        "                            - float(getattr(dualpath_host_staging_receiver, \"dualpath_host_alloc_start\", 0) or 0),\n"
        "                        ),\n"
        "                    }\n",
        "copy Decode timing before receiver cleanup",
    )
    for log_marker in (
        "DUALPATH_TTFT_D_RECEIVE rid=%s",
        "DUALPATH_TTFT_D_HOST_STAGING rid=%s",
        "DUALPATH_DECODE_HANDOFF_BREAKDOWN rid=%s",
        "DUALPATH_DECODE_PREALLOC_INNER rid=%s",
        "DUALPATH_DECODE_ALLOCATOR_MERGE rid=%s",
    ):
        source = gate_logger_call(source, log_marker)
    path.write_text(source)
    print(f"Patched: {path}")


def patch_ttft_v3_dispatch(path: Path) -> None:
    source = path.read_text()
    marker = "DUALPATH_TTFT_V3_DISPATCH"
    if marker in source:
        print(f"Already patched: {path}")
        return
    source = replace_once(
        source,
        "logger = logging.getLogger(__name__)\n",
        "logger = logging.getLogger(__name__)\n"
        "\n"
        "# DUALPATH_TTFT_V3_DISPATCH\n"
        "from sglang.srt.utils.dualpath_ttft import (\n"
        "    DETAIL as DUALPATH_TTFT_DETAIL,\n"
        "    ENABLED as DUALPATH_TTFT_ENABLED,\n"
        "    emit as dualpath_ttft_emit,\n"
        "    now_ns as dualpath_now_ns,\n"
        "    ns_seconds as dualpath_ns_seconds,\n"
        "    queue_window_ns as dualpath_queue_window_ns,\n"
        ")\n",
        "import TTFT V3 dispatch helpers",
    )
    source = replace_once(
        source,
        "        dualpath_dispatch_time = time.perf_counter()\n"
        "        if self.attn_tp_rank == 0:\n",
        "        dualpath_dispatch_time = (\n"
        "            time.perf_counter() if DUALPATH_TTFT_DETAIL else 0.0\n"
        "        )\n"
        "        if DUALPATH_TTFT_ENABLED:\n"
        "            dualpath_dispatch_ns = dualpath_now_ns()\n"
        "            for req in batch.reqs:\n"
        "                dualpath_decode = getattr(req, \"dualpath_decode_timing\", {})\n"
        "                dualpath_dispatch_raw = dualpath_ns_seconds(\n"
        "                    dualpath_dispatch_ns\n"
        "                    - getattr(req, \"dualpath_decode_ready_ns\", dualpath_dispatch_ns)\n"
        "                )\n"
        "                (\n"
        "                    dualpath_queue_start_ns,\n"
        "                    dualpath_queue_end_ns,\n"
        "                ) = dualpath_queue_window_ns(\n"
        "                    req.time_stats,\n"
        "                    \"decode_prealloc_queue_entry_time\",\n"
        "                    \"decode_transfer_queue_entry_time\",\n"
        "                )\n"
        "                dualpath_ttft_emit(\n"
        "                    logger,\n"
        "                    rid=req.rid,\n"
        "                    side=\"decode\",\n"
        "                    tp_rank=self.attn_tp_rank,\n"
        "                    values={\n"
        "                        \"decode_preallocation_raw_seconds\": dualpath_ns_seconds(getattr(req, \"dualpath_preallocation_ns\", 0)),\n"
        "                        \"decode_before_lustre_read_raw_seconds\": dualpath_ns_seconds(dualpath_decode.get(\"before_read_ns\", 0)),\n"
        "                        \"decode_lustre_read_raw_seconds\": dualpath_ns_seconds(dualpath_decode.get(\"lustre_read_ns\", 0)),\n"
        "                        \"decode_scatter_raw_seconds\": dualpath_ns_seconds(dualpath_decode.get(\"scatter_ns\", 0)),\n"
        "                        \"d_to_p_transfer_raw_seconds\": dualpath_ns_seconds(dualpath_decode.get(\"d2p_ns\", 0)),\n"
        "                        \"decode_host_to_device_raw_seconds\": float(dualpath_decode.get(\"h2d_seconds\", 0.0)),\n"
        "                        \"decode_dispatch_raw_seconds\": dualpath_dispatch_raw if self.attn_tp_rank == 0 else 0.0,\n"
        "                        \"decode_before_read_start_ns\": dualpath_decode.get(\"before_read_start_ns\", 0),\n"
        "                        \"decode_before_read_end_ns\": dualpath_decode.get(\"before_read_end_ns\", 0),\n"
        "                        \"decode_dispatch_ns\": dualpath_dispatch_ns,\n"
        "                        \"read_tokens\": int(dualpath_decode.get(\"read_tokens\", 0)),\n"
        "                        \"read_pages\": int(dualpath_decode.get(\"read_pages\", 0)),\n"
        "                        \"d_to_p_pages\": int(dualpath_decode.get(\"d2p_pages\", 0)),\n"
        "                        \"decode_host_allocation_seconds_diagnostic\": float(dualpath_decode.get(\"host_allocation_seconds_diagnostic\", 0.0)),\n"
        "                        \"queue_wait_raw_seconds\": dualpath_ns_seconds(dualpath_queue_end_ns - dualpath_queue_start_ns),\n"
        "                        \"queue_wait_start_ns\": dualpath_queue_start_ns,\n"
        "                        \"queue_wait_end_ns\": dualpath_queue_end_ns,\n"
        "                    },\n"
        "                )\n"
        "        if self.attn_tp_rank == 0:\n",
        "emit one Decode TTFT V3 record per request and rank",
    )
    source = gate_logger_call(source, "DUALPATH_TTFT_D_DISPATCH rid=%s")
    path.write_text(source)
    print(f"Patched: {path}")


def patch_ttft_v3_detail_logs(path: Path, markers: tuple[str, ...]) -> None:
    source = path.read_text()
    detail_import = "    DETAIL as DUALPATH_TTFT_DETAIL,\n"
    if detail_import not in source:
        source = replace_once(
            source,
            "logger = logging.getLogger(__name__)\n",
            "logger = logging.getLogger(__name__)\n"
            + "from sglang.srt.utils.dualpath_ttft import (\n"
            + detail_import
            + ")\n",
            f"import TTFT detail flag in {path.name}",
        )
    for log_marker in markers:
        source = gate_logger_call(source, log_marker)
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
    write_read_path_helper(srt / "utils/dualpath_read_path.py")
    write_ttft_v3_helper(srt / "utils/dualpath_ttft.py")
    patch_hicache_file_o_direct_read(srt / "mem_cache/hicache_storage.py")
    patch_prefill_o_direct_reusable_buffer(
        srt / "managers/cache_controller.py"
    )
    patch_shared_prefetch_progress(srt / "managers/cache_controller.py")
    patch_hiradix(srt / "mem_cache/hiradix_cache.py")
    patch_decode_l3_scheduler(srt / "managers/scheduler.py")
    patch_decode_l3_schedule_policy(srt / "managers/schedule_policy.py")
    patch_decode_l3_runtime_checker(
        srt / "managers/scheduler_runtime_checker_mixin.py"
    )
    patch_schedule_batch(srt / "managers/schedule_batch.py")
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
    patch_ttft_v3_cache_controller(srt / "managers/cache_controller.py")
    patch_ttft_v3_hiradix(srt / "mem_cache/hiradix_cache.py")
    patch_ttft_v3_scheduler(srt / "managers/scheduler.py")
    patch_ttft_v3_prefill(srt / "disaggregation/prefill.py")
    patch_ttft_v3_mooncake(srt / "disaggregation/mooncake/conn.py")
    patch_ttft_v3_decode(srt / "disaggregation/decode.py")
    patch_ttft_v3_dispatch(
        srt / "managers/scheduler_output_processor_mixin.py"
    )
    patch_ttft_v3_detail_logs(
        srt / "managers/schedule_batch.py",
        ("DUALPATH_REQ_TIMING rid=%s",),
    )
    # Last: rewrite the process-level read-path switches into per-request
    # decisions, after every other patch has settled their surrounding code.
    for rel in DYNAMIC_READ_PATH_SITES:
        patch_dynamic_read_path_file(srt / rel, rel)


if __name__ == "__main__":
    main()
