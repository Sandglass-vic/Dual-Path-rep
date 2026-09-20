#!/usr/bin/env python3
"""Synthetic benchmark for DualPath's Decode-side Lustre page reads."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import mmap
import multiprocessing as mp
import os
import queue
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cluster_barrier import synchronize, wait_until  # noqa: E402

RESULT_PREFIX = "KVIO_RESULT "
CACHE_PROBE_PREFIX = "KVIO_CACHE_PROBE "


def page_path(root: Path, rank: int, page: int) -> Path:
    # The production backend uses one flat directory of hash-named .bin files.
    # Deterministic names avoid a manifest lookup while preserving open/read/close.
    return root / f"{page:08x}-{rank:02x}-dualpath-kv-page.bin"


def allocate_buffer(
    size: int, buffer_type: str, alignment: int = 1
) -> tuple[Any, memoryview]:
    """Allocate an I/O target whose address is a multiple of `alignment`.

    O_DIRECT rejects unaligned user buffers, and a plain bytearray lands wherever
    the allocator put its object header, so the padded slice below is what makes
    the direct phases runnable outside the Torch path.
    """
    if buffer_type == "pinned":
        try:
            import torch

            tensor = torch.empty(size + alignment, dtype=torch.uint8, pin_memory=True)
            view = memoryview(tensor.numpy())
            offset = (-buffer_address(view)) % alignment
            return tensor, view[offset : offset + size]
        except Exception as exc:  # pragma: no cover - depends on runtime image
            raise RuntimeError(
                "Pinned allocation failed. Run in the benchmark image with a GPU "
                "assigned, or set BUFFER_TYPE=pageable."
            ) from exc

    if alignment > 1:
        # An anonymous mapping is page aligned, and the pad keeps the slice
        # aligned for alignments larger than one page.
        owner: Any = mmap.mmap(-1, size + alignment)
        view = memoryview(owner)
        offset = (-buffer_address(view)) % alignment
        return owner, view[offset : offset + size]

    owner = bytearray(size)
    return owner, memoryview(owner)


def write_full(fd: int, data: memoryview) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise OSError("short write while preparing KV page")
        offset += written


def prepare(args: argparse.Namespace) -> None:
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    expected_bytes = args.tp_ranks * args.pages_per_rank * args.page_bytes
    marker = root / "dataset.json"
    if marker.exists() and not args.force:
        metadata = json.loads(marker.read_text())
        expected = {
            "tp_ranks": args.tp_ranks,
            "pages_per_rank": args.pages_per_rank,
            "page_bytes": args.page_bytes,
        }
        if all(metadata.get(key) == value for key, value in expected.items()):
            print(
                f"PREPARE_REUSE data_dir={root} total_gib={expected_bytes / 2**30:.3f}",
                flush=True,
            )
            return
        raise RuntimeError(
            f"{marker} describes a different dataset; use --force or a new directory"
        )

    if args.force:
        for path in root.glob("*.bin"):
            path.unlink()
        marker.unlink(missing_ok=True)

    # Normal writes allocate real Lustre blocks. Reusing one initialized chunk is
    # sufficient because the benchmark measures reads, not compression.
    chunk = memoryview(bytearray((index % 251 for index in range(args.page_bytes))))
    started = time.perf_counter()
    files_written = 0
    for rank in range(args.tp_ranks):
        for page in range(args.pages_per_rank):
            path = page_path(root, rank, page)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                write_full(fd, chunk)
                # Dirty pages cannot be evicted by POSIX_FADV_DONTNEED, so a
                # cold phase would otherwise read the data straight back out of
                # the page cache this very loop populated.
                os.fdatasync(fd)
            finally:
                os.close(fd)
            files_written += 1
            if files_written % 1024 == 0:
                print(
                    f"PREPARE_PROGRESS files={files_written}/"
                    f"{args.tp_ranks * args.pages_per_rank}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    metadata = {
        "format": 1,
        "tp_ranks": args.tp_ranks,
        "pages_per_rank": args.pages_per_rank,
        "page_bytes": args.page_bytes,
        "total_bytes": expected_bytes,
        "created_by_host": socket.gethostname(),
    }
    marker.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        f"PREPARE_COMPLETE files={files_written} total_gib={expected_bytes / 2**30:.3f} "
        f"seconds={elapsed:.3f} gib_s={expected_bytes / elapsed / 2**30:.3f}",
        flush=True,
    )


def cleanup(args: argparse.Namespace) -> None:
    root = Path(args.data_dir)
    marker = root / "dataset.json"
    if not marker.is_file():
        raise RuntimeError(f"refusing cleanup without benchmark marker: {marker}")
    metadata = json.loads(marker.read_text())
    expected = metadata["tp_ranks"] * metadata["pages_per_rank"]
    pages = list(root.glob("*-dualpath-kv-page.bin"))
    page_set = set(pages)
    unexpected = [path for path in root.iterdir() if path != marker and path not in page_set]
    if unexpected or len(pages) != expected:
        raise RuntimeError(
            f"refusing cleanup: expected {expected} page files, found {len(pages)}; "
            f"unexpected={unexpected[:5]}"
        )
    for path in pages:
        path.unlink()
    marker.unlink()
    root.rmdir()
    print(f"CLEANUP_COMPLETE data_dir={root} files={len(pages)}", flush=True)


def drop_cache(args: argparse.Namespace) -> None:
    """Evict the page files from this node's page cache and report the effect.

    Running this on g5 before a g6 cold phase also drops the NFS server-side
    cache for the same files, so a g6 cold read really traverses the NVMe.
    """
    root = Path(args.data_dir)
    paths = [
        page_path(root, rank, page)
        for rank in range(args.tp_ranks)
        for page in range(args.pages_per_rank)
    ]
    os.sync()
    advise_drop(paths)
    resident, total = resident_pages(paths[0])
    print(
        f"KVIO_DROP_CACHE host={socket.gethostname().split('.')[0]} "
        f"files={len(paths)} probe_resident={resident}/{total}",
        flush=True,
    )


def advise_drop(paths: list[Path]) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def buffer_address(target: memoryview) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(target))


def percentile(ordered: list[int], fraction: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def describe_ns(samples: list[int]) -> dict[str, Any]:
    """Latency distribution for one timed step, in nanoseconds."""
    if not samples:
        return {"count": 0}
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "total_ns": sum(ordered),
        "mean_ns": statistics.fmean(ordered),
        "min_ns": ordered[0],
        "p50_ns": percentile(ordered, 0.50),
        "p90_ns": percentile(ordered, 0.90),
        "p99_ns": percentile(ordered, 0.99),
        "max_ns": ordered[-1],
        "stdev_ns": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
    }


def merge_ns(distributions: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-rank distributions without shipping every raw sample."""
    present = [item for item in distributions if item.get("count")]
    if not present:
        return {"count": 0}
    total_count = sum(int(item["count"]) for item in present)
    total_ns = sum(int(item["total_ns"]) for item in present)
    return {
        "count": total_count,
        "total_ns": total_ns,
        "mean_ns": total_ns / total_count,
        "min_ns": min(int(item["min_ns"]) for item in present),
        # Per-rank percentiles are averaged: every rank issues the same number of
        # identically sized page reads, so the ranks are equally weighted.
        "p50_ns": statistics.fmean(float(item["p50_ns"]) for item in present),
        "p90_ns": statistics.fmean(float(item["p90_ns"]) for item in present),
        "p99_ns": statistics.fmean(float(item["p99_ns"]) for item in present),
        "max_ns": max(int(item["max_ns"]) for item in present),
    }


def read_page(path: Path, target: memoryview, io_mode: str, alignment: int) -> None:
    flags = os.O_RDONLY
    if io_mode == "direct":
        if not hasattr(os, "O_DIRECT"):
            raise RuntimeError("Python/os does not expose O_DIRECT on this platform")
        address = buffer_address(target)
        if address % alignment:
            raise RuntimeError(
                f"O_DIRECT target address 0x{address:x} is not {alignment}-byte aligned"
            )
        if len(target) % alignment:
            raise RuntimeError(
                f"O_DIRECT read length {len(target)} is not {alignment}-byte aligned"
            )
        flags |= os.O_DIRECT

    fd = os.open(path, flags)
    try:
        offset = 0
        while offset < len(target):
            count = os.readv(fd, [target[offset:]])
            if count == 0:
                raise RuntimeError(
                    f"short read: {path} returned {offset}/{len(target)} bytes"
                )
            offset += count
    finally:
        os.close(fd)


def read_page_timed(
    path: Path, target: memoryview, flags: int
) -> tuple[int, int, int]:
    """Same open/readinto/close sequence as read_page, timed per step.

    Returns (open_ns, read_ns, close_ns). The flags are resolved once per phase
    so the alignment checks stay out of the timed region.
    """
    open_started = time.perf_counter_ns()
    fd = os.open(path, flags)
    read_started = time.perf_counter_ns()
    try:
        offset = 0
        while offset < len(target):
            count = os.readv(fd, [target[offset:]])
            if count == 0:
                raise RuntimeError(
                    f"short read: {path} returned {offset}/{len(target)} bytes"
                )
            offset += count
        read_finished = time.perf_counter_ns()
    finally:
        os.close(fd)
    closed = time.perf_counter_ns()
    return (
        read_started - open_started,
        read_finished - read_started,
        closed - read_finished,
    )


def read_flags(target: memoryview, io_mode: str, alignment: int) -> int:
    """Validate O_DIRECT alignment once and return the open(2) flags."""
    flags = os.O_RDONLY
    if io_mode == "direct":
        if not hasattr(os, "O_DIRECT"):
            raise RuntimeError("Python/os does not expose O_DIRECT on this platform")
        address = buffer_address(target)
        if address % alignment:
            raise RuntimeError(
                f"O_DIRECT target address 0x{address:x} is not {alignment}-byte aligned"
            )
        if len(target) % alignment:
            raise RuntimeError(
                f"O_DIRECT read length {len(target)} is not {alignment}-byte aligned"
            )
        flags |= os.O_DIRECT
    return flags


def resident_pages(path: Path) -> tuple[int, int]:
    size = path.stat().st_size
    system_page = mmap.PAGESIZE
    page_count = (size + system_page - 1) // system_page
    libc = ctypes.CDLL(None, use_errno=True)
    vector = (ctypes.c_ubyte * page_count)()
    with path.open("rb", buffering=0) as source:
        with mmap.mmap(source.fileno(), size, access=mmap.ACCESS_COPY) as mapping:
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            result = libc.mincore(
                ctypes.c_void_p(address), ctypes.c_size_t(size), vector
            )
            if result != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
    return sum(1 for value in vector if value & 1), page_count


def cache_probe(args: argparse.Namespace) -> None:
    path = page_path(Path(args.data_dir), args.rank, args.page)
    if not path.is_file():
        raise RuntimeError(f"cache probe page does not exist: {path}")
    owner, target = allocate_buffer(
        args.page_bytes, args.buffer_type, args.direct_alignment
    )
    address = buffer_address(target)
    if address % args.direct_alignment:
        raise RuntimeError(
            f"probe buffer 0x{address:x} is not {args.direct_alignment}-byte aligned"
        )

    os.sync()
    fd = os.open(path, os.O_RDONLY)
    try:
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    before, total = resident_pages(path)

    read_page(path, target, "direct", args.direct_alignment)
    after_direct, _ = resident_pages(path)

    fd = os.open(path, os.O_RDONLY)
    try:
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    before_buffered, _ = resident_pages(path)
    read_page(path, target, "buffered", args.direct_alignment)
    after_buffered, _ = resident_pages(path)

    result = {
        "host": socket.gethostname().split(".")[0],
        "path": str(path),
        "page_bytes": args.page_bytes,
        "system_page_bytes": mmap.PAGESIZE,
        "buffer_type": args.buffer_type,
        "buffer_address": address,
        "direct_alignment": args.direct_alignment,
        "resident_before": before,
        "resident_after_direct": after_direct,
        "resident_before_buffered": before_buffered,
        "resident_after_buffered": after_buffered,
        "resident_pages_total": total,
        "direct_populated_cache": after_direct > before,
        "buffered_populated_cache": after_buffered > before_buffered,
    }
    print(CACHE_PROBE_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    del owner


def reader_worker(
    rank: int,
    data_dir: str,
    pages_per_rank: int,
    page_bytes: int,
    passes: int,
    buffer_type: str,
    drop_cache: bool,
    copy_to_staging: bool,
    copy_implementation: str,
    torch_threads: int,
    layers: int,
    page_tokens: int,
    head_dim: int,
    io_mode: str,
    direct_alignment: int,
    start_epoch_ns: Any,
    start_event: Any,
    output: mp.Queue,
) -> None:
    try:
        paths = [page_path(Path(data_dir), rank, page) for page in range(pages_per_rank)]
        exact_torch = copy_implementation == "torch"
        target_backing_owner = None
        if exact_torch:
            import torch

            # SGLang model_runner sets this to one for every TP worker.
            torch.set_num_threads(torch_threads)
            element_bytes = torch.empty((), dtype=torch.bfloat16).element_size()
            denominator = 2 * layers * page_tokens * head_dim * element_bytes
            if page_bytes % denominator:
                raise RuntimeError(
                    f"page_bytes={page_bytes} is incompatible with MHA layout: "
                    f"2*layers*page_tokens*head_dim*bfloat16={denominator}"
                )
            head_num = page_bytes // denominator
            if io_mode == "direct":
                target_backing_owner = torch.empty(
                    page_bytes + direct_alignment,
                    dtype=torch.uint8,
                    pin_memory=buffer_type == "pinned",
                )
                base = target_backing_owner.data_ptr()
                aligned_offset = (-base) % direct_alignment
                target_owner = target_backing_owner[
                    aligned_offset : aligned_offset + page_bytes
                ].view(torch.bfloat16)
            else:
                target_owner = torch.zeros(
                    (2, layers, page_tokens, head_num, head_dim),
                    dtype=torch.bfloat16,
                    pin_memory=buffer_type == "pinned",
                ).flatten()
            target = memoryview(target_owner.view(torch.uint8).contiguous().numpy())
        else:
            target_owner, target = allocate_buffer(
                page_bytes,
                buffer_type,
                direct_alignment if io_mode == "direct" else 1,
            )
        staging_owner = None
        staging = None
        if copy_to_staging:
            if exact_torch:
                staging_owner = torch.empty(
                    (2, layers, pages_per_rank * page_tokens, head_num, head_dim),
                    dtype=torch.bfloat16,
                    pin_memory=buffer_type == "pinned",
                )
            else:
                if page_bytes % layers:
                    raise RuntimeError(
                        f"page_bytes={page_bytes} is not divisible by layers={layers}"
                    )
                staging_owner, staging = allocate_buffer(
                    pages_per_rank * page_bytes, buffer_type
                )
        if drop_cache:
            advise_drop(paths)
        flags = read_flags(target, io_mode, direct_alignment)
        output.put({"ready": rank})
        start_event.wait()
        wait_until(start_epoch_ns.value)

        wall_start_ns = time.time_ns()
        perf_start = time.perf_counter()
        bytes_read = 0
        checksum = 0
        layer_bytes = page_bytes // layers
        open_samples: list[int] = []
        read_samples: list[int] = []
        close_samples: list[int] = []
        scatter_samples: list[int] = []
        for _ in range(passes):
            for page, path in enumerate(paths):
                if exact_torch:
                    # Match HiCacheFile.get(), which constructs this view for
                    # every file read rather than caching a Python buffer.
                    target = memoryview(
                        target_owner.view(torch.uint8).contiguous().numpy()
                    )
                open_ns, read_ns, close_ns = read_page_timed(path, target, flags)
                open_samples.append(open_ns)
                read_samples.append(read_ns)
                close_samples.append(close_ns)
                scatter_started = time.perf_counter_ns()
                if exact_torch and staging_owner is not None:
                    index = page * page_tokens
                    staging_owner[:, :, index : index + page_tokens, :, :] = (
                        target_owner.reshape(
                            2, layers, page_tokens, head_num, head_dim
                        )
                    )
                elif staging is not None:
                    # Production get() reads a page_first_direct page into a flat
                    # target, then set_from_flat_data_page() scatters its layer
                    # slices into the persistent layer_first PD staging pool.
                    for layer in range(layers):
                        source_start = layer * layer_bytes
                        destination_start = (
                            layer * pages_per_rank * layer_bytes + page * layer_bytes
                        )
                        staging[destination_start : destination_start + layer_bytes] = (
                            target[source_start : source_start + layer_bytes]
                        )
                if staging_owner is not None or staging is not None:
                    scatter_samples.append(time.perf_counter_ns() - scatter_started)
                bytes_read += page_bytes
                checksum ^= target[0]
        perf_end = time.perf_counter()
        wall_end_ns = time.time_ns()
        file_io_ns = sum(open_samples) + sum(read_samples) + sum(close_samples)
        output.put(
            {
                "rank": rank,
                "bytes": bytes_read,
                "seconds": perf_end - perf_start,
                "file_io_seconds": file_io_ns / 1e9,
                "scatter_seconds": sum(scatter_samples) / 1e9,
                "open_ns": describe_ns(open_samples),
                "read_ns": describe_ns(read_samples),
                "close_ns": describe_ns(close_samples),
                "scatter_ns": describe_ns(scatter_samples),
                "wall_start_ns": wall_start_ns,
                "wall_end_ns": wall_end_ns,
                "checksum": checksum,
            }
        )
        # Keep both backing allocations live through the timed region.
        del target_owner, target_backing_owner, staging_owner
    except BaseException as exc:
        output.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}"})


def read(args: argparse.Namespace) -> None:
    root = Path(args.data_dir)
    metadata = json.loads((root / "dataset.json").read_text())
    for key in ("tp_ranks", "pages_per_rank", "page_bytes"):
        requested = getattr(args, key)
        if metadata[key] != requested:
            raise RuntimeError(
                f"dataset {key}={metadata[key]}, but reader requested {requested}"
            )

    if args.drop_cache:
        # POSIX_FADV_DONTNEED silently skips dirty pages; flush once here rather
        # than from every rank so a cold phase starts with an evictable cache.
        os.sync()

    ctx = mp.get_context("spawn")
    output: mp.Queue = ctx.Queue()
    start_epoch_ns = ctx.Value("q", 0)
    start_event = ctx.Event()
    workers = [
        ctx.Process(
            target=reader_worker,
            args=(
                rank,
                str(root),
                args.pages_per_rank,
                args.page_bytes,
                args.passes,
                args.buffer_type,
                args.drop_cache,
                args.copy_to_staging,
                args.copy_implementation,
                args.torch_threads,
                args.layers,
                args.page_tokens,
                args.head_dim,
                args.io_mode,
                args.direct_alignment,
                start_epoch_ns,
                start_event,
                output,
            ),
        )
        for rank in range(args.tp_ranks)
    ]
    for worker in workers:
        worker.start()

    ready = set()
    while len(ready) < len(workers):
        try:
            message = output.get(timeout=600)
        except queue.Empty as exc:
            raise RuntimeError("timed out waiting for reader allocation") from exc
        if "error" in message:
            for worker in workers:
                worker.terminate()
            raise RuntimeError(f"reader worker failed during setup: {message}")
        ready.add(message["ready"])

    node_index = int(os.environ.get("SLURM_PROCID", "0"))
    handshake = synchronize(
        args.nodes, node_index, args.sync_host, args.sync_port, lead_ns=2_000_000_000
    )
    start_epoch_ns.value = handshake["release_epoch_ns"]
    start_event.set()
    rank_results = [output.get() for _ in workers]
    for worker in workers:
        worker.join()
    errors = [item for item in rank_results if "error" in item]
    if errors:
        raise RuntimeError(f"reader workers failed: {errors}")
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError(
            f"reader worker exit codes: {[worker.exitcode for worker in workers]}"
        )

    total_bytes = sum(item["bytes"] for item in rank_results)
    wall_start_ns = min(item["wall_start_ns"] for item in rank_results)
    wall_end_ns = max(item["wall_end_ns"] for item in rank_results)
    elapsed = (wall_end_ns - wall_start_ns) / 1e9
    rank_rates = [item["bytes"] / item["seconds"] / 2**30 for item in rank_results]
    offset_ns = handshake["leader_minus_local_ns"]
    # TP ranks read in parallel, so the critical path of each timed step is the
    # slowest rank, not the sum across ranks.
    file_io_seconds = max(item["file_io_seconds"] for item in rank_results)
    scatter_seconds = max(item["scatter_seconds"] for item in rank_results)
    result = {
        "phase": args.phase,
        "repetition": args.repetition,
        "nodes": args.nodes,
        "host": socket.gethostname().split(".")[0],
        "tp_ranks": args.tp_ranks,
        "pages_per_rank": args.pages_per_rank,
        "page_bytes": args.page_bytes,
        "passes": args.passes,
        "buffer_type": args.buffer_type,
        "drop_cache": args.drop_cache,
        "copy_to_staging": args.copy_to_staging,
        "copy_implementation": args.copy_implementation,
        "torch_threads": args.torch_threads,
        "layers": args.layers,
        "page_tokens": args.page_tokens,
        "head_dim": args.head_dim,
        "io_mode": args.io_mode,
        "direct_alignment": args.direct_alignment,
        "bytes": total_bytes,
        "gib": total_bytes / 2**30,
        "seconds": elapsed,
        "gib_s": total_bytes / elapsed / 2**30,
        "file_io_seconds": file_io_seconds,
        "scatter_seconds": scatter_seconds,
        "file_io_gib_s": total_bytes / file_io_seconds / 2**30,
        "scatter_gib_s": (
            total_bytes / scatter_seconds / 2**30 if scatter_seconds else None
        ),
        "file_io_fraction": file_io_seconds / elapsed,
        "scatter_fraction": scatter_seconds / elapsed,
        "open_ns": merge_ns([item["open_ns"] for item in rank_results]),
        "read_ns": merge_ns([item["read_ns"] for item in rank_results]),
        "close_ns": merge_ns([item["close_ns"] for item in rank_results]),
        "scatter_step_ns": merge_ns([item["scatter_ns"] for item in rank_results]),
        "clock_leader_minus_local_ns": offset_ns,
        "clock_offset_round_trip_ns": handshake["offset_round_trip_ns"],
        "wall_start_ns_leader": wall_start_ns + offset_ns,
        "wall_end_ns_leader": wall_end_ns + offset_ns,
        "rank_gib_s_min": min(rank_rates),
        "rank_gib_s_median": statistics.median(rank_rates),
        "rank_gib_s_p95": (
            statistics.quantiles(rank_rates, n=100)[94]
            if len(rank_rates) > 1
            else rank_rates[0]
        ),
        "rank_gib_s_max": max(rank_rates),
        "wall_start_ns": wall_start_ns,
        "wall_end_ns": wall_end_ns,
    }
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    if args.output:
        # srun splits long labelled output lines, so the file is the
        # authoritative copy and the log line is for humans.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def add_layout_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tp-ranks", type=int, default=8)
    parser.add_argument("--pages-per-rank", type=int, default=2048)
    parser.add_argument("--page-bytes", type=int, default=2 * 1024 * 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    add_layout_arguments(prepare_parser)
    prepare_parser.add_argument("--force", action="store_true")

    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("--data-dir", required=True)

    drop_parser = subparsers.add_parser("drop-cache")
    add_layout_arguments(drop_parser)

    probe_parser = subparsers.add_parser("cache-probe")
    add_layout_arguments(probe_parser)
    probe_parser.add_argument("--rank", type=int, default=0)
    probe_parser.add_argument("--page", type=int, default=0)
    probe_parser.add_argument(
        "--buffer-type", choices=("pinned", "pageable"), default="pinned"
    )
    probe_parser.add_argument("--direct-alignment", type=int, default=4096)

    read_parser = subparsers.add_parser("read")
    add_layout_arguments(read_parser)
    read_parser.add_argument("--passes", type=int, default=1)
    read_parser.add_argument("--buffer-type", choices=("pinned", "pageable"), default="pinned")
    read_parser.add_argument("--drop-cache", action="store_true")
    read_parser.add_argument("--copy-to-staging", action="store_true")
    read_parser.add_argument(
        "--copy-implementation", choices=("torch", "python"), default="torch"
    )
    read_parser.add_argument("--torch-threads", type=int, default=1)
    read_parser.add_argument("--layers", type=int, default=64)
    read_parser.add_argument("--page-tokens", type=int, default=64)
    read_parser.add_argument("--head-dim", type=int, default=128)
    read_parser.add_argument("--sync-host", default="")
    read_parser.add_argument("--sync-port", type=int, default=29577)
    read_parser.add_argument("--phase", required=True)
    read_parser.add_argument("--repetition", type=int, required=True)
    read_parser.add_argument("--nodes", type=int, required=True)
    read_parser.add_argument("--io-mode", choices=("buffered", "direct"), default="buffered")
    read_parser.add_argument("--direct-alignment", type=int, default=4096)
    read_parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    for name in ("tp_ranks", "pages_per_rank", "page_bytes"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if getattr(args, "passes", 1) <= 0:
        parser.error("--passes must be positive")
    for name in ("torch_threads", "layers", "page_tokens", "head_dim"):
        if getattr(args, name, 1) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "cleanup":
        cleanup(args)
    elif args.command == "drop-cache":
        drop_cache(args)
    elif args.command == "cache-probe":
        cache_probe(args)
    else:
        read(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
