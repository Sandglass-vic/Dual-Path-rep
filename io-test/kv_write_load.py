#!/usr/bin/env python3
"""Generate Decode-style concurrent HiCache file writes on one TP node."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cluster_barrier import synchronize, wait_until  # noqa: E402
from kv_read_bench import describe_ns, merge_ns  # noqa: E402

RESULT_PREFIX = "KVWRITE_RESULT "


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def worker(
    rank: int,
    output_dir: str,
    pages: int,
    page_bytes: int,
    buffer_type: str,
    sync_mode: str,
    start_epoch_ns: Any,
    start_event: Any,
    output: mp.Queue,
) -> None:
    try:
        if buffer_type == "pinned":
            import torch

            torch.set_num_threads(1)
            value = torch.empty(page_bytes, dtype=torch.uint8, pin_memory=True)
            value.fill_(rank + 1)
            payload = memoryview(value.numpy())
        else:
            value = bytearray([rank + 1]) * page_bytes
            payload = memoryview(value)
        rank_dir = Path(output_dir) / f"rank-{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        if sync_mode != "none":
            fsync_directory(Path(output_dir))
        output.put({"ready": rank})
        start_event.wait()
        wait_until(start_epoch_ns.value)
        wall_start_ns = time.time_ns()
        started = time.perf_counter()
        paths = []
        open_samples: list[int] = []
        write_samples: list[int] = []
        close_samples: list[int] = []
        for page in range(pages):
            # Match HiCacheFile.set(): NumPy tofile to one file per KV page.
            path = rank_dir / f"{page:08x}.bin"
            open_started = time.perf_counter_ns()
            with path.open("wb", buffering=0) as stream:
                write_started = time.perf_counter_ns()
                offset = 0
                while offset < len(payload):
                    written = stream.write(payload[offset:])
                    if written is None or written <= 0:
                        raise OSError(f"short write at offset {offset}: {path}")
                    offset += written
                if sync_mode == "per-file-fdatasync":
                    os.fdatasync(stream.fileno())
                write_finished = time.perf_counter_ns()
            closed = time.perf_counter_ns()
            open_samples.append(write_started - open_started)
            write_samples.append(write_finished - write_started)
            close_samples.append(closed - write_finished)
            paths.append(path)
        write_completed = time.perf_counter()
        write_end_ns = time.time_ns()
        if sync_mode == "batch-fdatasync":
            for path in paths:
                fd = os.open(path, os.O_RDWR)
                try:
                    os.fdatasync(fd)
                finally:
                    os.close(fd)
        if sync_mode != "none":
            fsync_directory(rank_dir)
        completed = time.perf_counter()
        wall_end_ns = time.time_ns()
        output.put(
            {
                "rank": rank,
                "bytes": pages * page_bytes,
                "write_seconds": write_completed - started,
                "sync_seconds": completed - write_completed,
                "seconds": completed - started,
                "file_io_seconds": (
                    sum(open_samples) + sum(write_samples) + sum(close_samples)
                )
                / 1e9,
                "open_ns": describe_ns(open_samples),
                "write_ns": describe_ns(write_samples),
                "close_ns": describe_ns(close_samples),
                "started": started,
                "completed": completed,
                "wall_start_ns": wall_start_ns,
                "write_end_ns": write_end_ns,
                "wall_end_ns": wall_end_ns,
            }
        )
    except BaseException as exc:
        output.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tp-ranks", type=int, default=8)
    parser.add_argument("--pages-per-rank", type=int, default=4096)
    parser.add_argument("--page-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument(
        "--buffer-type", choices=("pinned", "pageable"), default="pinned"
    )
    parser.add_argument(
        "--sync-mode",
        choices=("none", "per-file-fdatasync", "batch-fdatasync"),
        default="none",
    )
    parser.add_argument("--phase", default="write")
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--sync-host", default="")
    parser.add_argument("--sync-port", type=int, default=29677)
    parser.add_argument("--start-epoch-ns", type=int, default=0)
    args = parser.parse_args()
    for name in ("tp_ranks", "pages_per_rank", "page_bytes", "repetition", "nodes"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 1 <= args.sync_port <= 65535:
        parser.error("--sync-port must be between 1 and 65535")
    if args.nodes > 1 and args.start_epoch_ns <= 0 and not args.sync_host:
        parser.error("--sync-host is required for multi-node writes")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.sync_mode != "none":
        fsync_directory(output_dir.parent)
        fsync_directory(output_dir)

    ctx = mp.get_context("spawn")
    output: mp.Queue = ctx.Queue()
    start_epoch_ns = ctx.Value("q", 0)
    start_event = ctx.Event()
    workers = [
        ctx.Process(
            target=worker,
            args=(
                rank,
                args.output_dir,
                args.pages_per_rank,
                args.page_bytes,
                args.buffer_type,
                args.sync_mode,
                start_epoch_ns,
                start_event,
                output,
            ),
        )
        for rank in range(args.tp_ranks)
    ]
    for process in workers:
        process.start()
    try:
        ready = set()
        while len(ready) < args.tp_ranks:
            message = output.get(timeout=600)
            if "error" in message:
                raise RuntimeError(message)
            ready.add(int(message["ready"]))
        handshake = {"leader_minus_local_ns": 0, "offset_round_trip_ns": 0}
        if args.start_epoch_ns > 0:
            start_epoch_ns.value = args.start_epoch_ns
        else:
            node_index = int(os.environ.get("SLURM_PROCID", "0"))
            handshake = synchronize(
                args.nodes, node_index, args.sync_host, args.sync_port
            )
            start_epoch_ns.value = handshake["release_epoch_ns"]
        start_event.set()
        results = [output.get(timeout=1800) for _ in workers]
    except BaseException:
        start_event.set()
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join()
        raise
    for process in workers:
        process.join()
    errors = [item for item in results if "error" in item]
    if errors:
        for process in workers:
            if process.is_alive():
                process.terminate()
        raise RuntimeError(errors)
    if any(process.exitcode != 0 for process in workers):
        raise RuntimeError(
            f"writer worker exit codes: {[process.exitcode for process in workers]}"
        )

    total_bytes = sum(int(item["bytes"]) for item in results)
    wall_start_ns = min(int(item["wall_start_ns"]) for item in results)
    write_end_ns = max(int(item["write_end_ns"]) for item in results)
    wall_end_ns = max(int(item["wall_end_ns"]) for item in results)
    write_elapsed = (write_end_ns - wall_start_ns) / 1e9
    elapsed = (wall_end_ns - wall_start_ns) / 1e9
    rates = [
        int(item["bytes"]) / float(item["seconds"]) / 2**30
        for item in results
    ]
    buffered_gib_s = total_bytes / write_elapsed / 2**30
    durable_gib_s = (
        total_bytes / elapsed / 2**30 if args.sync_mode != "none" else None
    )
    result = {
        "phase": args.phase,
        "repetition": args.repetition,
        "nodes": args.nodes,
        "host": socket.gethostname().split(".")[0],
        "tp_ranks": args.tp_ranks,
        "pages_per_rank": args.pages_per_rank,
        "page_bytes": args.page_bytes,
        "buffer_type": args.buffer_type,
        "sync_mode": args.sync_mode,
        "bytes": total_bytes,
        "gib": total_bytes / 2**30,
        "write_seconds": write_elapsed,
        "sync_seconds": elapsed - write_elapsed,
        "seconds": elapsed,
        "file_io_seconds": max(float(item["file_io_seconds"]) for item in results),
        "buffered_gib_s": buffered_gib_s,
        "durable_gib_s": durable_gib_s,
        "gib_s": durable_gib_s if durable_gib_s is not None else buffered_gib_s,
        "open_ns": merge_ns([item["open_ns"] for item in results]),
        "write_ns": merge_ns([item["write_ns"] for item in results]),
        "close_ns": merge_ns([item["close_ns"] for item in results]),
        "clock_leader_minus_local_ns": handshake["leader_minus_local_ns"],
        "clock_offset_round_trip_ns": handshake["offset_round_trip_ns"],
        "wall_start_ns_leader": wall_start_ns + handshake["leader_minus_local_ns"],
        "wall_end_ns_leader": wall_end_ns + handshake["leader_minus_local_ns"],
        "wall_start_ns": wall_start_ns,
        "write_end_ns": write_end_ns,
        "wall_end_ns": wall_end_ns,
        "rank_gib_s_min": min(rates),
        "rank_gib_s_median": statistics.median(rates),
        "rank_gib_s_p95": (
            statistics.quantiles(rates, n=100)[94] if len(rates) > 1 else rates[0]
        ),
        "rank_gib_s_max": max(rates),
    }
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
