#!/usr/bin/env python3
"""Dependency-free concurrent POSIX bandwidth benchmark with O_DIRECT support."""

from __future__ import annotations

import argparse
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

from cluster_barrier import coordinate  # noqa: E402


PREFIX = "POSIXIO_RESULT "


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def advise_drop(path: Path) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def full_io(fd: int, buffer: memoryview, write: bool) -> None:
    offset = 0
    while offset < len(buffer):
        count = os.writev(fd, [buffer[offset:]]) if write else os.readv(
            fd, [buffer[offset:]]
        )
        if count <= 0:
            raise OSError(f"short {'write' if write else 'read'} at offset {offset}")
        offset += count


def worker(
    job: int,
    directory: str,
    filename_prefix: str,
    rw: str,
    mode: str,
    size_bytes: int,
    block_bytes: int,
    durable: bool,
    start_event: Any,
    output: mp.Queue,
) -> None:
    try:
        path = Path(directory) / f"{filename_prefix}.{job}"
        if rw == "read" and mode == "buffered-cold":
            advise_drop(path)
        flags = os.O_RDONLY if rw == "read" else os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        direct = mode == "direct"
        if direct:
            if not hasattr(os, "O_DIRECT"):
                raise RuntimeError("O_DIRECT is unavailable")
            flags |= os.O_DIRECT
            owner = mmap.mmap(-1, block_bytes)
            buffer = memoryview(owner)
        else:
            owner = bytearray(block_bytes)
            buffer = memoryview(owner)
        if rw == "write":
            buffer[:] = bytes([job % 251]) * block_bytes

        output.put({"ready": job})
        start_event.wait()
        wall_start_ns = time.time_ns()
        started = time.perf_counter()
        latencies_ns: list[int] = []
        fd = os.open(path, flags, 0o600)
        try:
            remaining = size_bytes
            while remaining:
                length = min(block_bytes, remaining)
                io_start_ns = time.perf_counter_ns()
                full_io(fd, buffer[:length], rw == "write")
                latencies_ns.append(time.perf_counter_ns() - io_start_ns)
                remaining -= length
            io_completed = time.perf_counter()
            if rw == "write" and durable:
                os.fdatasync(fd)
        finally:
            os.close(fd)
        completed = time.perf_counter()
        wall_end_ns = time.time_ns()
        output.put(
            {
                "job": job,
                "bytes": size_bytes,
                "io_seconds": io_completed - started,
                "seconds": completed - started,
                "sync_seconds": completed - io_completed,
                "wall_start_ns": wall_start_ns,
                "wall_end_ns": wall_end_ns,
                "latency_ns_p50": percentile(latencies_ns, 0.50),
                "latency_ns_p95": percentile(latencies_ns, 0.95),
            }
        )
        buffer.release()
        if direct:
            owner.close()
    except BaseException as exc:
        output.put({"job": job, "error": f"{type(exc).__name__}: {exc}"})


def prepare(args: argparse.Namespace) -> None:
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    block = memoryview(bytearray(args.block_bytes))
    for job in range(args.jobs):
        path = directory / f"{args.filename_prefix}.{job}"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            remaining = args.size_bytes
            while remaining:
                length = min(args.block_bytes, remaining)
                full_io(fd, block[:length], True)
                remaining -= length
            os.fdatasync(fd)
        finally:
            os.close(fd)
    print(
        f"PREPARE_COMPLETE files={args.jobs} "
        f"bytes={args.jobs * args.size_bytes}",
        flush=True,
    )


def run_io(args: argparse.Namespace) -> None:
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    if args.rw == "read":
        for job in range(args.jobs):
            path = directory / f"{args.filename_prefix}.{job}"
            if path.stat().st_size != args.size_bytes:
                raise RuntimeError(f"unexpected source size: {path}")

    ctx = mp.get_context("spawn")
    output: mp.Queue = ctx.Queue()
    start_event = ctx.Event()
    workers = [
        ctx.Process(
            target=worker,
            args=(
                job,
                str(directory),
                args.filename_prefix,
                args.rw,
                args.mode,
                args.size_bytes,
                args.block_bytes,
                args.durable,
                start_event,
                output,
            ),
        )
        for job in range(args.jobs)
    ]
    for process in workers:
        process.start()
    try:
        ready = set()
        while len(ready) < args.jobs:
            message = output.get(timeout=600)
            if "error" in message:
                raise RuntimeError(message)
            ready.add(int(message["ready"]))
        node_index = int(os.environ.get("SLURM_PROCID", "0"))
        handshake = coordinate(args.nodes, node_index, args.sync_host, args.sync_port)
        start_event.set()
        records = [output.get(timeout=3600) for _ in workers]
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
    errors = [record for record in records if "error" in record]
    if errors or any(process.exitcode != 0 for process in workers):
        raise RuntimeError(
            f"worker failures: errors={errors}, "
            f"exitcodes={[process.exitcode for process in workers]}"
        )

    total_bytes = sum(int(record["bytes"]) for record in records)
    seconds = max(float(record["seconds"]) for record in records)
    io_seconds = max(float(record["io_seconds"]) for record in records)
    offset_ns = handshake["leader_minus_local_ns"]
    wall_start_ns = min(int(record["wall_start_ns"]) for record in records)
    wall_end_ns = max(int(record["wall_end_ns"]) for record in records)
    result = {
        "schema_version": 2,
        "backend": "python-posix",
        "host": socket.gethostname().split(".")[0],
        "direction": args.rw,
        "mode": args.mode,
        "jobs": args.jobs,
        "size_per_job": args.size_bytes,
        "block_bytes": args.block_bytes,
        "durable": args.durable,
        "bytes": total_bytes,
        "seconds": seconds,
        "io_seconds": io_seconds,
        "sync_seconds": seconds - io_seconds,
        "gib_s": total_bytes / seconds / 2**30,
        "io_gib_s": total_bytes / io_seconds / 2**30,
        "clock_leader_minus_local_ns": offset_ns,
        "clock_offset_round_trip_ns": handshake["offset_round_trip_ns"],
        "wall_start_ns_leader": wall_start_ns + offset_ns,
        "wall_end_ns_leader": wall_end_ns + offset_ns,
        "latency_ns": {
            "p50": statistics.median(
                float(record["latency_ns_p50"]) for record in records
            ),
            "p95": max(float(record["latency_ns_p95"]) for record in records),
        },
        "per_job_gib_s": {
            str(record["job"]): int(record["bytes"])
            / float(record["seconds"])
            / 2**30
            for record in records
        },
    }
    rendered = json.dumps(result, sort_keys=True)
    print(PREFIX + rendered, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "io"):
        child = subparsers.add_parser(command)
        child.add_argument("--directory", required=True)
        child.add_argument("--filename-prefix", default="source")
        child.add_argument("--jobs", type=positive, required=True)
        child.add_argument("--size-bytes", type=positive, required=True)
        child.add_argument("--block-bytes", type=positive, required=True)
    io_parser = subparsers.choices["io"]
    io_parser.add_argument("--rw", choices=("read", "write"), required=True)
    io_parser.add_argument(
        "--mode", choices=("direct", "buffered-cold", "buffered-hot"), required=True
    )
    io_parser.add_argument("--durable", action="store_true")
    io_parser.add_argument("--nodes", type=positive, default=1)
    io_parser.add_argument("--sync-host", default="")
    io_parser.add_argument("--sync-port", type=positive, default=29777)
    io_parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.size_bytes % args.block_bytes:
        parser.error("--size-bytes must be divisible by --block-bytes")
    if args.block_bytes % mmap.PAGESIZE:
        parser.error("--block-bytes must be page aligned")
    if getattr(args, "nodes", 1) > 1 and not getattr(args, "sync_host", ""):
        parser.error("--sync-host is required for multi-node I/O")
    prepare(args) if args.command == "prepare" else run_io(args)


if __name__ == "__main__":
    main()
