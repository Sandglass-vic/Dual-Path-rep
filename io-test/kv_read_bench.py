#!/usr/bin/env python3
"""Synthetic benchmark for DualPath's Decode-side Lustre page reads."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import socket
import statistics
import struct
import sys
import time
from pathlib import Path
from typing import Any


RESULT_PREFIX = "KVIO_RESULT "


def page_path(root: Path, rank: int, page: int) -> Path:
    # The production backend uses one flat directory of hash-named .bin files.
    # Deterministic names avoid a manifest lookup while preserving open/read/close.
    return root / f"{page:08x}-{rank:02x}-dualpath-kv-page.bin"


def allocate_buffer(size: int, buffer_type: str) -> tuple[Any, memoryview]:
    if buffer_type == "pinned":
        try:
            import torch

            tensor = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            return tensor, memoryview(tensor.numpy())
        except Exception as exc:  # pragma: no cover - depends on runtime image
            raise RuntimeError(
                "Pinned allocation failed. Run in the benchmark image with a GPU "
                "assigned, or set BUFFER_TYPE=pageable."
            ) from exc

    target = bytearray(size)
    return target, memoryview(target)


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


def advise_drop(paths: list[Path]) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


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
    start_epoch_ns: Any,
    start_event: Any,
    output: mp.Queue,
) -> None:
    try:
        paths = [page_path(Path(data_dir), rank, page) for page in range(pages_per_rank)]
        exact_torch = copy_implementation == "torch"
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
            target_owner = torch.zeros(
                (2, layers, page_tokens, head_num, head_dim),
                dtype=torch.bfloat16,
                pin_memory=buffer_type == "pinned",
            ).flatten()
            target = memoryview(target_owner.view(torch.uint8).contiguous().numpy())
        else:
            target_owner, target = allocate_buffer(page_bytes, buffer_type)
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
        output.put({"ready": rank})
        start_event.wait()
        start_ns = start_epoch_ns.value
        while time.time_ns() < start_ns:
            remaining = start_ns - time.time_ns()
            time.sleep(min(remaining / 1e9, 0.01))

        wall_start_ns = time.time_ns()
        perf_start = time.perf_counter()
        bytes_read = 0
        checksum = 0
        layer_bytes = page_bytes // layers
        for _ in range(passes):
            for page, path in enumerate(paths):
                with path.open("rb", buffering=0) as source:
                    if exact_torch:
                        # Match HiCacheFile.get(), which constructs this view for
                        # every file read rather than caching a Python buffer.
                        target = memoryview(
                            target_owner.view(torch.uint8).contiguous().numpy()
                        )
                    offset = 0
                    while offset < page_bytes:
                        count = source.readinto(target[offset:])
                        if count == 0:
                            raise RuntimeError(
                                f"short read: {path} returned {offset}/{page_bytes} bytes"
                            )
                        offset += count
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
                bytes_read += page_bytes
                checksum ^= target[0]
        perf_end = time.perf_counter()
        wall_end_ns = time.time_ns()
        output.put(
            {
                "rank": rank,
                "bytes": bytes_read,
                "seconds": perf_end - perf_start,
                "wall_start_ns": wall_start_ns,
                "wall_end_ns": wall_end_ns,
                "checksum": checksum,
            }
        )
        # Keep both backing allocations live through the timed region.
        del target_owner, staging_owner
    except BaseException as exc:
        output.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}"})


def coordinate_start(nodes: int, node_index: int, host: str, port: int) -> int:
    if nodes == 1:
        return time.time_ns() + 2_000_000_000
    if not host:
        raise RuntimeError("--sync-host is required for a multi-node read")

    if node_index == 0:
        connections = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("0.0.0.0", port))
            server.listen(nodes - 1)
            server.settimeout(300)
            for _ in range(nodes - 1):
                connection, _ = server.accept()
                if connection.recv(5) != b"READY":
                    connection.close()
                    raise RuntimeError("invalid synchronization handshake")
                connections.append(connection)
            start_ns = time.time_ns() + 3_000_000_000
            payload = struct.pack("!Q", start_ns)
            for connection in connections:
                connection.sendall(payload)
                connection.close()
            return start_ns

    deadline = time.monotonic() + 300
    while True:
        try:
            with socket.create_connection((host, port), timeout=5) as connection:
                connection.sendall(b"READY")
                payload = b""
                while len(payload) < 8:
                    chunk = connection.recv(8 - len(payload))
                    if not chunk:
                        raise RuntimeError("synchronization server closed early")
                    payload += chunk
                return struct.unpack("!Q", payload)[0]
        except OSError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"timed out connecting to sync server {host}:{port}")
            time.sleep(1)


def read(args: argparse.Namespace) -> None:
    root = Path(args.data_dir)
    metadata = json.loads((root / "dataset.json").read_text())
    for key in ("tp_ranks", "pages_per_rank", "page_bytes"):
        requested = getattr(args, key)
        if metadata[key] != requested:
            raise RuntimeError(
                f"dataset {key}={metadata[key]}, but reader requested {requested}"
            )

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
    start_epoch_ns.value = coordinate_start(
        args.nodes, node_index, args.sync_host, args.sync_port
    )
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
        "bytes": total_bytes,
        "gib": total_bytes / 2**30,
        "seconds": elapsed,
        "gib_s": total_bytes / elapsed / 2**30,
        "rank_gib_s_min": min(rank_rates),
        "rank_gib_s_median": statistics.median(rank_rates),
        "rank_gib_s_max": max(rank_rates),
        "wall_start_ns": wall_start_ns,
        "wall_end_ns": wall_end_ns,
    }
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)


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
    else:
        read(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
