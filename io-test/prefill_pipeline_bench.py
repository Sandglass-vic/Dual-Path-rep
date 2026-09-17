#!/usr/bin/env python3
"""Reproduce the Prefill HiCache file-read pipeline on one TP8 node."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import queue
import socket
import statistics
import threading
import time
from pathlib import Path
from typing import Any

from kv_read_bench import buffer_address, page_path, read_page


RESULT_PREFIX = "PREFILL_PIPELINE_RESULT "


def hash_page(agent: int, page: int, page_tokens: int, prior: str | None) -> str:
    hasher = hashlib.sha256()
    if prior:
        hasher.update(bytes.fromhex(prior))
    token_base = (agent * 1_000_000 + page * page_tokens) & 0xFFFFFFFF
    for offset in range(page_tokens):
        token = (token_base + offset) & 0xFFFFFFFF
        hasher.update(token.to_bytes(4, byteorder="little", signed=False))
    return hasher.hexdigest()


def worker(
    rank: int,
    data_dir: str,
    agents: int,
    pages_per_agent: int,
    page_bytes: int,
    layers: int,
    page_tokens: int,
    head_dim: int,
    direct_alignment: int,
    torch_threads: int,
    host_pool_tokens: int,
    host_start_token: int,
    h2d_copy_rounds: int,
    prefetch_capacity_tokens: int,
    initial_prefetch_occupied_tokens: int,
    start_epoch_ns: Any,
    start_event: Any,
    output: mp.Queue,
) -> None:
    try:
        import torch

        torch.set_num_threads(torch_threads)
        total_pages = agents * pages_per_agent
        request_tokens_total = total_pages * page_tokens
        if host_pool_tokens <= 0:
            host_pool_tokens = request_tokens_total
        if host_start_token % page_tokens:
            raise RuntimeError("host start token must be page aligned")
        if host_start_token + request_tokens_total > host_pool_tokens:
            raise RuntimeError("request allocations exceed the Host pool")
        element_bytes = torch.empty((), dtype=torch.bfloat16).element_size()
        denominator = 2 * layers * page_tokens * head_dim * element_bytes
        if page_bytes % denominator:
            raise RuntimeError("page size is incompatible with the MHA layout")
        head_num = page_bytes // denominator

        target_backing = torch.empty(
            page_bytes + direct_alignment,
            dtype=torch.uint8,
            pin_memory=True,
        )
        aligned_offset = (-target_backing.data_ptr()) % direct_alignment
        target_owner = target_backing[
            aligned_offset : aligned_offset + page_bytes
        ].view(torch.bfloat16)
        if target_owner.data_ptr() % direct_alignment:
            raise RuntimeError("failed to create an aligned pinned target")

        # One persistent layer_first Host pool covering all agent allocations.
        host_pool = torch.empty(
            (2, layers, host_pool_tokens, head_num, head_dim),
            dtype=torch.bfloat16,
            pin_memory=True,
        )

        io_queue: queue.Queue[Any] = queue.Queue()
        request_results: list[dict[str, float | int]] = []
        rejected_requests: list[dict[str, int]] = []
        request_lock = threading.Lock()
        h2d_result: dict[str, float | int] = {"rounds": 0, "seconds": 0.0}

        device_k = None
        device_v = None
        h2d_stream = None
        if h2d_copy_rounds > 0:
            torch.cuda.set_device(rank)
            request_tokens = pages_per_agent * page_tokens
            device_k = torch.empty(
                (layers, request_tokens, head_num, head_dim),
                dtype=torch.bfloat16,
                device=f"cuda:{rank}",
            )
            device_v = torch.empty_like(device_k)
            h2d_stream = torch.cuda.Stream(device=rank)
            # Initialize the CUDA context and copy path before timed work.
            with torch.cuda.stream(h2d_stream):
                device_k[0].copy_(
                    host_pool[0, 0, host_start_token : host_start_token + request_tokens],
                    non_blocking=True,
                )
            h2d_stream.synchronize()

        def h2d_loop() -> None:
            if h2d_copy_rounds <= 0:
                return
            torch.cuda.set_device(rank)
            request_tokens = pages_per_agent * page_tokens
            begin = time.perf_counter()
            with torch.cuda.stream(h2d_stream):
                for copy_round in range(h2d_copy_rounds):
                    agent = copy_round % agents
                    first = host_start_token + agent * request_tokens
                    last = first + request_tokens
                    for layer in range(layers):
                        device_k[layer].copy_(
                            host_pool[0, layer, first:last], non_blocking=True
                        )
                        device_v[layer].copy_(
                            host_pool[1, layer, first:last], non_blocking=True
                        )
            h2d_stream.synchronize()
            h2d_result["rounds"] = h2d_copy_rounds
            h2d_result["seconds"] = time.perf_counter() - begin

        def io_loop() -> None:
            while True:
                item = io_queue.get()
                if item is None:
                    return
                agent, paths, submitted, metadata_start, metadata_end = item
                io_start = time.perf_counter()
                for page, path in enumerate(paths):
                    target = memoryview(target_owner.view(torch.uint8).numpy())
                    read_page(path, target, "direct", direct_alignment)
                    host_index = host_start_token + (
                        agent * pages_per_agent + page
                    ) * page_tokens
                    host_pool[
                        :, :, host_index : host_index + page_tokens, :, :
                    ] = target_owner.reshape(
                        2, layers, page_tokens, head_num, head_dim
                    )
                io_end = time.perf_counter()
                with request_lock:
                    request_results.append(
                        {
                            "agent": agent,
                            "submitted": submitted,
                            "metadata_start": metadata_start,
                            "metadata_end": metadata_end,
                            "io_start": io_start,
                            "io_end": io_end,
                        }
                    )

        output.put({"ready": rank, "target_address": buffer_address(
            memoryview(target_owner.view(torch.uint8).numpy())
        )})
        start_event.wait()
        start_ns = start_epoch_ns.value
        while time.time_ns() < start_ns:
            remaining = start_ns - time.time_ns()
            time.sleep(min(remaining / 1e9, 0.01))

        submitted = time.perf_counter()
        io_thread = threading.Thread(target=io_loop)
        h2d_thread = threading.Thread(target=h2d_loop)
        io_thread.start()
        h2d_thread.start()

        root = Path(data_dir)
        prefetch_tokens_occupied = initial_prefetch_occupied_tokens
        for agent in range(agents):
            request_tokens = pages_per_agent * page_tokens
            if (
                prefetch_capacity_tokens > 0
                and prefetch_tokens_occupied >= prefetch_capacity_tokens
            ):
                rejected_requests.append(
                    {
                        "agent": agent,
                        "occupied_tokens": prefetch_tokens_occupied,
                        "capacity_tokens": prefetch_capacity_tokens,
                    }
                )
                continue

            metadata_start = time.perf_counter()
            paths: list[Path] = []
            prior_hash = None
            for page in range(pages_per_agent):
                prior_hash = hash_page(agent, page, page_tokens, prior_hash)
                path = page_path(root, rank, agent * pages_per_agent + page)
                if not path.exists():
                    raise FileNotFoundError(path)
                paths.append(path)
            metadata_end = time.perf_counter()
            io_queue.put(
                (agent, paths, submitted, metadata_start, metadata_end)
            )
            # Match HiRadixCache.prefetch_from_storage(): the limit is checked
            # before submission, then the entire request is charged.
            prefetch_tokens_occupied += request_tokens

        io_queue.put(None)
        io_thread.join()
        h2d_thread.join()
        completed = max(float(item["io_end"]) for item in request_results)
        per_request = []
        for item in sorted(request_results, key=lambda value: int(value["agent"])):
            per_request.append(
                {
                    "agent": int(item["agent"]),
                    "metadata_wait_s": float(item["metadata_start"]) - submitted,
                    "metadata_s": float(item["metadata_end"])
                    - float(item["metadata_start"]),
                    "io_queue_s": float(item["io_start"])
                    - float(item["metadata_end"]),
                    "io_service_s": float(item["io_end"])
                    - float(item["io_start"]),
                    "total_s": float(item["io_end"]) - submitted,
                }
            )
        output.put(
            {
                "rank": rank,
                "requested_bytes": agents * pages_per_agent * page_bytes,
                "read_bytes": len(request_results)
                * pages_per_agent
                * page_bytes,
                "pipeline_s": completed - submitted,
                "submitted": submitted,
                "completed": completed,
                "requests": per_request,
                "rejected_requests": rejected_requests,
                "prefetch_tokens_occupied_after_submit": prefetch_tokens_occupied,
                "host_pool_tokens": host_pool_tokens,
                "host_start_token": host_start_token,
                "h2d_copy_rounds": int(h2d_result["rounds"]),
                "h2d_seconds": float(h2d_result["seconds"]),
            }
        )
        del target_owner, target_backing, host_pool
    except BaseException as exc:
        output.put({"rank": rank, "error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tp-ranks", type=int, default=8)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--pages-per-agent", type=int, default=512)
    parser.add_argument("--page-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument("--page-tokens", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--direct-alignment", type=int, default=4096)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--host-pool-tokens", type=int, default=0)
    parser.add_argument("--host-start-token", type=int, default=0)
    parser.add_argument("--h2d-copy-rounds", type=int, default=0)
    parser.add_argument("--prefetch-capacity-tokens", type=int, default=0)
    parser.add_argument("--initial-prefetch-occupied-tokens", type=int, default=0)
    parser.add_argument("--start-epoch-ns", type=int, default=0)
    args = parser.parse_args()

    metadata = json.loads((Path(args.data_dir) / "dataset.json").read_text())
    expected_pages = args.agents * args.pages_per_agent
    if metadata["tp_ranks"] != args.tp_ranks:
        raise RuntimeError("dataset TP rank count does not match")
    if metadata["pages_per_rank"] != expected_pages:
        raise RuntimeError("dataset page count does not match agents * pages")
    if metadata["page_bytes"] != args.page_bytes:
        raise RuntimeError("dataset page size does not match")

    ctx = mp.get_context("spawn")
    output: mp.Queue = ctx.Queue()
    start_ns = args.start_epoch_ns or (time.time_ns() + 2_000_000_000)
    start_epoch_ns = ctx.Value("q", start_ns)
    start_event = ctx.Event()
    workers = [
        ctx.Process(
            target=worker,
            args=(
                rank,
                args.data_dir,
                args.agents,
                args.pages_per_agent,
                args.page_bytes,
                args.layers,
                args.page_tokens,
                args.head_dim,
                args.direct_alignment,
                args.torch_threads,
                args.host_pool_tokens,
                args.host_start_token,
                args.h2d_copy_rounds,
                args.prefetch_capacity_tokens,
                args.initial_prefetch_occupied_tokens,
                start_epoch_ns,
                start_event,
                output,
            ),
        )
        for rank in range(args.tp_ranks)
    ]
    for process in workers:
        process.start()
    ready = []
    while len(ready) < args.tp_ranks:
        message = output.get(timeout=600)
        if "error" in message:
            raise RuntimeError(message)
        ready.append(message)
    start_event.set()
    results = [output.get(timeout=1800) for _ in workers]
    for process in workers:
        process.join()
    errors = [item for item in results if "error" in item]
    if errors:
        raise RuntimeError(errors)

    total_requested_bytes = sum(int(item["requested_bytes"]) for item in results)
    total_read_bytes = sum(int(item["read_bytes"]) for item in results)
    wall_s = max(float(item["completed"]) for item in results) - min(
        float(item["submitted"]) for item in results
    )
    slowest_rank = max(results, key=lambda item: float(item["pipeline_s"]))
    requests = slowest_rank["requests"]
    result = {
        "host": socket.gethostname().split(".")[0],
        "tp_ranks": args.tp_ranks,
        "agents": args.agents,
        "pages_per_agent": args.pages_per_agent,
        "page_bytes": args.page_bytes,
        "prefetch_capacity_tokens": args.prefetch_capacity_tokens,
        "initial_prefetch_occupied_tokens": args.initial_prefetch_occupied_tokens,
        "host_pool_tokens": int(slowest_rank["host_pool_tokens"]),
        "host_start_token": int(slowest_rank["host_start_token"]),
        "h2d_copy_rounds": int(slowest_rank["h2d_copy_rounds"]),
        "h2d_seconds": float(slowest_rank["h2d_seconds"]),
        "requested_bytes": total_requested_bytes,
        "requested_gib": total_requested_bytes / 2**30,
        "read_bytes": total_read_bytes,
        "read_gib": total_read_bytes / 2**30,
        "seconds": wall_s,
        "read_gib_s": total_read_bytes / wall_s / 2**30,
        "rank_pipeline_s_min": min(float(item["pipeline_s"]) for item in results),
        "rank_pipeline_s_median": statistics.median(
            float(item["pipeline_s"]) for item in results
        ),
        "rank_pipeline_s_max": max(float(item["pipeline_s"]) for item in results),
        "slowest_rank": int(slowest_rank["rank"]),
        "slowest_rank_requests": requests,
        "rejected_requests": slowest_rank["rejected_requests"],
        "prefetch_tokens_occupied_after_submit": slowest_rank[
            "prefetch_tokens_occupied_after_submit"
        ],
    }
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
