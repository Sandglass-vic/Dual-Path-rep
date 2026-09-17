#!/usr/bin/env python3
"""Generate Decode-style concurrent HiCache file writes on one TP8 node."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import socket
import statistics
import time
from pathlib import Path


RESULT_PREFIX = "KVWRITE_RESULT "


def worker(
    rank: int,
    output_dir: str,
    pages: int,
    page_bytes: int,
    start_epoch_ns: int,
    output: mp.Queue,
) -> None:
    try:
        import torch

        torch.set_num_threads(1)
        value = torch.empty(page_bytes, dtype=torch.uint8, pin_memory=True)
        value.fill_(rank + 1)
        array = value.numpy()
        rank_dir = Path(output_dir) / f"rank-{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        output.put({"ready": rank})
        while time.time_ns() < start_epoch_ns:
            remaining = start_epoch_ns - time.time_ns()
            time.sleep(min(remaining / 1e9, 0.01))
        started = time.perf_counter()
        for page in range(pages):
            # Match HiCacheFile.set(): NumPy tofile to one file per KV page.
            array.tofile(rank_dir / f"{page:08x}.bin")
        completed = time.perf_counter()
        output.put(
            {
                "rank": rank,
                "bytes": pages * page_bytes,
                "seconds": completed - started,
                "started": started,
                "completed": completed,
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
    parser.add_argument("--start-epoch-ns", type=int, required=True)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    output: mp.Queue = ctx.Queue()
    workers = [
        ctx.Process(
            target=worker,
            args=(
                rank,
                args.output_dir,
                args.pages_per_rank,
                args.page_bytes,
                args.start_epoch_ns,
                output,
            ),
        )
        for rank in range(args.tp_ranks)
    ]
    for process in workers:
        process.start()
    ready = set()
    while len(ready) < args.tp_ranks:
        message = output.get(timeout=600)
        if "error" in message:
            raise RuntimeError(message)
        ready.add(int(message["ready"]))
    results = [output.get(timeout=1800) for _ in workers]
    for process in workers:
        process.join()
    errors = [item for item in results if "error" in item]
    if errors:
        raise RuntimeError(errors)

    total_bytes = sum(int(item["bytes"]) for item in results)
    elapsed = max(float(item["completed"]) for item in results) - min(
        float(item["started"]) for item in results
    )
    rates = [
        int(item["bytes"]) / float(item["seconds"]) / 2**30
        for item in results
    ]
    result = {
        "host": socket.gethostname().split(".")[0],
        "tp_ranks": args.tp_ranks,
        "pages_per_rank": args.pages_per_rank,
        "page_bytes": args.page_bytes,
        "bytes": total_bytes,
        "gib": total_bytes / 2**30,
        "seconds": elapsed,
        "gib_s": total_bytes / elapsed / 2**30,
        "rank_gib_s_min": min(rates),
        "rank_gib_s_median": statistics.median(rates),
        "rank_gib_s_max": max(rates),
    }
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
