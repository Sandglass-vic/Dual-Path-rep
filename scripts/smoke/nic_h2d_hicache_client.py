#!/usr/bin/env python3
"""Client half of nic_h2d_hicache_e2e.sh: evict a prompt to the host pool and reload it."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

PROMPT_WORDS = 1500
EVICTORS = 4


def post(port: int, payload: dict) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def wait_ready(port: int, server_pid: int) -> None:
    deadline = time.time() + 900
    while time.time() < deadline:
        if not Path(f"/proc/{server_pid}").exists():
            raise SystemExit("server exited during startup")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5):
                return
        except OSError:
            time.sleep(3)
    raise SystemExit("server did not become ready")


def prompt(seed: int) -> str:
    words = [f"w{(seed * 7919 + i * 104729) % 100003}" for i in range(PROMPT_WORDS)]
    return "Repeat the last three words exactly. " + " ".join(words)


def generate(port: int, text: str) -> tuple[str, dict]:
    result = post(port, {
        "text": text,
        "sampling_params": {"temperature": 0, "max_new_tokens": 24},
    })
    return result["text"], result["meta_info"]


def rx_write_requests(hca: str) -> int:
    return int(Path(f"/sys/class/infiniband/{hca}/ports/1/hw_counters/rx_write_requests").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--hca", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    args = parser.parse_args()
    wait_ready(args.port, args.server_pid)

    target = prompt(1)
    before = rx_write_requests(args.hca)
    first_text, first_meta = generate(args.port, target)
    for seed in range(2, 2 + EVICTORS):
        generate(args.port, prompt(seed))
    # Write-through backups run asynchronously; let them drain before reloading.
    time.sleep(2)
    reload_text, reload_meta = generate(args.port, target)
    after = rx_write_requests(args.hca)
    print("E2E_RESULT " + json.dumps({
        "label": args.label,
        "mode": os.environ.get("DUAL_PATH_CUDAMEMCPY"),
        "prompt_tokens": first_meta["prompt_tokens"],
        "first_cached_tokens": first_meta.get("cached_tokens"),
        "reload_cached_tokens": reload_meta.get("cached_tokens"),
        "reload_matches_first": reload_text == first_text,
        "first_text": first_text,
        "hca_rx_write_requests": after - before,
    }), flush=True)


if __name__ == "__main__":
    main()
