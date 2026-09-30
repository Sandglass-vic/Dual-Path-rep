#!/usr/bin/env python3
"""Validate DUAL_PATH_CUDAMEMCPY=nic against HiCache's stock direct copies.

Runs inside an image patched by scripts/infra/patch_sglang_nic_h2d.py, on one
GPU, with a Qwen2.5-32B TP4-shaped pool (64 layers, 2 KV heads x 128, bf16,
page 64) and 512 random pages (32K tokens) per transfer. For each host layout
(layer_first, page_first_direct) and each mode (nic, hicache) it drives the
real MHATokenToKVPoolHost methods the way SGLang does:

  h2d  load_to_device_per_layer for every layer on a load stream, recording one
       event per layer; a consumer stream waits on each layer's event and
       immediately compares that layer with the expected data. A broken stream
       protocol shows up as a mismatch, not a hang.
  d2h  backup_from_device_all_layer on a write stream, then finish_event.

It prints one SELFTEST_RESULT JSON line per case and exits non-zero on any
mismatch.

  DUAL_PATH_NIC_DEVICE=mlx5_0 python3 scripts/smoke/nic_h2d_selftest.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("DUAL_PATH_CUDAMEMCPY", "nic")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache import dualpath_nic_copy as nic  # noqa: E402
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost  # noqa: E402


def device_pool(args):
    tokens = args.dev_pages * args.page_size
    shape = (tokens + args.page_size, args.kv_heads, args.head_dim)
    k = [torch.zeros(shape, dtype=torch.bfloat16, device="cuda") for _ in range(args.layers)]
    v = [torch.zeros(shape, dtype=torch.bfloat16, device="cuda") for _ in range(args.layers)]
    return SimpleNamespace(
        store_dtype=torch.bfloat16, dtype=torch.bfloat16, size=tokens,
        page_size=args.page_size, start_layer=0, end_layer=args.layers - 1,
        head_num=args.kv_heads, head_dim=args.head_dim, v_head_dim=args.head_dim,
        layer_num=args.layers, device="cuda", k_buffer=k, v_buffer=v,
    )


def token_indices(pages, page_size):
    offsets = np.arange(page_size, dtype=np.int64)
    return torch.from_numpy((pages[:, None] * page_size + offsets).reshape(-1))


def host_rows(host, kv: int, layer: int, idx):
    buffer = host.k_buffer if kv == 0 else host.v_buffer
    if host.layout == "layer_first":
        return buffer[layer][idx]
    ps = host.page_size
    return buffer[idx // ps, layer, idx % ps]


def set_host_rows(host, kv: int, layer: int, idx, value) -> None:
    buffer = host.k_buffer if kv == 0 else host.v_buffer
    if host.layout == "layer_first":
        buffer[layer][idx] = value
    else:
        ps = host.page_size
        buffer[idx // ps, layer, idx % ps] = value


def run_h2d(host, dev, host_idx, dev_idx, expected, mode):
    nic.ENABLED = mode == "nic"
    dev_idx_gpu = dev_idx.cuda()
    for layer in range(host.layer_num):
        dev.k_buffer[layer].zero_()
        dev.v_buffer[layer].zero_()
    torch.cuda.synchronize()
    load = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    events = [torch.cuda.Event() for _ in range(host.layer_num)]
    mismatches = torch.zeros(1, dtype=torch.int64, device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(load)
    submit_started = time.perf_counter()
    with torch.cuda.stream(load):
        for layer in range(host.layer_num):
            host.load_to_device_per_layer(dev, host_idx, dev_idx, layer, "direct")
            events[layer].record(load)
    submit_s = time.perf_counter() - submit_started
    end.record(load)
    with torch.cuda.stream(consumer):
        for layer in range(host.layer_num):
            consumer.wait_event(events[layer])
            for kv, buffer in ((0, dev.k_buffer), (1, dev.v_buffer)):
                got = buffer[layer].index_select(0, dev_idx_gpu)
                mismatches += (got != expected[kv][layer]).sum()
    consumer.synchronize()
    load.synchronize()
    return int(mismatches.item()), start.elapsed_time(end) / 1e3, submit_s


def run_d2h(host, dev, host_idx, dev_idx, mode):
    nic.ENABLED = mode == "nic"
    dev_idx_gpu = dev_idx.cuda()
    for layer in range(host.layer_num):
        for kv, buffer in ((0, dev.k_buffer), (1, dev.v_buffer)):
            rows = torch.randn((len(dev_idx), host.device_pool.head_num, host.device_pool.head_dim),
                               dtype=torch.bfloat16, device="cuda")
            buffer[layer].index_copy_(0, dev_idx_gpu, rows)
            set_host_rows(host, kv, layer, host_idx, 0)
    torch.cuda.synchronize()
    write = torch.cuda.Stream()
    write.wait_stream(torch.cuda.current_stream())
    start, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(write)
    submit_started = time.perf_counter()
    with torch.cuda.stream(write):
        host.backup_from_device_all_layer(dev, host_idx, dev_idx, "direct")
    submit_s = time.perf_counter() - submit_started
    finish.record(write)
    finish.synchronize()
    mismatches = 0
    for layer in range(host.layer_num):
        for kv, buffer in ((0, dev.k_buffer), (1, dev.v_buffer)):
            want = buffer[layer].index_select(0, dev_idx_gpu).cpu()
            mismatches += int((host_rows(host, kv, layer, host_idx) != want).sum())
    return mismatches, start.elapsed_time(finish) / 1e3, submit_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layouts", default="layer_first,page_first_direct")
    parser.add_argument("--modes", default="nic,hicache")
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--pages", type=int, default=512)
    parser.add_argument("--dev-pages", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if not nic.ENABLED:
        raise SystemExit("run with DUAL_PATH_CUDAMEMCPY=nic so the host pools register with the NIC")

    rng = np.random.default_rng(args.seed)
    dev = device_pool(args)
    bytes_per_rep = 2 * args.layers * args.pages * args.page_size * args.kv_heads * args.head_dim * 2
    failures = 0
    # SGLang cudaHostRegisters host pools and never unregisters them, so keep
    # every pool alive: a freed range reused by the next pool fails to register.
    pools = []
    for layout in args.layouts.split(","):
        host = MHATokenToKVPoolHost(dev, 2.0, 0, args.page_size, layout, pin_memory=True)
        pools.append(host)
        host_pages = rng.choice(host.page_num - 1, size=args.pages, replace=False)
        dev_pages = rng.choice(args.dev_pages, size=args.pages, replace=False)
        host_idx, dev_idx = token_indices(host_pages, args.page_size), token_indices(dev_pages, args.page_size)
        if layout == "layer_first":
            # HiCacheController.move_indices sorts host indices for this layout.
            host_idx, order = host_idx.sort()
            dev_idx = dev_idx.index_select(0, order)
        expected = ([], [])
        for layer in range(args.layers):
            for kv in (0, 1):
                rows = torch.randn((len(host_idx), args.kv_heads, args.head_dim), dtype=torch.bfloat16)
                set_host_rows(host, kv, layer, host_idx, rows)
                expected[kv].append(rows.cuda())
        for direction in ("h2d", "d2h"):
            for mode in args.modes.split(","):
                for rep in range(args.reps):
                    if direction == "h2d":
                        bad, seconds, submit = run_h2d(host, dev, host_idx, dev_idx, expected, mode)
                    else:
                        bad, seconds, submit = run_d2h(host, dev, host_idx, dev_idx, mode)
                    failures += bad != 0
                    print("SELFTEST_RESULT " + json.dumps({
                        "layout": layout, "direction": direction, "mode": mode, "rep": rep,
                        "mismatched_elements": bad, "gpu_seconds": seconds,
                        "gbytes_s": bytes_per_rep / seconds / 1e9, "submit_ms": submit * 1e3,
                        "hca": nic.get_engine().hca, "submit": nic.SUBMIT,
                    }), flush=True)
        torch.cuda.synchronize()
    print("SELFTEST " + ("PASS" if failures == 0 else f"FAIL cases={failures}"), flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
