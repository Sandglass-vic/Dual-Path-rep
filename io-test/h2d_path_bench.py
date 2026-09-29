#!/usr/bin/env python3
"""Compare HiCache's copy-engine H2D/D2H with a NIC-driven copy on one node.

Buffers mirror one Qwen2.5-32B TP4 rank (64 layers, 2 KV heads x 128, bf16,
page 64): a GPU KV pool in SGLang's layer_first device layout and a pinned host
pool in either of the two host layouts the project runs:

  layer_first        Prefill HiCache   (--hicache-mem-layout layer_first)
  page_first_direct  Decode HiCache    (--hicache-mem-layout page_first_direct)

Modes (``--modes``):

  hicache         sgl_kernel direct path, exactly what --hicache-io-backend
                  direct calls: cudaMemcpyAsync per contiguous run
  hicache-kernel  sgl_kernel zero-copy kernels (io_backend=kernel); layer_first only
  nic             Mooncake Transfer Engine RDMA loopback on the HCA paired with
                  the GPU: one request per contiguous run, host and GPU pools
                  registered once

H2D is issued per layer, as ``load_to_device_per_layer``; D2H covers all layers
in one call, as ``backup_from_device_all_layer``.

One worker process per GPU in ``--gpus`` runs every case at the same time. The
parent never touches CUDA; it brackets each case with HCA counter snapshots and
integrates ``nvidia-smi dmon`` PCIe samples, so each result states whether its
bytes crossed the HCA and the GPU link. Each case repeats until it has run for
``--min-case-seconds`` so the 1 s dmon samples cover it.
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_path_counters import DmonSampler, hca_delta, hca_snapshot, verdict  # noqa: E402

RESULT_PREFIX = "H2D_PATH_RESULT "
DEFAULT_HCA_MAP = {"0": "mlx5_0", "1": "mlx5_0", "2": "mlx5_4", "3": "mlx5_4"}
BF16_BYTES = 2
VERIFY_PAGES = 4


def hca_map() -> dict[str, str]:
    raw = os.environ.get("DUALPATH_PZA100_IB_DEVICE_MAP", "")
    return json.loads(raw) if raw else DEFAULT_HCA_MAP


def local_cpus(hca: str) -> set[int]:
    path = Path(f"/sys/class/infiniband/{hca}/device/local_cpulist")
    cpus: set[int] = set()
    for part in path.read_text().strip().split(","):
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return cpus


def build_cases(args) -> list[dict]:
    cases = []
    for layout in args.layouts:
        for pattern in args.patterns:
            for direction in args.directions:
                for mode in args.modes:
                    if mode == "hicache-kernel" and layout != "layer_first":
                        continue
                    cases.append(
                        {"layout": layout, "pattern": pattern, "direction": direction, "mode": mode}
                    )
    return cases


def select_pages(pattern: str, pages: int, host_pages: int, dev_pages: int, seed: int):
    import numpy as np

    if pattern == "contiguous":
        return np.arange(pages, dtype=np.int64), np.arange(pages, dtype=np.int64)
    rng = np.random.default_rng(seed)
    host = rng.choice(host_pages, size=pages, replace=False).astype(np.int64)
    dev = rng.choice(dev_pages, size=pages, replace=False).astype(np.int64)
    return host, dev


def runs(host_pages, dev_pages, mergeable: bool):
    """Split page pairs into runs that are contiguous on both sides."""
    import numpy as np

    if not mergeable or len(host_pages) == 0:
        return host_pages, dev_pages, np.ones(len(host_pages), dtype=np.int64)
    breaks = np.flatnonzero(
        (np.diff(host_pages) != 1) | (np.diff(dev_pages) != 1)
    ) + 1
    starts = np.concatenate(([0], breaks))
    lengths = np.diff(np.concatenate((starts, [len(host_pages)])))
    return host_pages[starts], dev_pages[starts], lengths


class Pools:
    def __init__(self, args, layout: str, device):
        import torch

        self.torch = torch
        self.args = args
        self.layout = layout
        self.layers = args.layers
        self.page_size = args.page_size
        self.host_pages = args.host_pages
        self.dev_pages = args.dev_pages
        self.token_bytes = args.kv_heads * args.head_dim * BF16_BYTES
        shape = (args.kv_heads, args.head_dim)
        dev_tokens = self.dev_pages * self.page_size
        self.k_dev = torch.zeros((self.layers, dev_tokens, *shape), dtype=torch.bfloat16, device=device)
        self.v_dev = torch.zeros_like(self.k_dev)
        if layout == "layer_first":
            host_shape = (2, self.layers, self.host_pages * self.page_size, *shape)
        else:
            host_shape = (2, self.host_pages, self.layers, self.page_size, *shape)
        self.kv_host = torch.zeros(host_shape, dtype=torch.bfloat16, pin_memory=True)
        self.k_host, self.v_host = self.kv_host[0], self.kv_host[1]
        self.k_dev_layers = [self.k_dev[layer] for layer in range(self.layers)]
        self.v_dev_layers = [self.v_dev[layer] for layer in range(self.layers)]
        if layout == "layer_first":
            self.k_host_layers = [self.k_host[layer] for layer in range(self.layers)]
            self.v_host_layers = [self.v_host[layer] for layer in range(self.layers)]
        self.dev_ptrs = {
            "k": torch.tensor([t.data_ptr() for t in self.k_dev_layers], dtype=torch.uint64, device=device),
            "v": torch.tensor([t.data_ptr() for t in self.v_dev_layers], dtype=torch.uint64, device=device),
        }
        if layout == "layer_first":
            self.host_ptrs = {
                "k": torch.tensor([t.data_ptr() for t in self.k_host_layers], dtype=torch.uint64, device=device),
                "v": torch.tensor([t.data_ptr() for t in self.v_host_layers], dtype=torch.uint64, device=device),
            }

    def host_page(self, kv: int, layer: int, page: int):
        if self.layout == "layer_first":
            start = page * self.page_size
            return self.kv_host[kv, layer, start : start + self.page_size]
        return self.kv_host[kv, page, layer]

    def dev_page(self, kv: int, layer: int, page: int):
        start = page * self.page_size
        buffer = self.k_dev if kv == 0 else self.v_dev
        return buffer[layer, start : start + self.page_size]

    # Byte addresses used by the NIC path.
    def host_addr(self, kv: int, layer: int, pages):
        base = self.kv_host.data_ptr()
        if self.layout == "layer_first":
            tokens = (kv * self.layers + layer) * self.host_pages * self.page_size + pages * self.page_size
        else:
            tokens = ((kv * self.host_pages + pages) * self.layers + layer) * self.page_size
        return base + tokens * self.token_bytes

    def dev_addr(self, kv: int, layer: int, pages):
        base = (self.k_dev if kv == 0 else self.v_dev).data_ptr()
        tokens = layer * self.dev_pages * self.page_size + pages * self.page_size
        return base + tokens * self.token_bytes


class HicacheCopier:
    def __init__(self, pools: Pools, host_pages, dev_pages, kernel: bool, device):
        import torch
        from sgl_kernel import kvcacheio

        self.kv = kvcacheio
        self.pools = pools
        self.kernel = kernel
        ps = pools.page_size
        order = host_pages.argsort() if pools.layout == "layer_first" else slice(None)
        host_sorted, dev_sorted = host_pages[order], dev_pages[order]
        offsets = torch.arange(ps, dtype=torch.int64)
        host_idx = (torch.from_numpy(host_sorted)[:, None] * ps + offsets).reshape(-1)
        dev_idx = (torch.from_numpy(dev_sorted)[:, None] * ps + offsets).reshape(-1)
        if kernel:
            host_idx, dev_idx = host_idx.to(device), dev_idx.to(device)
        self.host_idx, self.dev_idx = host_idx, dev_idx
        self.requests_per_layer = 2 * len(runs(host_sorted, dev_sorted, pools.layout == "layer_first")[2])

    def h2d_layer(self, layer: int) -> None:
        p = self.pools
        if self.kernel:
            self.kv.transfer_kv_per_layer(
                src_k=p.k_host[layer], dst_k=p.k_dev[layer],
                src_v=p.v_host[layer], dst_v=p.v_dev[layer],
                src_indices=self.host_idx, dst_indices=self.dev_idx,
                item_size=p.token_bytes,
            )
        elif p.layout == "layer_first":
            self.kv.transfer_kv_direct(
                src_layers=[p.k_host_layers[layer], p.v_host_layers[layer]],
                dst_layers=[p.k_dev_layers[layer], p.v_dev_layers[layer]],
                src_indices=self.host_idx, dst_indices=self.dev_idx,
                page_size=p.page_size,
            )
        else:
            self.kv.transfer_kv_per_layer_direct_pf_lf(
                src_ptrs=[p.k_host, p.v_host],
                dst_ptrs=[p.k_dev_layers[layer], p.v_dev_layers[layer]],
                src_indices=self.host_idx, dst_indices=self.dev_idx,
                layer_id=layer, page_size=p.page_size,
            )

    def d2h_all(self) -> None:
        p = self.pools
        if self.kernel:
            self.kv.transfer_kv_all_layer(
                src_k_layers=p.dev_ptrs["k"], dst_k_layers=p.host_ptrs["k"],
                src_v_layers=p.dev_ptrs["v"], dst_v_layers=p.host_ptrs["v"],
                src_indices=self.dev_idx, dst_indices=self.host_idx,
                item_size=p.token_bytes, num_layers=p.layers,
            )
        elif p.layout == "layer_first":
            self.kv.transfer_kv_direct(
                src_layers=p.k_dev_layers + p.v_dev_layers,
                dst_layers=p.k_host_layers + p.v_host_layers,
                src_indices=self.dev_idx, dst_indices=self.host_idx,
                page_size=p.page_size,
            )
        else:
            self.kv.transfer_kv_all_layer_direct_lf_pf(
                src_ptrs=p.k_dev_layers + p.v_dev_layers,
                dst_ptrs=[p.k_host, p.v_host],
                src_indices=self.dev_idx, dst_indices=self.host_idx,
                page_size=p.page_size,
            )


class NicEngine:
    """One Mooncake Transfer Engine per worker, targeting its own segment."""

    def __init__(self, hca: str, host: str):
        from mooncake.engine import TransferEngine

        self.engine = TransferEngine()
        if self.engine.initialize(host, "P2PHANDSHAKE", "rdma", hca) != 0:
            raise RuntimeError(f"Mooncake initialize failed on {hca}")
        self.session = f"{host}:{self.engine.get_rpc_port()}"
        self.registered: list[int] = []

    def register(self, tensors) -> None:
        self.unregister()
        ptrs = [t.data_ptr() for t in tensors]
        lengths = [t.numel() * t.element_size() for t in tensors]
        if self.engine.batch_register_memory(ptrs, lengths) != 0:
            raise RuntimeError("Mooncake batch_register_memory failed")
        self.registered = ptrs

    def unregister(self) -> None:
        if self.registered:
            self.engine.batch_unregister_memory(self.registered)
            self.registered = []


class NicCopier:
    def __init__(self, engine: NicEngine, pools: Pools, host_pages, dev_pages, op: str, max_batch: int):
        self.engine = engine
        self.op = op
        self.max_batch = max_batch
        started = time.perf_counter()
        host_starts, dev_starts, lengths = runs(host_pages, dev_pages, pools.layout == "layer_first")
        nbytes = (lengths * pools.page_size * pools.token_bytes).tolist()
        self.h2d: list[tuple[list[int], list[int], list[int]]] = []
        for layer in range(pools.layers):
            host_addrs, dev_addrs, sizes = [], [], []
            for kv in (0, 1):
                host_addrs += pools.host_addr(kv, layer, host_starts).tolist()
                dev_addrs += pools.dev_addr(kv, layer, dev_starts).tolist()
                sizes += nbytes
            self.h2d.append((host_addrs, dev_addrs, sizes))
        self.all_host = [a for h, _, _ in self.h2d for a in h]
        self.all_dev = [a for _, d, _ in self.h2d for a in d]
        self.all_sizes = [s for _, _, z in self.h2d for s in z]
        self.plan_ms = (time.perf_counter() - started) * 1e3
        self.requests_per_layer = len(self.h2d[0][2])

    def _transfer(self, local, peer, sizes, write: bool) -> None:
        call = (
            self.engine.engine.batch_transfer_sync_write
            if write
            else self.engine.engine.batch_transfer_sync_read
        )
        step = self.max_batch or len(sizes)
        for begin in range(0, len(sizes), step):
            end = begin + step
            if call(self.engine.session, local[begin:end], peer[begin:end], sizes[begin:end]) < 0:
                raise RuntimeError("Mooncake batch transfer failed")

    def h2d_layer(self, layer: int) -> None:
        host, dev, sizes = self.h2d[layer]
        if self.op == "write":
            self._transfer(host, dev, sizes, write=True)
        else:
            self._transfer(dev, host, sizes, write=False)

    def d2h_all(self) -> None:
        if self.op == "write":
            self._transfer(self.all_dev, self.all_host, self.all_sizes, write=True)
        else:
            self._transfer(self.all_host, self.all_dev, self.all_sizes, write=False)


def verify(pools: Pools, copier, direction: str, host_pages, dev_pages) -> bool:
    torch = pools.torch
    picks = list(range(min(VERIFY_PAGES, len(host_pages)))) + list(
        range(max(0, len(host_pages) - VERIFY_PAGES), len(host_pages))
    )
    layers = sorted({0, pools.layers // 2, pools.layers - 1})
    for kv in (0, 1):
        for layer in layers:
            for pick in picks:
                hp, dp = int(host_pages[pick]), int(dev_pages[pick])
                src = pools.host_page(kv, layer, hp) if direction == "h2d" else pools.dev_page(kv, layer, dp)
                dst = pools.dev_page(kv, layer, dp) if direction == "h2d" else pools.host_page(kv, layer, hp)
                src.copy_(torch.randn(src.shape, dtype=src.dtype, device=src.device))
                dst.zero_()
    torch.cuda.synchronize()
    if direction == "h2d":
        for layer in range(pools.layers):
            copier.h2d_layer(layer)
    else:
        copier.d2h_all()
    torch.cuda.synchronize()
    for kv in (0, 1):
        for layer in layers:
            for pick in picks:
                hp, dp = int(host_pages[pick]), int(dev_pages[pick])
                host = pools.host_page(kv, layer, hp).to(pools.k_dev.device)
                if not torch.equal(host, pools.dev_page(kv, layer, dp)):
                    return False
    return True


def timed_rep(pools: Pools, copier, direction: str, stream):
    torch = pools.torch
    is_nic = isinstance(copier, NicCopier)
    layer_s: list[float] = []
    if is_nic:
        started = time.perf_counter()
        if direction == "h2d":
            for layer in range(pools.layers):
                t0 = time.perf_counter()
                copier.h2d_layer(layer)
                layer_s.append(time.perf_counter() - t0)
        else:
            copier.d2h_all()
        total = time.perf_counter() - started
        return total, total, layer_s
    events = [torch.cuda.Event(enable_timing=True) for _ in range(pools.layers + 1)]
    submit_started = time.perf_counter()
    with torch.cuda.stream(stream):
        events[0].record(stream)
        if direction == "h2d":
            for layer in range(pools.layers):
                copier.h2d_layer(layer)
                events[layer + 1].record(stream)
        else:
            copier.d2h_all()
            events[-1].record(stream)
    submit = time.perf_counter() - submit_started
    stream.synchronize()
    total = events[0].elapsed_time(events[-1]) / 1e3
    if direction == "h2d":
        layer_s = [events[i].elapsed_time(events[i + 1]) / 1e3 for i in range(pools.layers)]
    return total, submit, layer_s


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def worker(rank: int, gpu: int, args, cases, barrier, results) -> None:
    try:
        run_worker(rank, gpu, args, cases, barrier, results)
    except BaseException:
        barrier.abort()
        raise


def run_worker(rank: int, gpu: int, args, cases, barrier, results) -> None:
    import torch

    hca = hca_map()[str(gpu)]
    # Pinned host pages land on the NUMA node of the thread that first touches
    # them, so bind to the HCA's CPUs when the Slurm step allows any of them.
    numa_cpus = local_cpus(hca) & os.sched_getaffinity(0)
    if numa_cpus:
        os.sched_setaffinity(0, numa_cpus)
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    stream = torch.cuda.Stream(device=device)
    engine = NicEngine(hca, args.nic_host) if "nic" in args.modes else None
    pools = None
    bytes_per_rep = 2 * args.layers * args.pages * args.page_size * args.kv_heads * args.head_dim * BF16_BYTES

    for index, case in enumerate(cases):
        if pools is None or pools.layout != case["layout"]:
            if engine:
                engine.unregister()
            pools = None
            torch.cuda.empty_cache()
            pools = Pools(args, case["layout"], device)
            if engine:
                engine.register([pools.kv_host, pools.k_dev, pools.v_dev])
        host_pages, dev_pages = select_pages(
            case["pattern"], args.pages, args.host_pages, args.dev_pages, args.seed + rank
        )
        if case["mode"] == "nic":
            copier = NicCopier(engine, pools, host_pages, dev_pages, args.nic_op, args.nic_max_batch)
        else:
            copier = HicacheCopier(pools, host_pages, dev_pages, case["mode"] == "hicache-kernel", device)
        ok = verify(pools, copier, case["direction"], host_pages, dev_pages)
        for _ in range(args.warmup):
            timed_rep(pools, copier, case["direction"], stream)

        barrier.wait()
        cpu_started = time.process_time()
        wall_started = time.time()
        totals, submits, layers = [], [], []
        while len(totals) < args.reps or time.time() - wall_started < args.min_case_seconds:
            total, submit, layer_s = timed_rep(pools, copier, case["direction"], stream)
            totals.append(total)
            submits.append(submit)
            layers += layer_s
        wall_ended = time.time()
        cpu_s = time.process_time() - cpu_started
        barrier.wait()
        # Hold the next case's setup and verify transfers until the parent has
        # taken its closing counter snapshot.
        barrier.wait()

        median = statistics.median(totals)
        results.put({
            **case,
            "case_index": index,
            "rank": rank,
            "gpu": gpu,
            "hca": hca,
            "numa_local": bool(numa_cpus),
            "concurrent_gpus": len(args.gpus),
            "bytes_per_rep": bytes_per_rep,
            "reps": len(totals),
            "total_s_median": median,
            "total_s_min": min(totals),
            "gbytes_s": bytes_per_rep / median / 1e9,
            "layer_ms_p50": percentile(layers, 0.5) * 1e3 if layers else None,
            "layer_ms_p99": percentile(layers, 0.99) * 1e3 if layers else None,
            "submit_ms_median": statistics.median(submits) * 1e3,
            "cpu_s_per_rep": cpu_s / len(totals),
            "requests_per_layer": copier.requests_per_layer,
            "plan_ms": getattr(copier, "plan_ms", None),
            "verified": ok,
            "wall_started": wall_started,
            "wall_ended": wall_ended,
            "nic_op": args.nic_op if case["mode"] == "nic" else None,
            "mc_ib_sl": os.environ.get("MC_IB_SL") if case["mode"] == "nic" else None,
        })
    if engine:
        engine.unregister()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default="0", help="comma-separated GPU indices, one worker each")
    parser.add_argument("--modes", default="hicache,hicache-kernel,nic")
    parser.add_argument("--layouts", default="layer_first,page_first_direct")
    parser.add_argument("--patterns", default="contiguous,random")
    parser.add_argument("--directions", default="h2d,d2h")
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--pages", type=int, default=512, help="pages per transfer (512 = 32K tokens)")
    parser.add_argument("--host-pages", type=int, default=1024)
    parser.add_argument("--dev-pages", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--min-case-seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--nic-op", choices=("write", "read"), default="write")
    parser.add_argument("--nic-max-batch", type=int, default=0, help="requests per Mooncake call, 0 = all")
    parser.add_argument("--nic-host", default=os.environ.get("NIC_BENCH_HOST", "127.0.0.1"))
    parser.add_argument("--output", type=Path, help="append result lines to this JSONL file")
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    args.gpus = [int(gpu) for gpu in args.gpus.split(",") if gpu]
    for name in ("modes", "layouts", "patterns", "directions"):
        setattr(args, name, [item for item in getattr(args, name).split(",") if item])
    if args.pages > min(args.host_pages, args.dev_pages):
        parser.error("--pages must not exceed --host-pages or --dev-pages")

    cases = build_cases(args)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(args.gpus) + 1)
    results = ctx.Queue()
    procs = [
        ctx.Process(target=worker, args=(rank, gpu, args, cases, barrier, results), daemon=True)
        for rank, gpu in enumerate(args.gpus)
    ]
    for proc in procs:
        proc.start()

    hcas = sorted({hca_map()[str(gpu)] for gpu in args.gpus})
    dmon = DmonSampler()
    sink = args.output.open("a") if args.output else None
    host = socket.gethostname()
    try:
        for case in cases:
            barrier.wait(timeout=900)
            before = hca_snapshot(hcas)
            started = time.time()
            barrier.wait(timeout=900)
            ended = time.time()
            after = hca_snapshot(hcas)
            barrier.wait(timeout=900)
            records = [results.get(timeout=600) for _ in args.gpus]
            expected = sum(r["bytes_per_rep"] * r["reps"] for r in records)
            delta = hca_delta(before, after)
            time.sleep(1.2)
            pcie = dmon.summary(started, ended)
            case_verdict = verdict(delta, pcie, [str(g) for g in args.gpus], expected)
            for record in records:
                record.update({
                    "host": host,
                    "label": args.label,
                    "case_wall_s": ended - started,
                    "hca_delta": delta,
                    "gpu_pcie": pcie,
                    "verdict": case_verdict,
                    "aggregate_gbytes_s": expected / (ended - started) / 1e9,
                })
                line = json.dumps(record, sort_keys=True)
                print(RESULT_PREFIX + line, flush=True)
                if sink:
                    sink.write(line + "\n")
                    sink.flush()
    finally:
        dmon.stop()
        if sink:
            sink.close()
    for proc in procs:
        proc.join(timeout=60)
    return 0 if all(proc.exitcode == 0 for proc in procs) else 1


if __name__ == "__main__":
    sys.exit(main())
