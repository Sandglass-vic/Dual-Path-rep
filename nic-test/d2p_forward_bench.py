#!/usr/bin/env python3
"""Prefill / Decode forward slowdown under D->P RDMA (g6 host staging -> g5 GPU).

One task per node (Slurm node 0 = g5 = Prefill, node 1 = g6 = Decode). Each task
spawns, per GPU, one probe rank and one traffic generator.

Probes run a full Qwen2.5-32B TP4 forward per iteration (hidden 5120, 64 layers,
40 Q / 8 KV heads, head dim 128, intermediate 27648, vocab 152064, bf16; per
rank 10 Q / 2 KV heads, 6912 intermediate). Every layer: QKV GEMM, attention
over the cached context, O GEMM + all-reduce, gate/up GEMM, SiLU, down GEMM +
all-reduce; plus the embedding all-reduce and the logits GEMM + all-gather.
One layer's weights are reused for all 64 layers (each layer still streams
~240 MiB of weights from HBM, far above L2).

  prefill (g5)  T = 1004 new tokens attending to a 19,630-token context, eager
  decode  (g6)  B = 6 sequences x 1 token attending to 19,782 tokens each,
                captured in a CUDA graph

D->P traffic mirrors ``_send_kvcache_generic`` for a Decode-path prefix
(``executor=None``): one ``batch_transfer_sync_write`` per request carrying 64
layers x K/V = 128 blocks, each block the request's contiguous prefix pages for
that layer (18,626 hit tokens -> 291 pages x 32 KiB = 9.1 MiB, 1.19 GB per rank).
Source is pinned host memory on g6, destination GPU memory on g5, each GPU
through its PIX CX8. Several threads per GPU keep requests in flight so the
offered load can exceed what one synchronous caller reaches.

Load levels are the D->P rate per GPU in GB/s. Requests are released on a
schedule that averages that rate; each request itself goes at full speed, like
the real per-request bursts. A level above what the path sustains runs flat out;
the achieved rate is recorded.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import socket
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from allreduce_interference_bench import (  # noqa: E402
    HCA_MAP,
    NODE_IP,
    ROLE,
    Generator,
    bind_numa,
    hca_delta,
    hca_snapshot,
    synchronize,
    wait_until,
)

HIDDEN = 5120
LAYERS = 64
HEAD_DIM = 128
Q_HEADS = 40 // 4
KV_HEADS = 8 // 4
INTERMEDIATE = 27648 // 4
VOCAB_PER_RANK = 152064 // 4
BF16 = 2
PAGE_TOKENS = 64
KV_BYTES_PER_TOKEN_LAYER = KV_HEADS * HEAD_DIM * BF16  # one of K or V, per rank


# --------------------------------------------------------------------- probe
def build_forward(torch, dist, role: str, args):
    dev = torch.device("cuda")
    bf16 = torch.bfloat16
    tokens = args.prefill_tokens if role == "prefill" else args.decode_batch

    def rand(*shape):
        return torch.randn(*shape, dtype=bf16, device=dev) * 0.02

    x = rand(tokens, HIDDEN)
    embed = torch.empty_like(x)
    w_qkv = rand(HIDDEN, (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM)
    w_o = rand(Q_HEADS * HEAD_DIM, HIDDEN)
    w_gate_up = rand(HIDDEN, 2 * INTERMEDIATE)
    w_down = rand(INTERMEDIATE, HIDDEN)
    w_lm = rand(HIDDEN, VOCAB_PER_RANK)
    if role == "prefill":
        k_cache = rand(1, KV_HEADS, args.prefill_context, HEAD_DIM)
        logits_rows = 1
    else:
        k_cache = rand(tokens, KV_HEADS, args.decode_context, HEAD_DIM)
        logits_rows = tokens
    v_cache = torch.randn_like(k_cache)
    ag_out = torch.empty(4 * logits_rows, VOCAB_PER_RANK, dtype=bf16, device=dev)
    sdpa = torch.nn.functional.scaled_dot_product_attention
    q_size = Q_HEADS * HEAD_DIM

    def attention(qkv):
        q = qkv[:, :q_size]
        if role == "prefill":
            q = q.view(1, tokens, Q_HEADS, HEAD_DIM).transpose(1, 2)
        else:
            q = q.view(tokens, 1, Q_HEADS, HEAD_DIM).transpose(1, 2)
        return sdpa(q, k_cache, v_cache, enable_gqa=True)

    def forward(comm: bool = True) -> None:
        # Every layer reads the same x, so values stay bounded across iterations.
        if comm:
            embed.copy_(x)
            dist.all_reduce(embed)  # vocab-parallel embedding
        for _ in range(LAYERS):
            attn = attention(x @ w_qkv)
            o = attn.transpose(1, 2).reshape(tokens, q_size) @ w_o
            if comm:
                dist.all_reduce(o)
            gate, up = (x @ w_gate_up).chunk(2, dim=-1)
            d = (torch.nn.functional.silu(gate) * up) @ w_down
            if comm:
                dist.all_reduce(d)
        logits = x[-logits_rows:] @ w_lm
        if comm:
            dist.all_gather_into_tensor(ag_out, logits)

    return forward, tokens


def probe_rank(rank: int, node: int, args, cases, barriers, release, results) -> None:
    import torch
    import torch.distributed as dist

    bind_numa(rank)
    torch.cuda.set_device(rank)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(args.probe_port))
    dist.init_process_group("nccl", rank=rank, world_size=4, device_id=torch.device("cuda", rank))
    role = ROLE[node]
    forward, tokens = build_forward(torch, dist, role, args)

    def timed(fn, count: int) -> float:
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(count):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - started) / count * 1e3

    for _ in range(3):
        forward()
    compute_only_ms = timed(lambda: forward(comm=False), 5)

    run = forward
    graph_used = False
    if role == "decode" and args.decode_graph:
        try:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
            run = graph.replay
            graph_used = True
        except Exception as exc:  # noqa: BLE001
            print(f"probe rank {rank}: CUDA graph capture failed, eager: {exc}", flush=True)
            torch.cuda.synchronize()
    for _ in range(3):
        run()
    # Every count that drives collectives must be identical on all ranks.
    def agreed_ms(count: int) -> float:
        value = torch.tensor([timed(run, count)], device="cuda")
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return float(value.item())

    rough = agreed_ms(2)
    per_iter = agreed_ms(max(3, int(2000 / max(1.0, rough))))
    iters = max(5, int(args.measure_seconds * 1e3 / per_iter))
    barriers[0].wait()
    barriers[1].wait()
    base = release.value

    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    stop_flag = torch.zeros(1, dtype=torch.int32, device="cuda")

    def past(deadline: float) -> bool:
        # All ranks must stop at the same iteration or the collectives desync.
        stop_flag.fill_(int(time.time() > deadline))
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        return bool(stop_flag.item())

    for index, case in enumerate(cases):
        case_start = base + index * args.case_seconds
        measure_start = case_start + args.warm_seconds
        # Generators stop 1 s before the case ends; never measure past that.
        deadline = case_start + args.case_seconds - 1.5
        wait_until(int((measure_start - args.probe_warmup_seconds) * 1e9))
        warmup = 0
        while True:
            run()
            warmup += 1
            if past(measure_start):
                break
        window_start = time.time()
        done = 0
        for begin, end in events:
            begin.record()
            run()
            end.record()
            done += 1
            if past(deadline):
                break
        torch.cuda.synchronize()
        window_end = time.time()
        times = [b.elapsed_time(e) for b, e in events[:done]]
        if rank == 0:
            ordered = sorted(times)
            results.put({
                "kind": "probe", "case_index": index, **case, "node": node, "role": role,
                "tokens": tokens, "graph": graph_used, "compute_only_ms": compute_only_ms,
                "iters": done, "iters_planned": iters, "warmup_iters": warmup,
                "iter_ms_mean": statistics.mean(times), "iter_ms_p50": ordered[len(ordered) // 2],
                "iter_ms_p99": ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))],
                "iter_ms": [round(t, 4) for t in times],
                "window_start": window_start, "window_end": window_end,
                "overran": done < iters,
            })
    results.close()
    results.join_thread()
    # Skip NCCL / CUDA teardown: a peer that already exited can hang it.
    os._exit(0)


# ----------------------------------------------------------------- generator
def d2p_case(gen: Generator, args, rate_gbps: float, start: float, stop: float,
             window: tuple[float, float]) -> dict:
    """Paced real-shaped D->P requests from several threads until ``stop``."""
    pages = math.ceil(args.d2p_prefix_tokens / PAGE_TOKENS)
    block = pages * PAGE_TOKENS * KV_BYTES_PER_TOKEN_LAYER
    blocks = LAYERS * 2
    request_bytes = block * blocks
    span = gen.size - block
    src_base, dst_base = gen.host.data_ptr(), gen.peer["dev"]
    session = gen.peer["session"]
    interval = request_bytes / (rate_gbps * 1e9)
    lock = threading.Lock()
    state = {"next": start, "done_bytes": 0, "requests": 0, "busy_s": 0.0, "failed": 0}

    def worker(tid: int) -> None:
        cursor = tid * block * 7 % span
        while True:
            with lock:
                slot = max(state["next"], time.time())
                state["next"] = slot + interval
            if slot >= stop:
                return
            delay = slot - time.time()
            if delay > 0:
                time.sleep(delay)
            srcs, dsts = [], []
            for _ in range(blocks):
                srcs.append(src_base + cursor)
                dsts.append(dst_base + cursor)
                cursor = (cursor + block) % span
            began = time.time()
            ret = gen.engine.batch_transfer_sync_write(session, srcs, dsts, [block] * blocks)
            ended = time.time()
            with lock:
                if ret < 0:
                    state["failed"] += 1
                elif window[0] <= ended <= window[1]:
                    state["done_bytes"] += request_bytes
                    state["requests"] += 1
                    state["busy_s"] += ended - began

    wait_until(int(start * 1e9))
    threads = [threading.Thread(target=worker, args=(t,), daemon=True) for t in range(args.d2p_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if state["failed"]:
        raise RuntimeError(f"{state['failed']} D->P transfers failed")
    seconds = window[1] - window[0]
    return {
        "target_gbytes_s": rate_gbps,
        "achieved_gbytes_s": state["done_bytes"] / seconds / 1e9,
        "requests": state["requests"],
        "request_bytes": request_bytes,
        "block_bytes": block,
        "mean_request_ms": state["busy_s"] / state["requests"] * 1e3 if state["requests"] else None,
        "threads": args.d2p_threads,
    }


def generator(gpu: int, node: int, args, cases, barriers, release, results) -> None:
    bind_numa(gpu)
    gen = Generator(gpu, node, args)
    barriers[0].wait()
    barriers[1].wait()
    base = release.value
    for index, case in enumerate(cases):
        start = base + index * args.case_seconds
        stop = start + args.case_seconds - 1.0
        window = (start + args.warm_seconds, stop)
        if case["level"] == 0 or node != 1:
            continue
        record = d2p_case(gen, args, case["level"], start, stop, window)
        results.put({"kind": "generator", "case_index": index, **case, "node": node, "gpu": gpu,
                     "hca": gen.hca, **record})
    # The peer node writes into this engine's buffers; keep it registered until the end.
    wait_until(int((base + len(cases) * args.case_seconds) * 1e9))
    results.close()
    results.join_thread()
    # Mooncake's destructor can hang once the peer engine is gone.
    os._exit(0)


# -------------------------------------------------------------------- driver
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--levels", default="0,5,10,15,20,25", help="D->P GB/s per GPU")
    parser.add_argument("--case-seconds", type=float, default=25.0)
    parser.add_argument("--warm-seconds", type=float, default=4.0)
    parser.add_argument("--measure-seconds", type=float, default=8.0)
    parser.add_argument("--probe-warmup-seconds", type=float, default=1.5)
    parser.add_argument("--prefill-tokens", type=int, default=1004)
    parser.add_argument("--prefill-context", type=int, default=19630)
    parser.add_argument("--decode-batch", type=int, default=6)
    parser.add_argument("--decode-context", type=int, default=19782)
    parser.add_argument("--decode-graph", type=int, default=1)
    parser.add_argument("--d2p-prefix-tokens", type=int, default=18626)
    parser.add_argument("--d2p-threads", type=int, default=2)
    parser.add_argument("--buffer-bytes", type=int, default=1 << 30)
    parser.add_argument("--probe-port", type=int, default=29651)
    parser.add_argument("--barrier-port", type=int, default=29661)
    args = parser.parse_args()
    args.levels = [float(v) for v in args.levels.split(",") if v]
    node = int(os.environ.get("SLURM_NODEID", "0"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = [{"scheme": "d2p", "level": level} for level in args.levels]

    ctx = mp.get_context("spawn")
    barriers = (ctx.Barrier(9), ctx.Barrier(9))
    release = ctx.Value("d", 0.0)
    results = ctx.Queue()
    procs = [ctx.Process(target=probe_rank, args=(r, node, args, cases, barriers, release, results)) for r in range(4)]
    procs += [ctx.Process(target=generator, args=(g, node, args, cases, barriers, release, results)) for g in range(4)]
    for proc in procs:
        proc.start()

    barriers[0].wait()
    handshake = synchronize(2, node, NODE_IP[0], args.barrier_port, lead_ns=5_000_000_000)
    release.value = handshake["release_epoch_ns"] / 1e9
    barriers[1].wait()
    print(f"node {node} ({socket.gethostname()}) released; {len(cases)} cases x {args.case_seconds}s", flush=True)

    records = []
    hcas = sorted(set(HCA_MAP.values()))
    for index, case in enumerate(cases):
        start = release.value + index * args.case_seconds
        wait_until(int((start + args.warm_seconds) * 1e9))
        before = hca_snapshot(hcas)
        t0 = time.time()
        wait_until(int((start + args.case_seconds - 1.0) * 1e9))
        delta = hca_delta(before, hca_snapshot(hcas))
        seconds = time.time() - t0
        per_hca = {
            hca: {
                "tx_gbytes_s": v.get("port_xmit_data_bytes", 0) / seconds / 1e9,
                "rx_gbytes_s": v.get("port_rcv_data_bytes", 0) / seconds / 1e9,
            }
            for hca, v in delta.items()
        }
        records.append({"kind": "hca", "case_index": index, **case, "node": node, "hca": per_hca})
        print(f"node {node} case {index + 1}/{len(cases)} {case} hca={json.dumps(per_hca)}", flush=True)

    deadline = release.value + len(cases) * args.case_seconds + 120
    while any(proc.is_alive() for proc in procs) and time.time() < deadline:
        try:
            records.append(results.get(timeout=1))
        except Exception:  # noqa: BLE001
            pass
    for proc in procs:
        if proc.is_alive():
            print(f"node {node}: killing stuck child pid={proc.pid}", flush=True)
            proc.kill()
        proc.join()
    while not results.empty():
        records.append(results.get())
    path = out / f"node{node}.jsonl"
    with path.open("w") as sink:
        for record in records:
            sink.write(json.dumps(record, sort_keys=True) + "\n")
    failed = [p.exitcode for p in procs if p.exitcode not in (0, -9)]
    print(f"node {node} wrote {path}; failed children: {failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
