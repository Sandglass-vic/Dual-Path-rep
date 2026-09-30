#!/usr/bin/env python3
"""TP4 all-reduce slowdown under KV traffic on the CX8 compute fabric.

One task per node (Slurm node 0 = g5 = Prefill, node 1 = g6 = Decode). Each
task spawns, per GPU, one probe rank and one traffic generator.

Probes reproduce one engine's TP4 collectives for Qwen2.5-32B (hidden 5120,
64 layers, vocab 152064, bf16, NCCL; custom all-reduce is off at TP4):

  prefill (g5)  one forward: embedding + 2 per layer = 129 all-reduces of
                T x 10 KiB (T = 1004 appended tokens, 9.8 MiB), plus one
                logits all-gather of 1 x 38016 per rank; eager, like Prefill
  decode  (g6)  one step: 129 all-reduces of B x 10 KiB (B = 6, 60 KiB) plus
                one logits all-gather of B x 38016 per rank, captured in a
                CUDA graph like Decode

Traffic schemes (``--schemes``), mirroring what a turn moves:

  p2d          g5 GPU memory -> g6 host staging, RDMA write (Mooncake, sync batches)
  d2p          g6 host memory -> g5 GPU memory, RDMA write
  hicache_h2d  both nodes, pinned host -> GPU with cudaMemcpyAsync (copy engine)
  nic_h2d      both nodes, pinned host -> GPU through the paired CX8 with
               Mooncake batch_transfer_write_on_cuda (asynchronous, stream ordered)

Load levels are fractions of one GPU's share of the CX8 line rate: 800 Gb/s =
100 GB/s per HCA, two GPUs per HCA, so 100% = 50 GB/s per GPU. A scheme
that cannot reach the target runs flat out; the achieved rate is recorded.
hicache_h2d does not touch the HCA; its level is the equivalent byte rate.

Both nodes align to one timeline after a clock-offset barrier. Each case lasts
--case-seconds; traffic runs for the whole case. Probes run untimed for
--probe-warmup-seconds, then from --warm-seconds time the iteration count
calibrated at startup, cut short when the case window closes.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "io-test"))

from capture_path_counters import hca_delta, hca_snapshot  # noqa: E402
from cluster_barrier import synchronize, wait_until  # noqa: E402

HCA_MAP = {0: "mlx5_0", 1: "mlx5_0", 2: "mlx5_4", 3: "mlx5_4"}
NODE_IP = {0: "192.168.10.5", 1: "192.168.10.6"}
ROLE = {0: "prefill", 1: "decode"}
GPU_SHARE_PEAK_BYTES = 100e9 / 2  # CX8 XDR 800 Gb/s shared by 2 GPUs
HIDDEN = 5120
LAYERS = 64
VOCAB_PER_RANK = 152064 // 4
BF16 = 2
PACE_SECONDS = 0.01


def local_cpus(hca: str) -> set[int]:
    cpus: set[int] = set()
    for part in Path(f"/sys/class/infiniband/{hca}/device/local_cpulist").read_text().strip().split(","):
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    return cpus


def bind_numa(gpu: int) -> None:
    cpus = local_cpus(HCA_MAP[gpu]) & os.sched_getaffinity(0)
    if cpus:
        os.sched_setaffinity(0, cpus)


def build_cases(args) -> list[dict]:
    cases = []
    for scheme in args.schemes:
        for level in args.levels:
            cases.append({"scheme": scheme, "level": level})
    return cases


def scheme_active(scheme: str, node: int) -> bool:
    if scheme == "p2d":
        return node == 0
    if scheme == "d2p":
        return node == 1
    return True


# --------------------------------------------------------------------- probe
def probe_rank(rank: int, node: int, args, cases, barriers, release, results) -> None:
    import torch
    import torch.distributed as dist

    bind_numa(rank)
    torch.cuda.set_device(rank)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(args.probe_port))
    dist.init_process_group("nccl", rank=rank, world_size=4, device_id=torch.device("cuda", rank))
    role = ROLE[node]
    tokens = args.prefill_tokens if role == "prefill" else args.decode_batch
    logits_rows = 1 if role == "prefill" else args.decode_batch
    ar = torch.ones(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    ag_in = torch.ones(logits_rows, VOCAB_PER_RANK, dtype=torch.bfloat16, device="cuda")
    ag_out = torch.empty(4 * logits_rows, VOCAB_PER_RANK, dtype=torch.bfloat16, device="cuda")
    count = 1 + 2 * LAYERS

    def step() -> None:
        for _ in range(count):
            dist.all_reduce(ar)
        dist.all_gather_into_tensor(ag_out, ag_in)

    run = step
    graph_used = False
    if role == "decode" and args.decode_graph:
        try:
            for _ in range(3):
                step()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
            run = graph.replay
            graph_used = True
        except Exception as exc:  # noqa: BLE001
            print(f"probe rank {rank}: CUDA graph capture failed, eager: {exc}", flush=True)
            torch.cuda.synchronize()
    for _ in range(3):
        run()
    # Every count that drives collectives must be identical on all ranks.
    def agreed_seconds(count: int) -> float:
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(count):
            run()
        torch.cuda.synchronize()
        value = torch.tensor([(time.perf_counter() - started) / count], device="cuda")
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return float(value.item())

    rough = agreed_seconds(3)
    per_iter = agreed_seconds(max(3, int(2.0 / rough)))
    iters = max(5, int(args.measure_seconds / per_iter))
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
        # Untimed iterations right before the window, so the first timed one
        # does not pay the wake-up cost of an idle GPU / HCA / NCCL proxy.
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
                "iters": done, "iters_planned": iters, "graph": graph_used,
                "ar_bytes": ar.numel() * BF16, "ar_count": count,
                "ag_bytes_per_rank": ag_in.numel() * BF16,
                "iter_ms_mean": statistics.mean(times), "iter_ms_p50": ordered[len(ordered) // 2],
                "iter_ms_p99": ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))],
                "window_start": window_start, "window_end": window_end,
                "overran": done < iters, "warmup_iters": warmup,
                "iter_ms": [round(t, 4) for t in times],
            })
    results.close()
    results.join_thread()
    # Skip NCCL / CUDA teardown: a peer that already exited can hang it.
    os._exit(0)


# ----------------------------------------------------------------- generator
class Generator:
    def __init__(self, gpu: int, node: int, args):
        import torch
        from mooncake.engine import TransferEngine

        self.torch = torch
        self.gpu, self.node, self.args = gpu, node, args
        self.hca = HCA_MAP[gpu]
        torch.cuda.set_device(gpu)
        size = args.buffer_bytes
        self.size = size
        self.host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        self.dev = torch.empty(size, dtype=torch.uint8, device="cuda")
        self.stream = torch.cuda.Stream()
        self.engine = TransferEngine()
        ip = NODE_IP[node]
        if self.engine.initialize(ip, "P2PHANDSHAKE", "rdma", self.hca) != 0:
            raise RuntimeError(f"Mooncake initialize failed on {self.hca}")
        self.session = f"{ip}:{self.engine.get_rpc_port()}"
        if self.engine.batch_register_memory([self.host.data_ptr(), self.dev.data_ptr()], [size, size]) != 0:
            raise RuntimeError("batch_register_memory failed")
        rdv = Path(args.out) / "rendezvous"
        rdv.mkdir(parents=True, exist_ok=True)
        info = {"session": self.session, "host": self.host.data_ptr(), "dev": self.dev.data_ptr()}
        tmp = rdv / f".node{node}-gpu{gpu}.json"
        tmp.write_text(json.dumps(info))
        tmp.rename(rdv / f"node{node}-gpu{gpu}.json")
        peer = rdv / f"node{1 - node}-gpu{gpu}.json"
        deadline = time.time() + 600
        while True:
            try:
                self.peer = json.loads(peer.read_text())
                break
            except (OSError, ValueError):
                if time.time() > deadline:
                    raise RuntimeError(f"no peer rendezvous file {peer}")
                os.listdir(rdv)  # refresh the NFS directory cache
                time.sleep(0.5)

    def requests(self, nbytes: int, chunk: int, src_base: int, dst_base: int, cursor: int):
        count = max(1, nbytes // chunk)
        srcs, dsts = [], []
        for _ in range(count):
            if cursor + chunk > self.size:
                cursor = 0
            srcs.append(src_base + cursor)
            dsts.append(dst_base + cursor)
            cursor += chunk
        return srcs, dsts, [chunk] * count, cursor

    def run_case(self, scheme: str, level: int, start: float, stop: float, window: tuple[float, float]) -> dict:
        torch = self.torch
        target = level / 100 * GPU_SHARE_PEAK_BYTES
        per_period = int(target * PACE_SECONDS)
        rdma_chunk = self.args.rdma_request_bytes
        copy_chunk = self.args.memcpy_chunk_bytes
        cursor = 0
        done_bytes = 0
        pending: list[tuple[object, int]] = []
        wait_until(int(start * 1e9))
        tick = time.time()

        def account(nbytes: int) -> None:
            nonlocal done_bytes
            now = time.time()
            if window[0] <= now <= window[1]:
                done_bytes += nbytes

        while time.time() < stop:
            if scheme in ("p2d", "d2p"):
                if scheme == "p2d":
                    src, dst = self.dev.data_ptr(), self.peer["host"]
                else:
                    src, dst = self.host.data_ptr(), self.peer["dev"]
                srcs, dsts, lens, cursor = self.requests(per_period, rdma_chunk, src, dst, cursor)
                if self.engine.batch_transfer_sync_write(self.peer["session"], srcs, dsts, lens) < 0:
                    raise RuntimeError(f"{scheme} transfer failed")
                account(sum(lens))
            else:
                with torch.cuda.stream(self.stream):
                    if scheme == "hicache_h2d":
                        count = max(1, per_period // copy_chunk)
                        for _ in range(count):
                            if cursor + copy_chunk > self.size:
                                cursor = 0
                            self.dev[cursor:cursor + copy_chunk].copy_(
                                self.host[cursor:cursor + copy_chunk], non_blocking=True
                            )
                            cursor += copy_chunk
                        nbytes = count * copy_chunk
                    else:
                        srcs, dsts, lens, cursor = self.requests(
                            per_period, rdma_chunk, self.host.data_ptr(), self.dev.data_ptr(), cursor
                        )
                        self.engine.batch_transfer_write_on_cuda(
                            self.session, srcs, dsts, lens, self.stream.cuda_stream
                        )
                        nbytes = sum(lens)
                    event = torch.cuda.Event()
                    event.record(self.stream)
                pending.append((event, nbytes))
                # Keep at most two pacing periods in flight.
                while len(pending) > 1:
                    old, old_bytes = pending.pop(0)
                    old.synchronize()
                    account(old_bytes)
            tick += PACE_SECONDS
            delay = tick - time.time()
            if delay > 0:
                time.sleep(delay)
            else:
                tick = time.time()
        for event, nbytes in pending:
            event.synchronize()
            account(nbytes)
        seconds = window[1] - window[0]
        return {
            "target_gbytes_s": target / 1e9,
            "achieved_gbytes_s": done_bytes / seconds / 1e9,
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
        if case["level"] == 0 or not scheme_active(case["scheme"], node):
            continue
        record = gen.run_case(case["scheme"], case["level"], start, stop, window)
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
    parser.add_argument("--schemes", default="p2d,d2p,hicache_h2d,nic_h2d")
    parser.add_argument("--levels", default="0,10,20,30,40,50,60,70,80,90,100")
    parser.add_argument("--case-seconds", type=float, default=20.0)
    parser.add_argument("--warm-seconds", type=float, default=4.0)
    parser.add_argument("--measure-seconds", type=float, default=5.0)
    parser.add_argument("--probe-warmup-seconds", type=float, default=1.5,
                        help="untimed probe iterations just before each measurement window")
    parser.add_argument("--prefill-tokens", type=int, default=1004)
    parser.add_argument("--decode-batch", type=int, default=6)
    parser.add_argument("--decode-graph", type=int, default=1)
    parser.add_argument("--rdma-request-bytes", type=int, default=32 << 10)
    parser.add_argument("--memcpy-chunk-bytes", type=int, default=1 << 20)
    parser.add_argument("--buffer-bytes", type=int, default=512 << 20)
    parser.add_argument("--probe-port", type=int, default=29651)
    parser.add_argument("--barrier-port", type=int, default=29661)
    args = parser.parse_args()
    args.schemes = [s for s in args.schemes.split(",") if s]
    args.levels = [int(v) for v in args.levels.split(",") if v]
    node = int(os.environ.get("SLURM_NODEID", "0"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args)

    ctx = mp.get_context("spawn")
    # barriers[0]: every child initialized; barriers[1]: release instant published.
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

    counters = []
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
                "rx_requests": v.get("hw.rx_write_requests", 0) + v.get("hw.rx_read_requests", 0),
            }
            for hca, v in delta.items()
        }
        counters.append({"kind": "hca", "case_index": index, **case, "node": node, "hca": per_hca})
        print(f"node {node} case {index + 1}/{len(cases)} {case} hca={json.dumps(per_hca)}", flush=True)

    # Drain while children exit: a child blocks at exit until its queue items are read.
    records = counters
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
