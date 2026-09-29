"""Report which NCCL transport a single-node TP4 all-reduce uses on g5/g6.

torchrun --nproc-per-node 4 scripts/smoke/nccl_tp4_path_probe.py
Run with NCCL_DEBUG=INFO and grep the log for 'via'.
"""

import os
import time

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    world = dist.get_world_size()

    for size in (8 << 10, 1 << 20, 64 << 20):
        x = torch.ones(size // 2, dtype=torch.bfloat16, device="cuda")
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        iters = 50
        start = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iters
        busbw = size * 2 * (world - 1) / world / elapsed / 1e9
        if rank == 0:
            print(f"allreduce {size:>10d} B  {elapsed * 1e6:9.1f} us  "
                  f"busbw {busbw:7.2f} GB/s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
