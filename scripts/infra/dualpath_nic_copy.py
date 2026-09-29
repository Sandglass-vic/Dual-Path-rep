"""HiCache H2D/D2H through the GPU's paired RDMA NIC.

``DUAL_PATH_CUDAMEMCPY=nic`` routes ``MHATokenToKVPoolHost``'s ``direct``
backend copies (``load_to_device_per_layer`` and
``backup_from_device_all_layer``) through a Mooncake Transfer Engine loopback
on the HCA paired with the GPU, instead of ``cudaMemcpyAsync``. The NIC reads
one side over PCIe and writes the other with GPUDirect RDMA, so the copy
engine never touches the data. ``hicache`` (default) leaves SGLang unchanged.

Copies are synchronous in the calling thread:

1. synchronize the caller's current stream, so everything ordered before the
   copy on the GPU (including the stream's waits on the compute stream) has
   finished before the NIC reads or overwrites the buffers;
2. post the RDMA writes and wait for their completion;
3. read 4 bytes back from the destination, which forces the NIC's posted
   PCIe writes to land before the call returns.

Anything the caller records on the stream afterwards (``LayerDoneCounter``
events, ``finish_event``) therefore observes the copied data.

An asynchronous variant that lets the stream wait on a host flag set by a
Python thread deadlocks: a CUDA call in another thread that blocks on the GPU
while holding the GIL (e.g. a caching-allocator cudaMalloc) keeps the flag
writer from running. Asynchrony needs a native completion path.

Environment:
  DUAL_PATH_CUDAMEMCPY   hicache | nic
  DUAL_PATH_NIC_DEVICE   HCA to use; otherwise the GPU's entry in the JSON map
                         in IB_DEVICE / MOONCAKE_DEVICE / DUALPATH_PZA100_IB_DEVICE_MAP
  DUAL_PATH_NIC_HOST     loopback address for the Mooncake handshake (127.0.0.1)
  DUAL_PATH_NIC_FLUSH    1 (default): flush read after every transfer
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODE = os.environ.get("DUAL_PATH_CUDAMEMCPY", "hicache")
if MODE not in ("hicache", "nic"):
    raise ValueError(f"DUAL_PATH_CUDAMEMCPY must be hicache or nic, got {MODE!r}")
ENABLED = MODE == "nic"

SUPPORTED_LAYOUTS = ("layer_first", "page_first_direct")
FLUSH_BYTES = 4

_engine = None
_engine_lock = threading.Lock()


def get_engine() -> "NicCopyEngine":
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = NicCopyEngine()
        return _engine


def hca_for_gpu(gpu: int) -> str:
    explicit = os.environ.get("DUAL_PATH_NIC_DEVICE", "").strip()
    if explicit:
        return explicit
    for name in ("IB_DEVICE", "MOONCAKE_DEVICE", "DUALPATH_PZA100_IB_DEVICE_MAP"):
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw.split(",")[0].strip()
        if isinstance(parsed, dict) and str(gpu) in parsed:
            return parsed[str(gpu)].split(",")[0].strip()
    raise RuntimeError(
        "DUAL_PATH_CUDAMEMCPY=nic needs DUAL_PATH_NIC_DEVICE or a GPU-to-HCA "
        f"JSON map in IB_DEVICE for GPU {gpu}"
    )


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


class _Plan:
    """Runs of tokens that are contiguous in both the host and device pool."""

    def __init__(self, host_pool, host_indices, device_indices):
        host = host_indices.cpu().numpy().astype(np.int64, copy=False)
        dev = device_indices.cpu().numpy().astype(np.int64, copy=False)
        if host.shape != dev.shape:
            raise ValueError("host and device index counts differ")
        self.key = self.key_of(host_indices, device_indices)
        stride = host_pool.token_stride_size
        if host_pool.layout == "layer_first":
            host_off = host * stride
        else:
            ps = host_pool.page_size
            host_off = (host // ps * host_pool.layer_num * ps + host % ps) * stride
        if len(host) == 0:
            self.host_off = self.dev_tok = self.nbytes = np.zeros(0, dtype=np.int64)
            return
        breaks = np.flatnonzero((np.diff(host_off) != stride) | (np.diff(dev) != 1)) + 1
        starts = np.concatenate(([0], breaks))
        lengths = np.diff(np.concatenate((starts, [len(host)])))
        self.host_off = host_off[starts]
        self.dev_tok = dev[starts]
        self.nbytes = lengths * stride

    @staticmethod
    def key_of(host_indices, device_indices):
        empty = host_indices.numel() == 0
        return (
            host_indices.data_ptr(),
            device_indices.data_ptr(),
            host_indices.numel(),
            -1 if empty else int(host_indices[0]),
            -1 if empty else int(host_indices[-1]),
            -1 if empty else int(device_indices[0]),
        )


class NicCopyEngine:
    def __init__(self):
        from mooncake.engine import TransferEngine

        self.gpu = torch.cuda.current_device()
        self.hca = hca_for_gpu(self.gpu)
        host = os.environ.get("DUAL_PATH_NIC_HOST", "127.0.0.1")
        self.engine = TransferEngine()
        if self.engine.initialize(host, "P2PHANDSHAKE", "rdma", self.hca) != 0:
            raise RuntimeError(f"Mooncake TransferEngine initialize failed on {self.hca}")
        self.session = f"{host}:{self.engine.get_rpc_port()}"
        self.flush_enabled = os.environ.get("DUAL_PATH_NIC_FLUSH", "1") == "1"
        self.registered: set[tuple[int, int]] = set()
        self.pools: dict[int, tuple] = {}
        self.plans: dict[int, _Plan] = {}
        self.lock = threading.Lock()
        self.scratch = torch.zeros(4096, dtype=torch.uint8, pin_memory=True)
        self._register([(self.scratch.data_ptr(), _nbytes(self.scratch))])
        self._self_test()
        logger.info(
            "DUAL_PATH_CUDAMEMCPY=nic gpu=%d hca=%s session=%s flush=%s",
            self.gpu, self.hca, self.session, self.flush_enabled,
        )

    def _register(self, regions) -> None:
        new = [region for region in regions if region not in self.registered]
        if not new:
            return
        started = time.perf_counter()
        ret = self.engine.batch_register_memory([p for p, _ in new], [n for _, n in new])
        if ret != 0:
            raise RuntimeError("Mooncake batch_register_memory failed")
        self.registered.update(new)
        logger.info(
            "DUAL_PATH_NIC_REGISTER regions=%d bytes=%d seconds=%.3f",
            len(new), sum(n for _, n in new), time.perf_counter() - started,
        )

    def register_pool(self, host_pool) -> None:
        """Register a host pool and its device pool; idempotent per buffer set."""
        if host_pool.layout not in SUPPORTED_LAYOUTS:
            raise ValueError(
                f"DUAL_PATH_CUDAMEMCPY=nic supports host layouts {SUPPORTED_LAYOUTS}, "
                f"got {host_pool.layout}"
            )
        device_pool = host_pool.device_pool
        key = (
            host_pool.kv_buffer.data_ptr(),
            tuple(t.data_ptr() for t in device_pool.k_buffer),
            tuple(t.data_ptr() for t in device_pool.v_buffer),
        )
        if self.pools.get(id(host_pool)) == key:
            return
        regions = [(host_pool.kv_buffer.data_ptr(), _nbytes(host_pool.kv_buffer))]
        regions += [(t.data_ptr(), _nbytes(t)) for t in device_pool.k_buffer]
        regions += [(t.data_ptr(), _nbytes(t)) for t in device_pool.v_buffer]
        self._register(regions)
        self.pools[id(host_pool)] = key

    def _self_test(self) -> None:
        """Round-trip one buffer host -> GPU -> host."""
        size = 1 << 20
        src = torch.randint(0, 255, (size,), dtype=torch.uint8).pin_memory()
        back = torch.zeros(size, dtype=torch.uint8, pin_memory=True)
        gpu = torch.zeros(size, dtype=torch.uint8, device=f"cuda:{self.gpu}")
        regions = [(src.data_ptr(), size), (back.data_ptr(), size), (gpu.data_ptr(), size)]
        self._register(regions)
        stream = torch.cuda.current_stream()
        self._transfer([(src.data_ptr(), gpu.data_ptr(), size)], gpu.data_ptr(), stream)
        self._transfer([(gpu.data_ptr(), back.data_ptr(), size)], back.data_ptr(), stream)
        if not torch.equal(src, back):
            raise RuntimeError("DUAL_PATH_CUDAMEMCPY=nic self-test: data mismatch")
        self.engine.batch_unregister_memory([p for p, _ in regions])
        self.registered -= set(regions)

    def _transfer(self, requests, flush_addr: int, stream) -> None:
        stream.synchronize()
        src = [r[0] for r in requests]
        dst = [r[1] for r in requests]
        lengths = [r[2] for r in requests]
        with self.lock:
            if self.engine.batch_transfer_sync_write(self.session, src, dst, lengths) < 0:
                raise RuntimeError("DUAL_PATH_CUDAMEMCPY=nic: Mooncake batch write failed")
            if self.flush_enabled and self.engine.batch_transfer_sync_read(
                self.session, [self.scratch.data_ptr()], [flush_addr], [FLUSH_BYTES]
            ) < 0:
                raise RuntimeError("DUAL_PATH_CUDAMEMCPY=nic: flush read failed")

    def _plan(self, host_pool, host_indices, device_indices) -> _Plan:
        plan = self.plans.get(id(host_pool))
        if plan is None or plan.key != _Plan.key_of(host_indices, device_indices):
            plan = _Plan(host_pool, host_indices, device_indices)
            self.plans[id(host_pool)] = plan
        return plan

    def _layer_requests(self, host_pool, device_pool, plan: _Plan, layer: int, h2d: bool):
        stride = host_pool.token_stride_size
        if host_pool.layout == "layer_first":
            layer_off = layer * host_pool.size * stride
        else:
            layer_off = layer * host_pool.page_size * stride
        requests = []
        for host_base, dev_layer in (
            (host_pool.k_buffer.data_ptr(), device_pool.k_buffer[layer]),
            (host_pool.v_buffer.data_ptr(), device_pool.v_buffer[layer]),
        ):
            host_addr = host_base + layer_off + plan.host_off
            dev_addr = dev_layer.data_ptr() + plan.dev_tok * stride
            pairs = (host_addr, dev_addr) if h2d else (dev_addr, host_addr)
            requests += list(zip(pairs[0].tolist(), pairs[1].tolist(), plan.nbytes.tolist()))
        return requests

    def h2d_layer(self, host_pool, device_pool, host_indices, device_indices, layer_id) -> None:
        self.register_pool(host_pool)
        plan = self._plan(host_pool, host_indices, device_indices)
        if len(plan.nbytes) == 0:
            return
        requests = self._layer_requests(host_pool, device_pool, plan, layer_id, h2d=True)
        self._transfer(requests, requests[0][1], torch.cuda.current_stream())

    def d2h_all_layers(self, host_pool, device_pool, host_indices, device_indices) -> None:
        self.register_pool(host_pool)
        plan = _Plan(host_pool, host_indices, device_indices)
        if len(plan.nbytes) == 0:
            return
        requests = []
        for layer in range(host_pool.layer_num):
            requests += self._layer_requests(host_pool, device_pool, plan, layer, h2d=False)
        self._transfer(requests, requests[0][1], torch.cuda.current_stream())
