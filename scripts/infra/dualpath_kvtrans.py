"""DUAL_PATH_KVTRANS=layer|full runtime support.

Installed as ``sglang/srt/utils/dualpath_kvtrans.py`` by
``scripts/infra/patch_sglang_kvtrans.py``.

* unset  : Prefill path layer-wise (upstream HiCache), Decode path full block.
* layer  : both paths layer-wise. Decode RDMA-writes the prefix into Prefill
           HBM one layer at a time and bumps a per-request 32-bit flag after
           each layer; the Prefill forward waits on that flag on-GPU with
           ``cuStreamWaitValue32`` right before attention reads layer i.
* full   : both paths full block. Prefill loads the whole host-hit prefix
           into HBM and only admits the request once every TP rank finished.

The flag slots live in a memfd so a watchdog *process* can release a stuck
GPU wait even when the scheduler thread holds the GIL inside a blocking
kernel launch.
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import os
import subprocess
import sys
import threading
import time
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODE = os.environ.get("DUAL_PATH_KVTRANS", "").strip().lower()
if MODE not in ("", "layer", "full"):
    raise ValueError(
        f"DUAL_PATH_KVTRANS must be unset, 'layer' or 'full', got {MODE!r}"
    )

PREFILL_MODE = "full" if MODE == "full" else "layer"
DECODE_MODE = "layer" if MODE == "layer" else "full"
MODE_TAG = (MODE or "default").encode("ascii")

# A healthy Decode streams all layers well under a second; this bound only
# decides how long a dead peer may hold the Prefill GPU stream.
LAYER_STREAM_TIMEOUT_S = 30.0
# A slot is reused only after Decode acknowledged the room, or after this
# long, so a late flag write can never land in another request's slot.
SLOT_QUARANTINE_S = 120.0
# Even after the ack, keep the slot out long enough for any wait op already
# queued on a lagging GPU stream to observe the released value.
SLOT_MIN_QUARANTINE_S = 2.0
NUM_SLOTS = 512

_CU_STREAM_WAIT_VALUE_GEQ = 0x0
_CU_STREAM_WAIT_VALUE_FLUSH = 1 << 30
_CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98
_CUDA_HOST_REGISTER_PORTABLE_MAPPED = 0x01 | 0x02

remote_wait_ready = False


class _CudaDriver:
    def __init__(self):
        lib = ctypes.CDLL("libcuda.so.1")
        self.wait_value32 = lib.cuStreamWaitValue32_v2
        self.wait_value32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint,
        ]
        self.wait_value32.restype = ctypes.c_int
        self.host_get_device_pointer = lib.cuMemHostGetDevicePointer_v2
        self.host_get_device_pointer.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self.host_get_device_pointer.restype = ctypes.c_int
        get_attr = lib.cuDeviceGetAttribute
        get_attr.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        get_attr.restype = ctypes.c_int
        import torch

        value = ctypes.c_int(0)
        get_attr(
            ctypes.byref(value),
            _CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES,
            torch.cuda.current_device(),
        )
        self.wait_flags = _CU_STREAM_WAIT_VALUE_GEQ | (
            _CU_STREAM_WAIT_VALUE_FLUSH if value.value else 0
        )


_driver: Optional[_CudaDriver] = None


def _get_driver() -> _CudaDriver:
    global _driver
    if _driver is None:
        _driver = _CudaDriver()
    return _driver


def mark_remote_wait_ready() -> None:
    """Called once the Prefill HiCache layer counter exists in this process."""
    global remote_wait_ready
    remote_wait_ready = True


def wait_remote_layer(flag_device_ptrs: List[int], layer: int) -> None:
    """Enqueue a GPU-side wait for ``flag >= layer + 1`` on the current stream."""
    import torch

    driver = _get_driver()
    stream = torch.cuda.current_stream().cuda_stream
    for ptr in flag_device_ptrs:
        rc = driver.wait_value32(stream, ptr, layer + 1, driver.wait_flags)
        if rc != 0:
            raise RuntimeError(f"cuStreamWaitValue32 failed with CUresult {rc}")


_WATCHDOG_SOURCE = r"""
import mmap, os, sys, time
fd, n, parent = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
mm = mmap.mmap(fd, 0)
flags = memoryview(mm)[0:4 * n].cast("I")
target = memoryview(mm)[4 * n:8 * n].cast("I")
timedout = memoryview(mm)[8 * n:12 * n].cast("I")
deadline = memoryview(mm)[16 * n:24 * n].cast("d")
while os.getppid() == parent:
    now = time.monotonic()
    for i in range(n):
        d = deadline[i]
        if d and now > d:
            if flags[i] < target[i]:
                flags[i] = target[i]
                timedout[i] = 1
            deadline[i] = 0.0
    time.sleep(0.005)
"""


class RemoteLayerFlags:
    """Per-request 32-bit layer counters that Decode RDMA-writes into.

    Layout of the shared memfd (``n`` slots):
    flags[u32 n] | target[u32 n] | timedout[u32 n] | pad[u32 n] | deadline[f64 n]
    """

    def __init__(self, engine, num_slots: int = NUM_SLOTS):
        import torch

        self.num_slots = num_slots
        size = 24 * num_slots
        size = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
        self._fd = os.memfd_create("dualpath_kvtrans_flags")
        os.ftruncate(self._fd, size)
        self._mm = mmap.mmap(self._fd, size)
        buf = self._mm
        n = num_slots
        self.flags = np.frombuffer(buf, dtype=np.uint32, count=n, offset=0)
        self.target = np.frombuffer(buf, dtype=np.uint32, count=n, offset=4 * n)
        self.timedout = np.frombuffer(
            buf, dtype=np.uint32, count=n, offset=8 * n
        )
        self.deadline = np.frombuffer(
            buf, dtype=np.float64, count=n, offset=16 * n
        )
        self.host_ptr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        rc = torch.cuda.cudart().cudaHostRegister(
            self.host_ptr, size, _CUDA_HOST_REGISTER_PORTABLE_MAPPED
        )
        if int(rc) != 0:
            raise RuntimeError(f"cudaHostRegister(flags) failed: {rc}")
        device_ptr = ctypes.c_uint64(0)
        rc = _get_driver().host_get_device_pointer(
            ctypes.byref(device_ptr), self.host_ptr, 0
        )
        if rc != 0:
            raise RuntimeError(f"cuMemHostGetDevicePointer failed: {rc}")
        self.device_ptr = device_ptr.value
        if engine.batch_register([self.host_ptr], [4 * n]) != 0:
            raise RuntimeError("Mooncake register(flags) failed")

        self._lock = threading.Lock()
        self._free = list(range(num_slots - 1, -1, -1))
        # slot -> (room, earliest reuse time); reusable after ack or expiry.
        self._quarantine = {}
        self._acked_rooms = set()
        self._active_rooms = set()
        self._watchdog = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _WATCHDOG_SOURCE,
                str(self._fd),
                str(num_slots),
                str(os.getpid()),
            ],
            pass_fds=(self._fd,),
            close_fds=True,
        )

    def host_addr(self, slot: int) -> int:
        return self.host_ptr + 4 * slot

    def device_addr(self, slot: int) -> int:
        return self.device_ptr + 4 * slot

    def alloc(self, num_layers: int, room: int) -> Optional[int]:
        with self._lock:
            self._reclaim_locked()
            if not self._free:
                return None
            slot = self._free.pop()
            self._active_rooms.add(room)
        self.deadline[slot] = 0.0
        self.timedout[slot] = 0
        self.target[slot] = num_layers
        self.flags[slot] = 0
        return slot

    def arm(self, slot: int) -> None:
        """Bound the GPU wait once the request is admitted into a batch."""
        self.deadline[slot] = time.monotonic() + LAYER_STREAM_TIMEOUT_S

    def ready(self, slot: int) -> bool:
        return int(self.flags[slot]) >= 1

    def finished(self, slot: int) -> bool:
        return int(self.flags[slot]) >= int(self.target[slot])

    def timed_out(self, slot: int) -> bool:
        return bool(self.timedout[slot])

    def unblock(self, slot: int) -> None:
        self.flags[slot] = self.target[slot]

    def ack(self, room: int) -> None:
        """Decode reported the room done; it will not write that flag again."""
        with self._lock:
            if room in self._active_rooms:
                self._acked_rooms.add(room)

    def release(self, slot: int, room: int) -> None:
        self.unblock(slot)
        self.deadline[slot] = 0.0
        with self._lock:
            self._quarantine[slot] = (room, time.monotonic())
            self._reclaim_locked()

    def _reclaim_locked(self) -> None:
        if not self._quarantine:
            return
        now = time.monotonic()
        for slot, (room, released) in list(self._quarantine.items()):
            age = now - released
            if age > SLOT_QUARANTINE_S or (
                age > SLOT_MIN_QUARANTINE_S and room in self._acked_rooms
            ):
                self._acked_rooms.discard(room)
                self._active_rooms.discard(room)
                del self._quarantine[slot]
                self._free.append(slot)


class LayerSendConstants:
    """Registered Decode-side source buffers for the per-layer flag write
    and the 4-byte read-after-write flush."""

    def __init__(self, engine, num_layers: int):
        self.values = np.arange(1, num_layers + 1, dtype=np.uint32)
        self.scratch = np.zeros(16, dtype=np.uint32)
        self.values_ptr = self.values.ctypes.data
        self.scratch_ptr = self.scratch.ctypes.data
        if engine.batch_register(
            [self.values_ptr, self.scratch_ptr],
            [self.values.nbytes, self.scratch.nbytes],
        ) != 0:
            raise RuntimeError("Mooncake register(layer constants) failed")
        self.num_layers = num_layers

    def value_ptr(self, layer: int) -> int:
        return self.values_ptr + 4 * layer
