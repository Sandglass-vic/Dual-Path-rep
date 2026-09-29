#!/usr/bin/env python3
"""Add the DUAL_PATH_CUDAMEMCPY=hicache|nic switch to SGLang's HiCache host pool.

Installs scripts/infra/dualpath_nic_copy.py as sglang/srt/mem_cache/ and makes
MHATokenToKVPoolHost register its buffers with, and route its direct-backend
H2D/D2H through, the NIC copy engine when DUAL_PATH_CUDAMEMCPY=nic. Idempotent.
"""

from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path

MARKER = "_dualpath_nic_copy"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return source.replace(old, new, 1)


def patch_memory_pool_host(path: Path) -> None:
    source = path.read_text()
    if MARKER in source:
        return
    source = replace_once(
        source,
        "logger = logging.getLogger(__name__)\n",
        "logger = logging.getLogger(__name__)\n\n"
        "from sglang.srt.mem_cache import dualpath_nic_copy as _dualpath_nic_copy\n",
        "nic copy import",
    )
    source = replace_once(
        source,
        """        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
""",
        """        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        if _dualpath_nic_copy.ENABLED:
            _dualpath_nic_copy.get_engine().register_pool(self)

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
""",
        "MHA host pool registration",
    )
    source = replace_once(
        source,
        """        layer_id,
        io_backend,
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer(""",
        """        layer_id,
        io_backend,
    ):
        if _dualpath_nic_copy.ENABLED:
            if io_backend != "direct":
                raise ValueError(
                    "DUAL_PATH_CUDAMEMCPY=nic requires --hicache-io-backend direct"
                )
            _dualpath_nic_copy.get_engine().h2d_layer(
                self, device_pool, host_indices, device_indices, layer_id
            )
            return
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer(""",
        "MHA H2D",
    )
    source = replace_once(
        source,
        """    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer(""",
        """    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if _dualpath_nic_copy.ENABLED:
            if io_backend != "direct":
                raise ValueError(
                    "DUAL_PATH_CUDAMEMCPY=nic requires --hicache-io-backend direct"
                )
            _dualpath_nic_copy.get_engine().d2h_all_layers(
                self, device_pool, host_indices, device_indices
            )
            return
        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer(""",
        "MHA D2H",
    )
    path.write_text(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=Path("/usr/local/lib/python3.12/dist-packages"),
    )
    args = parser.parse_args()
    mem_cache = args.site_packages / "sglang/srt/mem_cache"
    module = mem_cache / "dualpath_nic_copy.py"
    shutil.copyfile(Path(__file__).with_name("dualpath_nic_copy.py"), module)
    host_pool = mem_cache / "memory_pool_host.py"
    patch_memory_pool_host(host_pool)
    for path in (module, host_pool):
        py_compile.compile(str(path), doraise=True)
    print(f"patched {host_pool} and installed {module}")


if __name__ == "__main__":
    main()
