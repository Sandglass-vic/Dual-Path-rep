#!/usr/bin/env python3
"""Write or read deterministic objects through Mooncake descriptor-based DFS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from mooncake.store import MooncakeDistributedStore, ReplicateConfig


def payload(index: int, size: int) -> bytes:
    block = hashlib.sha256(f"dualpath-mooncake-dfs-{index}".encode()).digest()
    return (block * ((size + len(block) - 1) // len(block)))[:size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("write", "read"), required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--metadata-server", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--object-count", type=int, default=16)
    parser.add_argument("--object-bytes", type=int, default=4 * 1024 * 1024)
    args = parser.parse_args()

    store = MooncakeDistributedStore()
    code = store.setup(
        local_hostname=args.host,
        metadata_server=args.metadata_server,
        global_segment_size=256 * 1024 * 1024,
        local_buffer_size=64 * 1024 * 1024,
        protocol="tcp",
        rdma_devices="",
        master_server_addr=args.master,
        enable_ssd_offload=True,
        ssd_offload_path=os.environ["MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"],
    )
    if code:
        raise RuntimeError(f"Mooncake setup failed: {code}")

    prefix = "dualpath-mooncake-dfs-smoke"
    if args.role == "write":
        config = ReplicateConfig()
        config.replica_num = 1
        config.dfs_replica_num = 1
        records = []
        for index in range(args.object_count):
            value = payload(index, args.object_bytes + index * 4096)
            key = f"{prefix}-{index:04d}"
            result = store.put(key, value, config)
            if result != 0:
                raise RuntimeError(f"put({key}) failed: {result}")
            records.append(
                {
                    "key": key,
                    "bytes": len(value),
                    "sha256": hashlib.sha256(value).hexdigest(),
                }
            )
        args.manifest.write_text(json.dumps(records, indent=2) + "\n")
        print(f"MOONCAKE_DFS_WRITE_OK objects={len(records)}")
    else:
        records = json.loads(args.manifest.read_text())
        for record in records:
            value = store.get(record["key"])
            if not isinstance(value, bytes):
                raise RuntimeError(
                    f"get({record['key']}) returned {type(value).__name__}"
                )
            if len(value) != record["bytes"]:
                raise RuntimeError(
                    f"short read for {record['key']}: "
                    f"{len(value)} != {record['bytes']}"
                )
            digest = hashlib.sha256(value).hexdigest()
            if digest != record["sha256"]:
                raise RuntimeError(f"checksum mismatch for {record['key']}")
        print(f"MOONCAKE_DFS_CROSS_NODE_READ_OK objects={len(records)}")


if __name__ == "__main__":
    main()
