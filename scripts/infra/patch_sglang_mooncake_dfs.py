#!/usr/bin/env python3
"""Backport request-scoped Mooncake DFS replica placement to SGLang 26.01."""

from __future__ import annotations

from pathlib import Path


TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/sglang/srt/mem_cache/storage/"
    "mooncake_store/mooncake_store.py"
)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return source.replace(old, new, 1)


def main() -> None:
    source = TARGET.read_text()
    if "DUALPATH_MOONCAKE_DFS_BACKPORT" in source:
        print("Mooncake DFS backport already applied")
        return

    source = replace_once(
        source,
        """        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
""",
        """        try:
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        except ImportError as e:
""",
        "import ReplicateConfig",
    )
    source = replace_once(
        source,
        """            extra_config = (
                getattr(storage_config, "extra_config", None)
                if storage_config
                else None
            )
            # Load configuration with master_server_address prioritized from extra_config if available
""",
        """            extra_config = (
                getattr(storage_config, "extra_config", None)
                if storage_config
                else None
            )
            if isinstance(extra_config, str):
                extra_config = json.loads(extra_config)
            # Load configuration with master_server_address prioritized from extra_config if available
""",
        "parse serialized extra config",
    )
    source = replace_once(
        source,
        """            tp_scale_factor = 1 if storage_config is None else storage_config.tp_size

            per_tp_global_segment_size = (
""",
        """            # DUALPATH_MOONCAKE_DFS_BACKPORT: SGLang PR #37390 adapted
            # to the older MooncakeStore implementation in NVIDIA SGLang 26.01.
            self.replica_num = (
                extra_config.get("replica_num", 1) if extra_config else 1
            )
            self.dfs_replica_num = (
                extra_config.get("dfs_replica_num", 0) if extra_config else 0
            )
            if isinstance(self.replica_num, bool) or not isinstance(
                self.replica_num, int
            ):
                raise ValueError("Mooncake replica_num must be an integer")
            if isinstance(self.dfs_replica_num, bool) or not isinstance(
                self.dfs_replica_num, int
            ):
                raise ValueError("Mooncake dfs_replica_num must be an integer")
            if self.replica_num < 1:
                raise ValueError("Mooncake replica_num must be at least 1")
            if self.dfs_replica_num not in (0, 1):
                raise ValueError("Mooncake dfs_replica_num must be 0 or 1")

            self.enable_ssd_offload = bool(
                extra_config.get("enable_ssd_offload", False)
                if extra_config
                else os.getenv("MOONCAKE_OFFLOAD_ENABLED", "").lower()
                in ("1", "true", "yes", "on")
            )
            self.ssd_offload_path = (
                extra_config.get("ssd_offload_path", "")
                if extra_config
                else os.getenv("MOONCAKE_OFFLOAD_FILE_STORAGE_PATH", "")
            )
            if self.dfs_replica_num:
                if self.config.standalone_storage:
                    raise ValueError(
                        "This SGLang 26.01 DFS backport supports embedded "
                        "Mooncake clients only"
                    )
                if not self.enable_ssd_offload:
                    raise ValueError(
                        "Mooncake DFS replicas require enable_ssd_offload=true"
                    )
                if os.getenv(
                    "MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR"
                ) != "distributed_storage_backend":
                    raise ValueError(
                        "Mooncake DFS replicas require "
                        "MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR="
                        "distributed_storage_backend"
                    )
                if not isinstance(self.ssd_offload_path, str) or not os.path.isabs(
                    self.ssd_offload_path
                ):
                    raise ValueError(
                        "Mooncake DFS replicas require an absolute "
                        "ssd_offload_path"
                    )
                probe = ReplicateConfig()
                if not hasattr(probe, "dfs_replica_num"):
                    raise RuntimeError(
                        "Installed Mooncake does not support "
                        "ReplicateConfig.dfs_replica_num"
                    )
                logger.info(
                    "Using Mooncake DFS replication: replica_num=%d, "
                    "dfs_replica_num=%d",
                    self.replica_num,
                    self.dfs_replica_num,
                )

            tp_scale_factor = 1 if storage_config is None else storage_config.tp_size

            per_tp_global_segment_size = (
""",
        "configure DFS replicas",
    )
    source = replace_once(
        source,
        """                    device_name,
                    self.config.master_server_address,
                )
""",
        """                    device_name,
                    self.config.master_server_address,
                    enable_ssd_offload=self.enable_ssd_offload,
                    ssd_offload_path=self.ssd_offload_path,
                )
""",
        "initialize distributed storage backend",
    )
    source = replace_once(
        source,
        """    def _put_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]
    ) -> List[int]:
        return self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)
""",
        """    def _put_batch_zero_copy_impl(
        self, key_strs: List[str], buffer_ptrs: List[int], buffer_sizes: List[int]
    ) -> List[int]:
        if self.replica_num == 1 and self.dfs_replica_num == 0:
            return self.store.batch_put_from(key_strs, buffer_ptrs, buffer_sizes)
        from mooncake.store import ReplicateConfig

        config = ReplicateConfig()
        config.replica_num = self.replica_num
        config.dfs_replica_num = self.dfs_replica_num
        return self.store.batch_put_from(
            key_strs, buffer_ptrs, buffer_sizes, config
        )
""",
        "forward request-scoped replica config",
    )

    compile(source, str(TARGET), "exec")
    TARGET.write_text(source)
    print(f"Patched {TARGET}")


if __name__ == "__main__":
    main()
