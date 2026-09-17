"""Single path-aware component registry for TTFT V3 plots."""

from __future__ import annotations

from typing import Any

from plot_style import COLORS


PREFILL_PATH_COMPONENTS = (
    ("prefill_before_lustre_read_seconds", "Prefill before Lustre Read", COLORS["purple"]),
    ("prefill_lustre_read_seconds", "Prefill KV Read (Lustre)", COLORS["orange"]),
    ("prefill_scatter_seconds", "Prefill Scatter", COLORS["brown"]),
    ("prefill_host_to_device_seconds", "Prefill Host->GPU", COLORS["blue"]),
    ("prefill_computation_seconds", "Prefill Computation", COLORS["green"]),
    ("p_to_d_transfer_seconds", "P->D transfer", COLORS["red"]),
    ("decode_host_to_device_seconds", "Decode Host->GPU", COLORS["cyan"]),
    ("decode_dispatch_seconds", "Decode dispatch", COLORS["yellow"]),
    ("misc_seconds", "Misc", COLORS["gray"]),
)

DECODE_PATH_COMPONENTS = (
    ("decode_preallocation_seconds", "Decode pre-allocation", COLORS["yellow"]),
    ("decode_before_lustre_read_seconds", "Decode before Lustre Read", COLORS["purple"]),
    ("decode_lustre_read_seconds", "Decode KV Read (Lustre)", COLORS["orange"]),
    ("decode_scatter_seconds", "Decode Scatter", COLORS["brown"]),
    ("d_to_p_transfer_seconds", "D->P transfer", COLORS["pink"]),
    ("prefill_computation_seconds", "Prefill Computation", COLORS["green"]),
    ("p_to_d_transfer_seconds", "P->D transfer", COLORS["red"]),
    ("decode_host_to_device_seconds", "Decode Host->GPU", COLORS["cyan"]),
    ("decode_dispatch_seconds", "Decode dispatch", COLORS["blue"]),
    ("misc_seconds", "Misc", COLORS["gray"]),
)


def path_from_row(row: dict[str, Any]) -> str:
    return normalize_path(row.get("l3_read_path", row.get("path", "prefill")))


def normalize_path(value: Any) -> str:
    return "decode" if "decode" in str(value).lower() else "prefill"


def components_for_path(path: str):
    return (
        DECODE_PATH_COMPONENTS
        if normalize_path(path) == "decode"
        else PREFILL_PATH_COMPONENTS
    )


def normalize_ttft_row(row: dict[str, Any]) -> dict[str, Any]:
    """Fill V3 names from legacy fullchain fields without changing old data."""
    row.setdefault("prefill_before_lustre_read_seconds", 0.0)
    row.setdefault(
        "prefill_lustre_read_seconds",
        row.get("prefill_l3_kv_read_seconds", row.get("l3_kv_read_seconds", 0.0)),
    )
    row.setdefault("prefill_scatter_seconds", 0.0)
    row.setdefault("prefill_host_to_device_seconds", 0.0)
    row.setdefault(
        "prefill_computation_seconds", row.get("prefill_logits_sampling_seconds", 0.0)
    )
    row.setdefault("p_to_d_transfer_seconds", row.get("pd_transfer_seconds", 0.0))
    row.setdefault(
        "decode_preallocation_seconds", row.get("decode_prealloc_seconds", 0.0)
    )
    row.setdefault("decode_before_lustre_read_seconds", 0.0)
    row.setdefault(
        "decode_lustre_read_seconds", row.get("decode_l3_kv_read_seconds", 0.0)
    )
    row.setdefault("decode_scatter_seconds", 0.0)
    row.setdefault(
        "d_to_p_transfer_seconds",
        row.get("decode_to_prefill_kv_transfer_seconds", 0.0),
    )
    row.setdefault("decode_host_to_device_seconds", 0.0)
    row.setdefault("decode_dispatch_seconds", 0.0)
    row.setdefault("misc_seconds", 0.0)
    return row
