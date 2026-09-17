#!/usr/bin/env python3
"""Build an additive first-token critical-path breakdown for a small PD run."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


V3_PREFIX = "DUALPATH_TTFT_V3 "
V3_STAGE_NAMES = (
    "prefill_before_lustre_read",
    "prefill_lustre_read",
    "prefill_scatter",
    "prefill_host_to_device",
    "prefill_computation",
    "p_to_d_transfer",
    "decode_preallocation",
    "decode_before_lustre_read",
    "decode_lustre_read",
    "decode_scatter",
    "d_to_p_transfer",
    "decode_host_to_device",
    "decode_dispatch",
)


def collect_v3(paths: list[Path]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    records: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            marker = line.find(V3_PREFIX)
            if marker < 0:
                continue
            payload = json.loads(line[marker + len(V3_PREFIX) :])
            if int(payload.get("schema_version", 0)) != 3:
                continue
            records[str(payload["rid"])][str(payload["side"])].append(payload)
    return records


def _v3_stage(record: dict[str, Any], stage: str, kind: str = "seconds") -> float:
    return max(0.0, float(record.get(f"{stage}_{kind}", 0.0)))


def _v3_pick_pipeline(
    records: list[dict[str, Any]], side: str
) -> dict[str, Any]:
    stages = (
        f"{side}_before_lustre_read",
        f"{side}_lustre_read",
        f"{side}_scatter",
    )
    return max(
        records,
        key=lambda item: sum(_v3_stage(item, stage, "raw_seconds") for stage in stages),
        default={},
    )


def _v3_pick_pair(
    records: list[dict[str, Any]], first: str, second: str
) -> dict[str, Any]:
    return max(
        records,
        key=lambda item: _v3_stage(item, first) + _v3_stage(item, second),
        default={},
    )


def _v3_pick_max(records: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    return max(records, key=lambda item: _v3_stage(item, stage), default={})


def summarize_v3(
    rounds: list[dict[str, Any]],
    logs: list[Path],
    output_dir: Path,
) -> bool:
    records = collect_v3(logs)
    if not records:
        return False
    merged: list[dict[str, Any]] = []
    for round_item in rounds:
        rid = str(round_item["request_id"])
        sides = records.get(rid, {})
        prefill = sides.get("prefill", [])
        decode = sides.get("decode", [])
        if not prefill or not decode:
            raise RuntimeError(
                f"{rid} missing TTFT V3 sides: "
                f"prefill={len(prefill)} decode={len(decode)}"
            )
        path = str((prefill[0] if prefill else decode[0]).get("path", "prefill"))
        selected: dict[str, dict[str, Any]] = {}
        if path == "decode":
            read_record = _v3_pick_pipeline(decode, "decode")
            for stage in (
                "decode_before_lustre_read",
                "decode_lustre_read",
                "decode_scatter",
            ):
                selected[stage] = read_record
        else:
            read_record = _v3_pick_pipeline(prefill, "prefill")
            for stage in (
                "prefill_before_lustre_read",
                "prefill_lustre_read",
                "prefill_scatter",
            ):
                selected[stage] = read_record

        compute_record = _v3_pick_pair(
            prefill, "prefill_host_to_device", "prefill_computation"
        )
        selected["prefill_host_to_device"] = compute_record
        selected["prefill_computation"] = compute_record
        selected["p_to_d_transfer"] = _v3_pick_max(prefill, "p_to_d_transfer")
        selected["decode_preallocation"] = _v3_pick_max(
            decode, "decode_preallocation"
        )
        selected["d_to_p_transfer"] = _v3_pick_max(decode, "d_to_p_transfer")
        selected["decode_host_to_device"] = _v3_pick_max(
            decode, "decode_host_to_device"
        )
        dispatch_records = [item for item in decode if int(item["tp_rank"]) == 0]
        selected["decode_dispatch"] = (
            dispatch_records[0]
            if dispatch_records
            else _v3_pick_max(decode, "decode_dispatch")
        )

        if path == "decode":
            critical_order = (
                "decode_preallocation",
                "decode_before_lustre_read",
                "decode_lustre_read",
                "decode_scatter",
                "d_to_p_transfer",
                "prefill_computation",
                "p_to_d_transfer",
                "decode_host_to_device",
                "decode_dispatch",
            )
        else:
            critical_order = (
                "prefill_before_lustre_read",
                "prefill_lustre_read",
                "prefill_scatter",
                "prefill_host_to_device",
                "prefill_computation",
                "p_to_d_transfer",
                "decode_host_to_device",
                "decode_dispatch",
            )

        output: dict[str, Any] = {**round_item, "ttft_timing_schema_version": 3}
        for stage in V3_STAGE_NAMES:
            record = selected.get(stage, {})
            output[f"{stage}_raw_seconds"] = _v3_stage(
                record, stage, "raw_seconds"
            )
            output[f"{stage}_seconds"] = _v3_stage(record, stage)
            output[f"{stage}_overlap_removed_seconds"] = _v3_stage(
                record, stage, "overlap_removed_seconds"
            )
            output[f"{stage}_selected_tp_rank_diagnostic"] = int(
                record.get("tp_rank", -1)
            )
            output[f"{stage}_selected_engine_diagnostic"] = str(
                record.get("engine", "")
            )

        critical_sum = sum(float(output[f"{stage}_seconds"]) for stage in critical_order)
        ttft = float(round_item["ttft_seconds"])
        output["misc_seconds"] = max(0.0, ttft - critical_sum)
        output["critical_stage_sum_seconds"] = critical_sum
        output["accounting_error_seconds"] = ttft - (
            critical_sum + output["misc_seconds"]
        )
        output["accounting_overlap_excess_seconds"] = max(0.0, critical_sum - ttft)
        tolerance = max(0.002, 0.005 * ttft)
        output["accounting_tolerance_seconds"] = tolerance
        output["accounting_error_exceeds_tolerance"] = (
            abs(float(output["accounting_error_seconds"])) > tolerance
            or float(output["accounting_overlap_excess_seconds"]) > tolerance
        )
        output["ttft_component_order"] = list(critical_order) + ["misc"]
        output["prefill_read_tokens_diagnostic"] = int(
            selected.get("prefill_lustre_read", {}).get("read_tokens", 0)
        )
        output["decode_read_tokens_diagnostic"] = int(
            selected.get("decode_lustre_read", {}).get("read_tokens", 0)
        )
        output["prefill_read_pages_diagnostic"] = int(
            selected.get("prefill_lustre_read", {}).get("read_pages", 0)
        )
        output["decode_read_pages_diagnostic"] = int(
            selected.get("decode_lustre_read", {}).get("read_pages", 0)
        )
        output["decode_host_allocation_seconds_diagnostic"] = max(
            float(item.get("decode_host_allocation_seconds_diagnostic", 0.0))
            for item in decode
        )
        output["prefill_tp_rank_count_diagnostic"] = len(prefill)
        output["decode_tp_rank_count_diagnostic"] = len(decode)
        output["l3_read_path"] = path

        # Compatibility aliases retained for older plotting and CSV consumers.
        output["prefill_logits_sampling_seconds"] = output[
            "prefill_computation_seconds"
        ]
        output["pd_transfer_seconds"] = output["p_to_d_transfer_seconds"]
        output["decode_prealloc_seconds"] = output[
            "decode_preallocation_seconds"
        ]
        output["decode_to_prefill_kv_transfer_seconds"] = output[
            "d_to_p_transfer_seconds"
        ]
        output["bootstrap_and_scheduling_seconds"] = 0.0
        output["decode_l3_bootstrap_handoff_seconds"] = 0.0
        output["kv_read_overhead_seconds"] = 0.0
        output["sse_egress_seconds"] = 0.0
        output["server_runtime_residual_seconds"] = 0.0
        output["router_client_runtime_seconds"] = 0.0
        merged.append(output)

    stage_means = {
        f"{stage}_{kind}": statistics.fmean(
            float(item[f"{stage}_{kind}"]) for item in merged
        )
        for stage in V3_STAGE_NAMES
        for kind in ("raw_seconds", "seconds", "overlap_removed_seconds")
    }
    summary = {
        "schema_version": 3,
        "requests": len(merged),
        "ttft_mean_seconds": statistics.fmean(
            float(item["ttft_seconds"]) for item in merged
        ),
        "stage_mean_seconds": stage_means,
        "misc_mean_seconds": statistics.fmean(
            float(item["misc_seconds"]) for item in merged
        ),
        "accounting_error_max_abs_seconds": max(
            abs(float(item["accounting_error_seconds"])) for item in merged
        ),
        "accounting_overlap_excess_max_seconds": max(
            float(item["accounting_overlap_excess_seconds"]) for item in merged
        ),
        "accounting_error_rounds": sum(
            bool(item["accounting_error_exceeds_tolerance"]) for item in merged
        ),
        "timing_coverage": {
            "prefill_records": sum(len(value.get("prefill", [])) for value in records.values()),
            "decode_records": sum(len(value.get("decode", [])) for value in records.values()),
            "complete_requests": len(merged),
        },
        "validation_status": "not_run_waiting_for_user",
    }
    with (output_dir / "fullchain_rounds.jsonl").open("w") as stream:
        for item in merged:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
    (output_dir / "fullchain_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return True


REQ = re.compile(
    r"DUALPATH_REQ_TIMING rid=(?P<rid>\S+) type=(?P<type>\S+).*?"
    r"queue_start=(?P<queue_start>[0-9.]+) "
    r"queue_end=(?P<queue_end>[0-9.]+) "
    r"prefill_start=(?P<prefill_start>[0-9.]+) "
    r"prefill_end=(?P<prefill_end>[0-9.]+) "
    r"prefill_bootstrap_start=(?P<bootstrap_start>[0-9.]+)"
)
L3 = re.compile(
    r"DUALPATH_L3_TIMING rid=(?P<rid>\S+) "
    r"(?:prefill_engine=(?P<engine>\S+) )?"
    r"(?:node=(?P<node>\S+) )?tp_rank=(?P<rank>\d+) "
    r"read_start=(?P<start>[0-9.]+) read_end=(?P<end>[0-9.]+) "
    r"read_s=(?P<duration>[0-9.]+) tokens=(?P<tokens>\d+)"
)
PREFILL_KV_READ_BREAKDOWN = re.compile(
    r"DUALPATH_PREFILL_KV_READ_BREAKDOWN rid=(?P<rid>\S+) "
    r"prefill_engine=(?P<engine>\S+) "
    r"(?:node=(?P<node>\S+) )?tp_rank=(?P<rank>\d+) "
    r"lustre_read_s=(?P<lustre>[0-9.]+) "
    r"scatter_s=(?P<scatter>[0-9.]+) pages=(?P<pages>\d+)"
)
DECODE_L3 = re.compile(
    r"DUALPATH_DECODE_L3_READ rid=(?P<rid>\S+) "
    r"decode_engine=(?P<engine>\S+) "
    r"(?:node=(?P<node>\S+) )?tp_rank=(?P<rank>\d+) "
    r"read_start=(?P<start>[0-9.]+) read_end=(?P<end>[0-9.]+) "
    r"read_s=(?P<duration>[0-9.]+) "
    r"(?:lustre_read_s=(?P<lustre>[0-9.]+) "
    r"scatter_s=(?P<scatter>[0-9.]+) )?tokens=(?P<tokens>\d+)"
)
D2P_REVERSE = re.compile(
    r"DUALPATH_D2P_REVERSE rid=(?P<rid>\S+) "
    r"prefill_engine=(?P<engine>\S+) "
    r"(?:node=(?P<node>\S+) )?tp_rank=(?P<rank>\d+) "
    r"reverse_start=(?P<start>[0-9.]+) reverse_end=(?P<end>[0-9.]+) "
    r"reverse_s=(?P<duration>[0-9.]+) tokens=(?P<tokens>\d+)"
)
PREFILL_PREP = re.compile(
    r"DUALPATH_PREFILL_PREP_BREAKDOWN rid=(?P<rid>\S+) "
    r"tp_rank=(?P<rank>\d+) .*?"
    r"batch_init_end=(?P<start>[0-9.]+) "
    r"hicache_end=(?P<end>[0-9.]+) "
    r"hicache_consumer=(?P<consumer>-?\d+) .*?"
    r"tokens=(?P<tokens>\d+) cached=(?P<cached>\d+)"
)
DECODE_HANDOFF = re.compile(
    r"DUALPATH_DECODE_HANDOFF_BREAKDOWN rid=(?P<rid>\S+) "
    r"decode_engine=(?P<engine>\S+) .*?tp_rank=(?P<rank>\d+) .*?"
    r"prealloc_start=(?P<start>[0-9.]+) "
    r"prealloc_end=(?P<end>[0-9.]+)"
)
P_TRANSFER = re.compile(
    r"DUALPATH_TTFT_P_TRANSFER(?:_V2)? rid=(?P<rid>\S+) "
    r"(?:prefill_engine=(?P<engine>\S+) )?"
    r"(?:node=(?P<node>\S+) )?"
    r"transfer_start=(?P<start>[0-9.]+) "
    r"transfer_end=(?P<end>[0-9.]+)"
)
D_RECEIVE = re.compile(
    r"DUALPATH_TTFT_D_RECEIVE rid=(?P<rid>\S+) "
    r"receive_start=(?P<start>[0-9.]+) "
    r"receive_end=(?P<end>[0-9.]+) "
    r"receive_wait_s=(?P<duration>[0-9.]+)"
)
D_HOST_STAGING = re.compile(
    r"DUALPATH_TTFT_D_HOST_STAGING rid=(?P<rid>\S+) "
    r"tp_rank=(?P<rank>\d+) h2d_start=(?P<start>[0-9.]+) "
    r"h2d_end=(?P<end>[0-9.]+) h2d_s=(?P<duration>[0-9.]+)"
)
D_DISPATCH = re.compile(
    r"DUALPATH_TTFT_D_DISPATCH rid=(?P<rid>\S+) "
    r"dispatch_time=(?P<time>[0-9.]+) "
    r"ready_to_dispatch_s=(?P<duration>[0-9.]+)"
)
HTTP_YIELD = re.compile(
    r"DUALPATH_TTFT_HTTP_YIELD rid=(?P<rid>\S+) "
    r"handler_entry=(?P<entry>[0-9.]+) "
    r"yield_time=(?P<yield_time>[0-9.]+) "
    r"handler_to_yield_s=(?P<duration>[0-9.]+)"
)

PREFILL_STAGES = (
    "bootstrap_and_scheduling_seconds",
    "prefill_lustre_read_seconds",
    "prefill_scatter_seconds",
    "decode_lustre_read_seconds",
    "decode_scatter_seconds",
    "kv_read_overhead_seconds",
    "decode_l3_bootstrap_handoff_seconds",
    "decode_prealloc_seconds",
    "decode_to_prefill_kv_transfer_seconds",
    "prefill_host_to_device_seconds",
    "prefill_logits_sampling_seconds",
    "pd_transfer_seconds",
)
ALL_STAGES = PREFILL_STAGES + (
    "decode_host_to_device_seconds",
    "decode_dispatch_seconds",
    "sse_egress_seconds",
    "server_runtime_residual_seconds",
    "router_client_runtime_seconds",
)


def median_fields(
    values: dict[str, dict[str, list[float]]]
) -> dict[str, dict[str, float]]:
    return {
        rid: {name: statistics.median(items) for name, items in fields.items()}
        for rid, fields in values.items()
    }


def collect_logs(
    paths: list[Path],
) -> tuple[dict[str, dict[str, float]], ...]:
    request: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    l3_ranks: dict[
        str, dict[tuple[str, int], dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    decode_l3_ranks: dict[
        str, dict[tuple[str, int], dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    prefill_breakdown_ranks: dict[
        str, dict[tuple[str, int], dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    d2p_reverse_ranks: dict[
        str, dict[tuple[str, int], dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    prefill_h2d_ranks: dict[
        str, dict[int, dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    decode_prealloc_ranks: dict[
        str, dict[tuple[str, int], dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    transfer: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    receive: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    host_staging_ranks: dict[
        str, dict[int, dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    dispatch: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    http: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for path in paths:
        if path.name.startswith("prefill-"):
            patterns = (
                (REQ, request),
                (L3, None),
                (PREFILL_KV_READ_BREAKDOWN, None),
                (DECODE_L3, None),
                (D2P_REVERSE, None),
                (PREFILL_PREP, None),
                (P_TRANSFER, transfer),
            )
        elif path.name.startswith("decode-"):
            patterns = (
                (DECODE_L3, None),
                (DECODE_HANDOFF, None),
                (D_RECEIVE, receive),
                (D_HOST_STAGING, None),
                (D_DISPATCH, dispatch),
                (HTTP_YIELD, http),
            )
        else:
            continue
        for line in path.read_text(errors="replace").splitlines():
            for pattern, destination in patterns:
                match = pattern.search(line)
                if not match:
                    continue
                groups = match.groupdict()
                if pattern is REQ and groups["type"].lower() != "prefill":
                    continue
                rid = groups.pop("rid")
                groups.pop("type", None)
                if pattern is L3:
                    engine = groups.pop("engine") or "0"
                    groups.pop("node", None)
                    rank = int(groups.pop("rank"))
                    for name, value in groups.items():
                        l3_ranks[rid][(engine, rank)][name].append(float(value))
                    break
                if pattern is PREFILL_KV_READ_BREAKDOWN:
                    engine = groups.pop("engine")
                    groups.pop("node", None)
                    rank = int(groups.pop("rank"))
                    for name, value in groups.items():
                        prefill_breakdown_ranks[rid][(engine, rank)][name].append(
                            float(value)
                        )
                    break
                if pattern is DECODE_L3:
                    engine = groups.pop("engine")
                    groups.pop("node", None)
                    rank = int(groups.pop("rank"))
                    for name, value in groups.items():
                        if value is not None:
                            decode_l3_ranks[rid][(engine, rank)][name].append(
                                float(value)
                            )
                    break
                if pattern is D2P_REVERSE:
                    engine = groups.pop("engine")
                    groups.pop("node", None)
                    rank = int(groups.pop("rank"))
                    for name, value in groups.items():
                        d2p_reverse_ranks[rid][(engine, rank)][name].append(
                            float(value)
                        )
                    break
                if pattern is PREFILL_PREP:
                    rank = int(groups.pop("rank"))
                    start = float(groups.pop("start"))
                    end = float(groups.pop("end"))
                    tokens = float(groups.pop("tokens"))
                    cached = float(groups.pop("cached"))
                    consumer = float(groups.pop("consumer"))
                    for name, value in (
                        ("start", start),
                        ("end", end),
                        ("duration", max(0.0, end - start)),
                        ("tokens", tokens),
                        ("cached", cached),
                        ("consumer", consumer),
                    ):
                        prefill_h2d_ranks[rid][rank][name].append(value)
                    break
                if pattern is DECODE_HANDOFF:
                    engine = groups.pop("engine")
                    rank = int(groups.pop("rank"))
                    start = float(groups.pop("start"))
                    end = float(groups.pop("end"))
                    for name, value in (
                        ("start", start),
                        ("end", end),
                        ("duration", max(0.0, end - start)),
                    ):
                        decode_prealloc_ranks[rid][(engine, rank)][name].append(
                            value
                        )
                    break
                if pattern is D_HOST_STAGING:
                    rank = int(groups.pop("rank"))
                    for name, value in groups.items():
                        host_staging_ranks[rid][rank][name].append(float(value))
                    break
                assert destination is not None
                groups.pop("engine", None)
                groups.pop("node", None)
                for name, value in groups.items():
                    destination[rid][name].append(float(value))
                break
    l3: dict[str, dict[str, float]] = {}
    for rid, rank_values in l3_ranks.items():
        reduced = {
            key: {
                name: statistics.median(items)
                for name, items in fields.items()
            }
            for key, fields in rank_values.items()
        }
        (engine, rank), slowest = max(
            reduced.items(), key=lambda item: item[1]["duration"]
        )
        engines = {key[0] for key in reduced}
        if len(engines) != 1:
            raise RuntimeError(f"{rid} has L3 timing from engines {sorted(engines)}")
        union_start = min(item["start"] for item in reduced.values())
        union_end = max(item["end"] for item in reduced.values())
        l3[rid] = {
            **slowest,
            "start": union_start,
            "end": union_end,
            "duration": union_end - union_start,
            "engine": engine,
            "rank": float(rank),
            "rank_count": float(len(reduced)),
            "lustre": (
                sum(prefill_breakdown_ranks[rid][(engine, rank)]["lustre"])
                if prefill_breakdown_ranks[rid][(engine, rank)].get("lustre")
                else slowest["duration"]
            ),
            "scatter": sum(
                prefill_breakdown_ranks[rid][(engine, rank)].get("scatter", ())
            ),
        }
    host_staging: dict[str, dict[str, float]] = {}
    for rid, rank_values in host_staging_ranks.items():
        reduced = {
            rank: {
                name: statistics.median(items)
                for name, items in fields.items()
            }
            for rank, fields in rank_values.items()
        }
        slowest_rank, slowest = max(
            reduced.items(), key=lambda item: item[1]["duration"]
        )
        host_staging[rid] = {
            **slowest,
            "rank": float(slowest_rank),
            "rank_count": float(len(reduced)),
        }
    decode_l3: dict[str, dict[str, float]] = {}
    for rid, rank_values in decode_l3_ranks.items():
        reduced = {
            key: {
                name: statistics.median(items)
                for name, items in fields.items()
            }
            for key, fields in rank_values.items()
        }
        if len({key[0] for key in reduced}) != 1:
            raise RuntimeError(
                f"{rid} has Decode L3 timing from multiple engines"
            )
        if len(reduced) != 8:
            raise RuntimeError(f"{rid} has {len(reduced)}/8 Decode L3 timings")
        (slowest_engine, slowest_rank), slowest = max(
            reduced.items(), key=lambda item: item[1]["duration"]
        )
        union_start = min(item["start"] for item in reduced.values())
        union_end = max(item["end"] for item in reduced.values())
        decode_l3[rid] = {
            **slowest,
            "start": union_start,
            "end": union_end,
            "duration": union_end - union_start,
            "local_max_duration": slowest["duration"],
            "engine": slowest_engine,
            "rank": float(slowest_rank),
            "rank_count": float(len(reduced)),
            "lustre": slowest.get("lustre", slowest["duration"]),
            "scatter": slowest.get("scatter", 0.0),
        }
    d2p_reverse: dict[str, dict[str, float]] = {}
    for rid, rank_values in d2p_reverse_ranks.items():
        reduced = {
            key: {
                name: statistics.median(items)
                for name, items in fields.items()
            }
            for key, fields in rank_values.items()
        }
        engines = {key[0] for key in reduced}
        if len(engines) != 1:
            raise RuntimeError(
                f"{rid} has D-to-P timing from engines {sorted(engines)}"
            )
        if len(reduced) != 8:
            raise RuntimeError(f"{rid} has {len(reduced)}/8 D-to-P timings")
        token_counts = {int(item["tokens"]) for item in reduced.values()}
        if len(token_counts) != 1:
            raise RuntimeError(
                f"{rid} has inconsistent D-to-P tokens {sorted(token_counts)}"
            )
        union_start = min(item["start"] for item in reduced.values())
        union_end = max(item["end"] for item in reduced.values())
        d2p_reverse[rid] = {
            "start": union_start,
            "end": union_end,
            "duration": union_end - union_start,
            "tokens": float(token_counts.pop()),
            "rank_count": float(len(reduced)),
        }
    prefill_h2d: dict[str, dict[str, float]] = {}
    for rid, rank_values in prefill_h2d_ranks.items():
        reduced = {
            rank: {
                name: statistics.median(items)
                for name, items in fields.items()
            }
            for rank, fields in rank_values.items()
        }
        if len(reduced) != 8:
            raise RuntimeError(f"{rid} has {len(reduced)}/8 Prefill H2D timings")
        cached_tokens = {int(item["cached"]) for item in reduced.values()}
        if len(cached_tokens) != 1:
            raise RuntimeError(
                f"{rid} has inconsistent Prefill H2D cached tokens "
                f"{sorted(cached_tokens)}"
            )
        cached = cached_tokens.pop()
        consumers = {int(item["consumer"]) for item in reduced.values()}
        if len(consumers) != 1:
            raise RuntimeError(
                f"{rid} has inconsistent Prefill H2D consumer indices "
                f"{sorted(consumers)}"
            )
        consumer = consumers.pop()
        union_start = min(item["start"] for item in reduced.values())
        union_end = max(item["end"] for item in reduced.values())
        prefill_h2d[rid] = {
            "start": union_start,
            "end": union_end,
            "duration": (
                union_end - union_start if cached > 0 and consumer >= 0 else 0.0
            ),
            "tokens": float(cached if consumer >= 0 else 0),
            "consumer": float(consumer),
            "rank_count": float(len(reduced)),
        }
    decode_prealloc: dict[str, dict[str, float]] = {}
    for rid, rank_values in decode_prealloc_ranks.items():
        reduced = {
            key: {
                name: statistics.median(items)
                for name, items in fields.items()
            }
            for key, fields in rank_values.items()
        }
        if len(reduced) != 8:
            raise RuntimeError(
                f"{rid} has {len(reduced)}/8 Decode pre-allocation timings"
            )
        engines = {key[0] for key in reduced}
        if len(engines) != 1:
            raise RuntimeError(
                f"{rid} has Decode pre-allocation timing from engines "
                f"{sorted(engines)}"
            )
        union_start = min(item["start"] for item in reduced.values())
        union_end = max(item["end"] for item in reduced.values())
        decode_prealloc[rid] = {
            "start": union_start,
            "end": union_end,
            "duration": union_end - union_start,
            "rank_count": float(len(reduced)),
        }
    return (
        median_fields(request),
        l3,
        decode_l3,
        d2p_reverse,
        prefill_h2d,
        decode_prealloc,
        median_fields(transfer),
        median_fields(receive),
        host_staging,
        median_fields(dispatch),
        median_fields(http),
    )


def exclusive_prefill_stages(
    request: dict[str, float],
    l3: dict[str, float],
    decode_l3: dict[str, float],
    d2p_reverse: dict[str, float],
    prefill_h2d: dict[str, float],
    decode_prealloc: dict[str, float],
    transfer: dict[str, float],
) -> dict[str, float]:
    def split_read_window(
        total: float, detail: dict[str, float]
    ) -> tuple[float, float, float]:
        lustre = max(0.0, float(detail.get("lustre", 0.0)))
        scatter = max(0.0, float(detail.get("scatter", 0.0)))
        measured = lustre + scatter
        if measured > total and measured > 0:
            scale = total / measured
            lustre *= scale
            scatter *= scale
        overhead = max(0.0, total - lustre - scatter)
        return lustre, scatter, overhead

    intervals: list[tuple[str, float, float]] = []
    bootstrap_start = request["bootstrap_start"]
    queue_start = request["queue_start"]
    if bootstrap_start > 0 and queue_start >= bootstrap_start:
        intervals.append(
            ("bootstrap_and_scheduling_seconds", bootstrap_start, queue_start)
        )
    intervals.append(
        (
            "bootstrap_and_scheduling_seconds",
            request["queue_start"],
            request["queue_end"],
        )
    )
    if l3:
        intervals.append(
            ("_prefill_kv_read_window_seconds", l3["start"], l3["end"])
        )
    if d2p_reverse:
        intervals.append(
            (
                "decode_to_prefill_kv_transfer_seconds",
                d2p_reverse["start"],
                d2p_reverse["end"],
            )
        )
    if prefill_h2d.get("duration", 0.0) > 0:
        intervals.append(
            (
                "prefill_host_to_device_seconds",
                prefill_h2d["start"],
                prefill_h2d["end"],
            )
        )
    intervals.append(
        (
            "prefill_logits_sampling_seconds",
            request["prefill_start"],
            request["prefill_end"],
        )
    )
    intervals.append(
        ("pd_transfer_seconds", transfer["start"], transfer["end"])
    )

    priority_order = (
        "_prefill_kv_read_window_seconds",
        "decode_to_prefill_kv_transfer_seconds",
        "prefill_host_to_device_seconds",
        "bootstrap_and_scheduling_seconds",
        "prefill_logits_sampling_seconds",
        "pd_transfer_seconds",
    )
    priority = {name: index for index, name in enumerate(priority_order)}
    boundaries = sorted({point for _, start, end in intervals for point in (start, end)})
    result = {name: 0.0 for name in PREFILL_STAGES}
    result["_prefill_kv_read_window_seconds"] = 0.0
    for start, end in zip(boundaries, boundaries[1:]):
        active = [
            name
            for name, interval_start, interval_end in intervals
            if interval_start < end and interval_end > start
        ]
        if active:
            owner = min(active, key=priority.__getitem__)
            result[owner] += end - start
    # Decode and Prefill perf_counter clocks are node-local and cannot be
    # compared. The Prefill-side bootstrap envelope contains the wait for the
    # Decode L3 read. Remove its measured critical-path duration after the
    # Prefill-clock D-to-P interval has been assigned exclusively above. The
    # remainder has the same local bootstrap/queue meaning in both L3 paths.
    decode_duration = float(decode_l3.get("duration", 0.0))
    decode_overhead = 0.0
    if decode_duration > 0:
        attributed_decode_duration = min(
            decode_duration,
            result["bootstrap_and_scheduling_seconds"],
        )
        result["bootstrap_and_scheduling_seconds"] -= attributed_decode_duration
        (
            result["decode_lustre_read_seconds"],
            result["decode_scatter_seconds"],
            decode_overhead,
        ) = split_read_window(attributed_decode_duration, decode_l3)
        result["kv_read_overhead_seconds"] += decode_overhead
    prefill_read_window = result.pop("_prefill_kv_read_window_seconds")
    (
        result["prefill_lustre_read_seconds"],
        result["prefill_scatter_seconds"],
        prefill_overhead,
    ) = split_read_window(prefill_read_window, l3)
    result["kv_read_overhead_seconds"] += prefill_overhead
    if d2p_reverse and bootstrap_start > 0:
        pre_reverse_wait = max(0.0, d2p_reverse["start"] - bootstrap_start)
        handoff_duration = min(
            max(
                0.0,
                pre_reverse_wait
                - result["decode_lustre_read_seconds"]
                - result["decode_scatter_seconds"]
                - decode_overhead,
            ),
            result["bootstrap_and_scheduling_seconds"],
        )
        result["decode_l3_bootstrap_handoff_seconds"] = handoff_duration
        result["bootstrap_and_scheduling_seconds"] -= handoff_duration
        attributed_prealloc = min(
            float(decode_prealloc.get("duration", 0.0)),
            result["decode_l3_bootstrap_handoff_seconds"],
        )
        result["decode_prealloc_seconds"] = attributed_prealloc
        result["decode_l3_bootstrap_handoff_seconds"] -= attributed_prealloc
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--log-glob", default="*.log")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rounds = [json.loads(line) for line in args.rounds.read_text().splitlines()]
    logs = sorted(args.log_dir.rglob(args.log_glob))
    if summarize_v3(rounds, logs, args.output_dir):
        return
    (
        request,
        l3,
        decode_l3,
        d2p_reverse,
        prefill_h2d,
        decode_prealloc,
        transfer,
        receive,
        host_staging,
        dispatch,
        http,
    ) = collect_logs(logs)
    merged: list[dict[str, Any]] = []

    for round_item in rounds:
        rid = round_item["request_id"]
        missing = [
            name
            for name, values in (
                ("prefill request", request),
                ("P-to-D transfer", transfer),
                ("Decode receive", receive),
                ("Decode pre-allocation", decode_prealloc),
                ("Decode dispatch", dispatch),
                ("HTTP yield", http),
            )
            if rid not in values
        ]
        if missing:
            raise RuntimeError(f"{rid} missing timing logs: {', '.join(missing)}")

        decode_l3_item = decode_l3.get(rid, {})
        d2p_reverse_item = d2p_reverse.get(rid, {})
        if decode_l3_item.get("tokens", 0) > 0 and not d2p_reverse_item:
            raise RuntimeError(f"{rid} missing D-to-P reverse timing")
        prefill_h2d_item = prefill_h2d.get(rid, {})
        if l3.get(rid, {}).get("tokens", 0) > 0 and not prefill_h2d_item:
            raise RuntimeError(f"{rid} missing Prefill Host-to-GPU timing")
        stages = exclusive_prefill_stages(
            request[rid],
            l3.get(rid, {}),
            decode_l3_item,
            d2p_reverse_item,
            prefill_h2d_item,
            decode_prealloc[rid],
            transfer[rid],
        )
        decode_host_to_device = host_staging.get(rid, {}).get("duration", 0.0)
        decode_dispatch = max(
            0.0, dispatch[rid]["time"] - receive[rid]["end"]
        )
        http_yield_time = next(
            value for name, value in http[rid].items() if name.startswith("yield")
        )
        sse_egress = max(0.0, http_yield_time - dispatch[rid]["time"])
        directly_observed = (
            sum(stages.values())
            + decode_host_to_device
            + decode_dispatch
            + sse_egress
        )
        ttft = float(round_item["ttft_seconds"])
        handler_window = http[rid]["duration"]
        server_runtime_gap = max(0.0, handler_window - directly_observed)
        server_overlap_excess = max(0.0, directly_observed - handler_window)
        router_client_gap = max(0.0, ttft - handler_window)
        client_overlap_excess = max(0.0, handler_window - ttft)
        overlap_excess = server_overlap_excess + client_overlap_excess
        reconstructed_ttft = (
            directly_observed
            + server_runtime_gap
            - server_overlap_excess
            + router_client_gap
            - client_overlap_excess
        )
        accounting_error = ttft - reconstructed_ttft
        request_latency = float(round_item["latency_seconds"])
        decode_tail = max(0.0, request_latency - ttft)
        request_accounting_error = request_latency - (reconstructed_ttft + decode_tail)
        merged.append(
            {
                **round_item,
                **stages,
                "decode_host_to_device_seconds": decode_host_to_device,
                "decode_dispatch_seconds": decode_dispatch,
                "sse_egress_seconds": sse_egress,
                "server_runtime_residual_seconds": server_runtime_gap,
                "router_client_runtime_seconds": router_client_gap,
                "cross_node_overlap_excess_seconds": overlap_excess,
                "server_overlap_excess_seconds": server_overlap_excess,
                "client_overlap_excess_seconds": client_overlap_excess,
                "accounting_error_seconds": accounting_error,
                "decode_generation_tail_seconds": decode_tail,
                "request_accounting_error_seconds": request_accounting_error,
                "decode_receive_wait_seconds_diagnostic": receive[rid][
                    "duration"
                ],
                "decode_host_to_device_slowest_tp_rank_diagnostic": (
                    int(host_staging[rid]["rank"]) if rid in host_staging else -1
                ),
                "decode_host_to_device_tp_rank_count_diagnostic": (
                    int(host_staging[rid]["rank_count"])
                    if rid in host_staging
                    else 0
                ),
                "decode_handler_to_yield_seconds_diagnostic": http[rid][
                    "duration"
                ],
                "prefill_l3_kv_read_wall_seconds_diagnostic": (
                    l3[rid]["end"] - l3[rid]["start"] if rid in l3 else 0.0
                ),
                "decode_l3_kv_read_seconds_diagnostic": (
                    decode_l3[rid]["duration"] if rid in decode_l3 else 0.0
                ),
                "decode_l3_local_max_rank_seconds_diagnostic": (
                    decode_l3[rid]["local_max_duration"]
                    if rid in decode_l3
                    else 0.0
                ),
                "decode_l3_slowest_tp_rank_diagnostic": (
                    int(decode_l3[rid]["rank"]) if rid in decode_l3 else -1
                ),
                "decode_l3_tp_rank_count_diagnostic": (
                    int(decode_l3[rid]["rank_count"])
                    if rid in decode_l3
                    else 0
                ),
                "decode_to_prefill_kv_transfer_wall_seconds_diagnostic": (
                    d2p_reverse[rid]["duration"] if rid in d2p_reverse else 0.0
                ),
                "decode_to_prefill_kv_transfer_tp_rank_count_diagnostic": (
                    int(d2p_reverse[rid]["rank_count"])
                    if rid in d2p_reverse
                    else 0
                ),
                "prefill_host_to_device_wall_seconds_diagnostic": (
                    prefill_h2d_item.get("duration", 0.0)
                ),
                "prefill_host_to_device_tokens_diagnostic": int(
                    prefill_h2d_item.get("tokens", 0)
                ),
                "prefill_host_to_device_tp_rank_count_diagnostic": int(
                    prefill_h2d_item.get("rank_count", 0)
                ),
                "decode_prealloc_wall_seconds_diagnostic": decode_prealloc[rid][
                    "duration"
                ],
                "decode_prealloc_tp_rank_count_diagnostic": int(
                    decode_prealloc[rid]["rank_count"]
                ),
                "l3_slowest_tp_rank_diagnostic": (
                    int(l3[rid]["rank"]) if rid in l3 else -1
                ),
                "l3_tp_rank_count_diagnostic": (
                    int(l3[rid]["rank_count"]) if rid in l3 else 0
                ),
                "has_l3_read": rid in l3 or rid in decode_l3,
                "has_prefill_l3_read": rid in l3,
                "has_decode_l3_read": rid in decode_l3,
            }
        )

    def mean(name: str) -> float:
        return statistics.fmean(float(item[name]) for item in merged)

    summary = {
        "requests": len(merged),
        "sampling_semantics": (
            "Included in Prefill Forward + Logits + Sampling. With temperature=0, "
            "sampling selects the highest-logit token deterministically."
        ),
        "ttft_mean_seconds": mean("ttft_seconds"),
        "stage_mean_seconds": {name: mean(name) for name in ALL_STAGES},
        "cross_node_overlap_excess_mean_seconds": mean(
            "cross_node_overlap_excess_seconds"
        ),
        "prefill_l3_kv_read_wall_mean_seconds_diagnostic": mean(
            "prefill_l3_kv_read_wall_seconds_diagnostic"
        ),
        "prefill_lustre_read_mean_seconds": mean("prefill_lustre_read_seconds"),
        "prefill_scatter_mean_seconds": mean("prefill_scatter_seconds"),
        "decode_lustre_read_mean_seconds": mean("decode_lustre_read_seconds"),
        "decode_scatter_mean_seconds": mean("decode_scatter_seconds"),
        "kv_read_overhead_mean_seconds": mean("kv_read_overhead_seconds"),
        "decode_l3_kv_read_wall_mean_seconds_diagnostic": mean(
            "decode_l3_kv_read_seconds_diagnostic"
        ),
        "decode_to_prefill_kv_transfer_mean_seconds": mean(
            "decode_to_prefill_kv_transfer_seconds"
        ),
        "decode_l3_bootstrap_handoff_mean_seconds": mean(
            "decode_l3_bootstrap_handoff_seconds"
        ),
        "prefill_host_to_device_mean_seconds": mean(
            "prefill_host_to_device_seconds"
        ),
        "prefill_host_to_device_wall_mean_seconds_diagnostic": mean(
            "prefill_host_to_device_wall_seconds_diagnostic"
        ),
        "decode_prealloc_mean_seconds": mean("decode_prealloc_seconds"),
        "decode_prealloc_wall_mean_seconds_diagnostic": mean(
            "decode_prealloc_wall_seconds_diagnostic"
        ),
        "accounting_error_max_abs_seconds": max(
            abs(float(item["accounting_error_seconds"])) for item in merged
        ),
        "request_accounting_error_max_abs_seconds": max(
            abs(float(item["request_accounting_error_seconds"]))
            for item in merged
        ),
        "timing_coverage": {
            "prefill_request": len(request),
            "l3_read": sum(1 for item in merged if item["has_l3_read"]),
            "prefill_l3_read": sum(1 for item in merged if item["has_prefill_l3_read"]),
            "decode_l3_read": sum(1 for item in merged if item["has_decode_l3_read"]),
            "decode_to_prefill_kv_transfer": len(d2p_reverse),
            "prefill_host_to_device": sum(
                1
                for item in merged
                if item["prefill_host_to_device_tokens_diagnostic"] > 0
            ),
            "decode_prealloc": sum(
                1
                for item in merged
                if item["decode_prealloc_tp_rank_count_diagnostic"] == 8
            ),
            "p_to_d_transfer": len(transfer),
            "decode_receive": len(receive),
            "decode_host_staging": len(host_staging),
            "decode_dispatch": len(dispatch),
            "http_yield": len(http),
        },
    }
    with (args.output_dir / "fullchain_rounds.jsonl").open("w") as stream:
        for item in merged:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
    (args.output_dir / "fullchain_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if summary["accounting_error_max_abs_seconds"] > 1e-6:
        raise RuntimeError(f"TTFT accounting did not close: {summary}")
    if summary["request_accounting_error_max_abs_seconds"] > 1e-6:
        raise RuntimeError(f"request accounting did not close: {summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
