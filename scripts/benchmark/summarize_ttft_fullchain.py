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
    r"(?:prefill_engine=(?P<engine>\S+) )?tp_rank=(?P<rank>\d+) "
    r"read_start=(?P<start>[0-9.]+) read_end=(?P<end>[0-9.]+) "
    r"read_s=(?P<duration>[0-9.]+) tokens=(?P<tokens>\d+)"
)
P_TRANSFER = re.compile(
    r"DUALPATH_TTFT_P_TRANSFER(?:_V2)? rid=(?P<rid>\S+) "
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
    "l3_kv_read_seconds",
    "prefill_logits_sampling_seconds",
    "pd_transfer_seconds",
)
ALL_STAGES = PREFILL_STAGES + (
    "decode_host_to_device_seconds",
    "decode_dispatch_seconds",
    "sse_egress_seconds",
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
                (P_TRANSFER, transfer),
            )
        elif path.name.startswith("decode-"):
            patterns = (
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
                    rank = int(groups.pop("rank"))
                    for name, value in groups.items():
                        l3_ranks[rid][(engine, rank)][name].append(float(value))
                    break
                if pattern is D_HOST_STAGING:
                    rank = int(groups.pop("rank"))
                    for name, value in groups.items():
                        host_staging_ranks[rid][rank][name].append(float(value))
                    break
                assert destination is not None
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
    return (
        median_fields(request),
        l3,
        median_fields(transfer),
        median_fields(receive),
        host_staging,
        median_fields(dispatch),
        median_fields(http),
    )


def exclusive_prefill_stages(
    request: dict[str, float],
    l3: dict[str, float],
    transfer: dict[str, float],
) -> dict[str, float]:
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
        intervals.append(("l3_kv_read_seconds", l3["start"], l3["end"]))
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
        "l3_kv_read_seconds",
        "bootstrap_and_scheduling_seconds",
        "prefill_logits_sampling_seconds",
        "pd_transfer_seconds",
    )
    priority = {name: index for index, name in enumerate(priority_order)}
    boundaries = sorted({point for _, start, end in intervals for point in (start, end)})
    result = {name: 0.0 for name in PREFILL_STAGES}
    for start, end in zip(boundaries, boundaries[1:]):
        active = [
            name
            for name, interval_start, interval_end in intervals
            if interval_start < end and interval_end > start
        ]
        if active:
            owner = min(active, key=priority.__getitem__)
            result[owner] += end - start
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
    logs = sorted(args.log_dir.glob(args.log_glob))
    request, l3, transfer, receive, host_staging, dispatch, http = collect_logs(logs)
    merged: list[dict[str, Any]] = []

    for round_item in rounds:
        rid = round_item["request_id"]
        missing = [
            name
            for name, values in (
                ("prefill request", request),
                ("P-to-D transfer", transfer),
                ("Decode receive", receive),
                ("Decode dispatch", dispatch),
                ("HTTP yield", http),
            )
            if rid not in values
        ]
        if missing:
            raise RuntimeError(f"{rid} missing timing logs: {', '.join(missing)}")

        stages = exclusive_prefill_stages(
            request[rid], l3.get(rid, {}), transfer[rid]
        )
        decode_host_to_device = host_staging.get(rid, {}).get("duration", 0.0)
        decode_dispatch = max(
            0.0, dispatch[rid]["time"] - receive[rid]["end"]
        )
        http_yield_time = next(
            value for name, value in http[rid].items() if name.startswith("yield")
        )
        sse_egress = max(0.0, http_yield_time - dispatch[rid]["time"])
        known = (
            sum(stages.values())
            + decode_host_to_device
            + decode_dispatch
            + sse_egress
        )
        ttft = float(round_item["ttft_seconds"])
        runtime_gap = max(0.0, ttft - known)
        overlap_excess = max(0.0, known - ttft)
        accounting_error = ttft - (known + runtime_gap - overlap_excess)
        request_latency = float(round_item["latency_seconds"])
        decode_tail = max(0.0, request_latency - ttft)
        request_accounting_error = request_latency - (
            known + runtime_gap - overlap_excess + decode_tail
        )
        merged.append(
            {
                **round_item,
                **stages,
                "decode_host_to_device_seconds": decode_host_to_device,
                "decode_dispatch_seconds": decode_dispatch,
                "sse_egress_seconds": sse_egress,
                "router_client_runtime_seconds": runtime_gap,
                "cross_node_overlap_excess_seconds": overlap_excess,
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
                "l3_kv_read_wall_seconds_diagnostic": (
                    l3[rid]["end"] - l3[rid]["start"] if rid in l3 else 0.0
                ),
                "l3_slowest_tp_rank_diagnostic": (
                    int(l3[rid]["rank"]) if rid in l3 else -1
                ),
                "l3_tp_rank_count_diagnostic": (
                    int(l3[rid]["rank_count"]) if rid in l3 else 0
                ),
                "has_l3_read": rid in l3,
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
        "l3_kv_read_wall_mean_seconds_diagnostic": mean(
            "l3_kv_read_wall_seconds_diagnostic"
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
