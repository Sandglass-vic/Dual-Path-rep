#!/usr/bin/env python3
"""Summarize the cold/hot x buffered/O_DIRECT KV read matrix.

For every quadrant this reports the end-to-end bandwidth, the file I/O and
staging-scatter times as separate components, and the per-page latency
distribution behind them.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from marker_log import records_with_prefix  # noqa: E402

RESULT_PREFIX = "KVIO_RESULT "
CACHE_PROBE_PREFIX = "KVIO_CACHE_PROBE "
QUADRANTS = ("cold-buffered", "cold-direct", "hot-buffered", "hot-direct")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def describe(values: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(values)
    return {
        "values": values,
        "mean": mean,
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "cv": statistics.pstdev(values) / mean if len(values) > 1 and mean else 0.0,
    }


def average_distribution(distributions: list[dict[str, Any]]) -> dict[str, Any]:
    present = [item for item in distributions if item.get("count")]
    if not present:
        return {"count": 0}
    keys = ("mean_ns", "p50_ns", "p90_ns", "p99_ns")
    result: dict[str, Any] = {
        "count": sum(int(item["count"]) for item in present),
        "min_ns": min(float(item["min_ns"]) for item in present),
        "max_ns": max(float(item["max_ns"]) for item in present),
    }
    for key in keys:
        result[key] = statistics.fmean(float(item[key]) for item in present)
    return result


def summarize_quadrant(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = describe([float(item["gib_s"]) for item in records])
    file_io = describe([float(item["file_io_gib_s"]) for item in records])
    scatter_rates = [
        float(item["scatter_gib_s"])
        for item in records
        if item.get("scatter_gib_s") is not None
    ]
    return {
        "repetitions": len(records),
        "io_mode": records[0]["io_mode"],
        "cache_state": "hot" if not records[0]["drop_cache"] else "cold",
        "gib_per_repetition": float(records[0]["gib"]),
        "bandwidth_gib_s": total,
        "file_io_gib_s": file_io,
        "scatter_gib_s": describe(scatter_rates) if scatter_rates else None,
        "seconds": describe([float(item["seconds"]) for item in records]),
        "file_io_seconds": describe(
            [float(item["file_io_seconds"]) for item in records]
        ),
        "scatter_seconds": describe(
            [float(item["scatter_seconds"]) for item in records]
        ),
        "file_io_fraction": describe(
            [float(item["file_io_fraction"]) for item in records]
        ),
        "scatter_fraction": describe(
            [float(item["scatter_fraction"]) for item in records]
        ),
        "page_latency_ns": {
            "open": average_distribution([item["open_ns"] for item in records]),
            "read": average_distribution([item["read_ns"] for item in records]),
            "close": average_distribution([item["close_ns"] for item in records]),
            "scatter": average_distribution(
                [item["scatter_step_ns"] for item in records]
            ),
        },
    }


def render_table(summary: dict[str, Any]) -> str:
    header = (
        f"{'quadrant':16s}{'GiB/s':>9s}{'fileIO':>9s}{'scatter':>9s}"
        f"{'io%':>7s}{'scat%':>7s}{'read p50':>10s}{'read p99':>10s}"
        f"{'scatter p50':>13s}{'CV':>7s}"
    )
    lines = [
        f"KVIO cache matrix on {summary['host']}"
        f"  ({summary['quadrants']['cold-buffered']['gib_per_repetition']:.1f} GiB"
        f" x {summary['quadrants']['cold-buffered']['repetitions']} repetitions,"
        f" O_DIRECT bypass verified={summary['direct_bypassed_page_cache']})",
        header,
        "-" * len(header),
    ]
    for quadrant, item in summary["quadrants"].items():
        scatter = item["scatter_gib_s"]
        read = item["page_latency_ns"]["read"]
        scatter_latency = item["page_latency_ns"]["scatter"]
        lines.append(
            f"{quadrant:16s}"
            f"{float(item['bandwidth_gib_s']['median']):9.3f}"
            f"{float(item['file_io_gib_s']['median']):9.3f}"
            + (f"{float(scatter['median']):9.3f}" if scatter else f"{'n/a':>9s}")
            + f"{float(item['file_io_fraction']['median']) * 100:7.1f}"
            f"{float(item['scatter_fraction']['median']) * 100:7.1f}"
            f"{float(read['p50_ns']) / 1e6:9.3f}m"
            f"{float(read['p99_ns']) / 1e6:9.3f}m"
            + (
                f"{float(scatter_latency['p50_ns']) / 1e6:12.3f}m"
                if scatter_latency.get("count")
                else f"{'n/a':>13s}"
            )
            + f"{float(item['bandwidth_gib_s']['cv']) * 100:6.1f}%"
        )
    lines.append("")
    for name, value in summary["ratios"].items():
        lines.append(f"  {name} = {value:.3f}x")
    lines.append(f"  {summary['interpretation']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="")
    parser.add_argument(
        "--records-dir",
        type=Path,
        help="directory of per-phase record JSON files, preferred over the log",
    )
    args = parser.parse_args()

    records = []
    if args.records_dir and args.records_dir.is_dir():
        records = [
            json.loads(path.read_text())
            for path in sorted(args.records_dir.glob("*.json"))
        ]
    if not records:
        records = records_with_prefix(args.log, RESULT_PREFIX)
    if not records:
        raise RuntimeError(f"no {RESULT_PREFIX.strip()} records found in {args.log}")
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_phase[str(record["phase"])].append(record)
    missing = [phase for phase in QUADRANTS if phase not in by_phase]
    if missing:
        raise RuntimeError(f"missing quadrants: {missing}; found {sorted(by_phase)}")

    quadrants = {
        phase: summarize_quadrant(by_phase[phase]) for phase in QUADRANTS
    }
    medians = {
        phase: float(payload["bandwidth_gib_s"]["median"])
        for phase, payload in quadrants.items()
    }
    io_medians = {
        phase: float(payload["file_io_gib_s"]["median"])
        for phase, payload in quadrants.items()
    }

    probes = records_with_prefix(args.log, CACHE_PROBE_PREFIX)
    probe = probes[-1] if probes else None
    bypass_verified = None
    if probe is not None:
        bypass_verified = bool(
            probe["resident_after_direct"] <= probe["resident_before"]
            and probe["resident_after_buffered"] > probe["resident_before_buffered"]
        )

    summary = {
        "schema_version": 2,
        "host": args.host or records[0]["host"],
        "cache_probe": probe,
        "direct_bypassed_page_cache": bypass_verified,
        "quadrants": quadrants,
        "ratios": {
            "hot_over_cold_buffered": medians["hot-buffered"]
            / medians["cold-buffered"],
            "cold_direct_over_cold_buffered": medians["cold-direct"]
            / medians["cold-buffered"],
            "hot_direct_over_hot_buffered": medians["hot-direct"]
            / medians["hot-buffered"],
            "hot_direct_over_cold_direct": medians["hot-direct"]
            / medians["cold-direct"],
        },
        "median_bandwidth_gib_s": medians,
        "median_file_io_gib_s": io_medians,
        "interpretation": (
            "O_DIRECT ignores the hot page cache"
            if medians["hot-direct"] < medians["hot-buffered"] * 0.8
            else "O_DIRECT was not clearly separated from hot buffered bandwidth"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(render_table(summary), flush=True)


if __name__ == "__main__":
    main()
