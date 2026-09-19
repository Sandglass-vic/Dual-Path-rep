#!/usr/bin/env python3
"""Summarize a pza100 cluster I/O benchmark run."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from marker_log import records_with_prefix as load_marker_records  # noqa: E402

KVWRITE_PREFIX = "KVWRITE_RESULT "
RAW_NAME = re.compile(
    r"^(?P<phase>.+)\.(?P<mode>direct|buffered-cold|buffered-hot)"
    r"\.r(?P<repetition>\d+)\.(?P<host>g[56])\.json$"
)
STATE_NAME = re.compile(
    r"^(?P<label>.+)\.(?P<point>before|after)\.(?P<host>g[56])\.json$"
)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty list")
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


def describe(values: list[float]) -> dict[str, float | list[float]]:
    mean = statistics.mean(values)
    return {
        "values": values,
        "mean": mean,
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "cv": statistics.pstdev(values) / mean if len(values) > 1 and mean else 0.0,
    }


def fio_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("backend") == "python-posix":
        return {
            "backend": "python-posix",
            "direction": payload["direction"],
            "bytes": int(payload["bytes"]),
            "seconds": float(payload["seconds"]),
            "gib_s": float(payload["gib_s"]),
            "latency_ns": payload["latency_ns"],
        }
    jobs = payload.get("jobs", [])
    if not jobs:
        raise RuntimeError(f"fio output has no jobs: {path}")
    job = jobs[0]
    candidates = [
        (name, job[name])
        for name in ("read", "write")
        if int(job.get(name, {}).get("io_bytes", 0)) > 0
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"cannot identify fio direction in {path}")
    direction, stats = candidates[0]
    runtime_seconds = float(stats["runtime"]) / 1000
    io_bytes = int(stats["io_bytes"])
    percentiles = stats.get("clat_ns", {}).get("percentile", {})

    def fio_percentile(target: float) -> float | None:
        if not percentiles:
            return None
        key = min(percentiles, key=lambda item: abs(float(item) - target))
        return float(percentiles[key])

    return {
        "backend": "fio",
        "direction": direction,
        "bytes": io_bytes,
        "seconds": runtime_seconds,
        "gib_s": io_bytes / runtime_seconds / 2**30,
        "latency_ns": {
            "p50": fio_percentile(50.0),
            "p95": fio_percentile(95.0),
        },
    }


def summarize_raw(result_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((result_root / "raw").glob("*/*.json")):
        match = RAW_NAME.match(path.name)
        if not match:
            continue
        record = fio_record(path)
        record["host"] = match.group("host")
        key = (
            path.parent.name,
            match.group("phase"),
            match.group("mode"),
            int(match.group("repetition")),
        )
        grouped[key].append(record)

    repetitions = []
    for (case_id, phase, mode, repetition), records in sorted(grouped.items()):
        expected_hosts = 2 if phase.startswith(("dual-", "mixed")) else 1
        if len(records) != expected_hosts:
            continue
        total_bytes = sum(int(record["bytes"]) for record in records)
        seconds = max(float(record["seconds"]) for record in records)
        repetitions.append(
            {
                "case": case_id,
                "phase": phase,
                "mode": mode,
                "repetition": repetition,
                "hosts": [record["host"] for record in records],
                "backends": sorted({record["backend"] for record in records}),
                "directions": {record["host"]: record["direction"] for record in records},
                "bytes": total_bytes,
                "aggregate_gib_s": total_bytes / seconds / 2**30,
                "per_host_gib_s": {
                    record["host"]: record["gib_s"] for record in records
                },
                "per_host_latency_ns": {
                    record["host"]: record["latency_ns"] for record in records
                },
            }
        )

    by_series: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    hosts_by_series: dict[
        tuple[str, str, str], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    latency_by_series: dict[
        tuple[str, str, str], dict[str, list[float]]
    ] = defaultdict(lambda: {"p50": [], "p95": []})
    for item in repetitions:
        key = (item["case"], item["phase"], item["mode"])
        by_series[key].append(
            float(item["aggregate_gib_s"])
        )
        for host, rate in item["per_host_gib_s"].items():
            hosts_by_series[key][host].append(float(rate))
        for latency in item["per_host_latency_ns"].values():
            for percentile_name in ("p50", "p95"):
                value = latency[percentile_name]
                if value is not None:
                    latency_by_series[key][percentile_name].append(float(value))
    series = [
        {
            "case": case,
            "phase": phase,
            "mode": mode,
            "bandwidth_gib_s": describe(values),
            "per_host_bandwidth_gib_s": {
                host: describe(host_values)
                for host, host_values in sorted(
                    hosts_by_series[(case, phase, mode)].items()
                )
            },
            "latency_ns": {
                name: describe(latency_values)
                for name, latency_values in latency_by_series[(case, phase, mode)].items()
                if latency_values
            },
        }
        for (case, phase, mode), values in sorted(by_series.items())
    ]

    series_index = {
        (item["case"], item["phase"], item["mode"]): item for item in series
    }
    comparisons = []
    cases = sorted({item["case"] for item in series})
    modes = sorted({item["mode"] for item in series})
    for case in cases:
        for mode in modes:
            for operation in ("read", "write"):
                keys = [
                    (case, f"g5-{operation}", mode),
                    (case, f"g6-{operation}", mode),
                    (case, f"dual-{operation}", mode),
                ]
                if not all(key in series_index for key in keys):
                    continue
                g5, g6, dual = (
                    float(series_index[key]["bandwidth_gib_s"]["median"])
                    for key in keys
                )
                baseline = statistics.mean((g5, g6))
                dual_hosts = series_index[keys[2]]["per_host_bandwidth_gib_s"]
                dual_g5 = float(dual_hosts["g5"]["median"])
                dual_g6 = float(dual_hosts["g6"]["median"])
                comparisons.append(
                    {
                        "case": case,
                        "mode": mode,
                        "operation": operation,
                        "g5_median_gib_s": g5,
                        "g6_median_gib_s": g6,
                        "dual_aggregate_median_gib_s": dual,
                        "aggregate_scaling": dual / baseline,
                        "per_node_retention": (dual / 2) / baseline,
                        "dual_to_sum_of_singles": dual / (g5 + g6),
                        "g5_retention": dual_g5 / g5,
                        "g6_retention": dual_g6 / g6,
                    }
                )
    return repetitions, series + comparisons


def summarize_writes(result_root: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    write_root = result_root / "kv-write"
    if not write_root.exists():
        return []
    for path in sorted(write_root.glob("*.log")):
        parts = path.stem.split(".")
        if len(parts) < 3:
            continue
        phase, sync_mode = parts[0], parts[1]
        for record in load_marker_records(path, KVWRITE_PREFIX):
            grouped[(phase, sync_mode, int(record["repetition"]))].append(record)
    repetitions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (phase, sync_mode, repetition), records in sorted(grouped.items()):
        total_bytes = sum(int(record["bytes"]) for record in records)
        buffered_seconds = max(float(record["write_seconds"]) for record in records)
        durable_values = [record["durable_gib_s"] for record in records]
        durable_seconds = max(float(record["seconds"]) for record in records)
        durable_aggregate = (
            total_bytes / durable_seconds / 2**30
            if all(value is not None for value in durable_values)
            else None
        )
        repetitions[(phase, sync_mode)].append(
            {
                "repetition": repetition,
                "hosts": [record["host"] for record in records],
                "bytes": total_bytes,
                "buffered_aggregate_gib_s": (
                    total_bytes / buffered_seconds / 2**30
                ),
                "durable_aggregate_gib_s": durable_aggregate,
                "per_host_gib_s": {
                    record["host"]: float(record["gib_s"]) for record in records
                },
                "records": records,
            }
        )
    result = []
    for (phase, sync_mode), items in sorted(repetitions.items()):
        buffered_rates = [
            float(item["buffered_aggregate_gib_s"]) for item in items
        ]
        durable_rates = [
            float(item["durable_aggregate_gib_s"]) for item in items
            if item["durable_aggregate_gib_s"] is not None
        ]
        result.append(
            {
                "phase": phase,
                "sync_mode": sync_mode,
                "repetitions": items,
                "buffered_bandwidth_gib_s": describe(buffered_rates),
                "durable_bandwidth_gib_s": (
                    describe(durable_rates) if durable_rates else None
                ),
            }
        )
    return result


def nested_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    interface_before = before["interface"]["counters"] or {}
    interface_after = after["interface"]["counters"] or {}
    interface = {
        key: int(interface_after[key]) - int(interface_before[key])
        for key in interface_before.keys() & interface_after.keys()
        if key.endswith("_bytes") or key.endswith("_packets")
    }
    hca: dict[str, dict[str, int]] = {}
    for device in before["hca"].keys() & after["hca"].keys():
        hca[device] = {}
        for key in before["hca"][device].keys() & after["hca"][device].keys():
            delta = int(after["hca"][device][key]) - int(before["hca"][device][key])
            if key.endswith("_data"):
                delta *= 4
            hca[device][f"{key}_bytes" if key.endswith("_data") else key] = delta
    rpc: dict[str, dict[str, list[int]]] = {}
    for endpoint in before["rpc"].keys() & after["rpc"].keys():
        rpc[endpoint] = {}
        for key in before["rpc"][endpoint].keys() & after["rpc"][endpoint].keys():
            old = before["rpc"][endpoint][key]
            new = after["rpc"][endpoint][key]
            if len(old) == len(new):
                rpc[endpoint][key] = [
                    int(new_value) - int(old_value)
                    for old_value, new_value in zip(old, new)
                ]
    return {"interface": interface, "hca": hca, "rpc": rpc}


def summarize_telemetry(result_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    telemetry_root = result_root / "telemetry"
    states: dict[tuple[str, str, str], dict[str, Any]] = {}
    environment: dict[str, Any] = {}
    if not telemetry_root.exists():
        return environment, []
    for path in telemetry_root.glob("*.json"):
        payload = json.loads(path.read_text())
        if path.name.startswith("preflight.") or path.name.startswith("completed."):
            environment[f"{payload['label']}:{payload['host']}"] = payload
        match = STATE_NAME.match(path.name)
        if match:
            states[(match.group("label"), match.group("point"), match.group("host"))] = payload
    deltas = []
    labels = sorted({key[0] for key in states})
    for label in labels:
        for host in ("g5", "g6"):
            before = states.get((label, "before", host))
            after = states.get((label, "after", host))
            if before and after:
                deltas.append(
                    {
                        "label": label,
                        "host": host,
                        **nested_delta(before, after),
                    }
                )
    return environment, deltas


def load_optional_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def theoretical_limits(environment: dict[str, Any]) -> dict[str, Any]:
    """Per-host hardware ceilings captured from sysfs during preflight."""
    limits: dict[str, Any] = {}
    for key, state in sorted(environment.items()):
        label, _, host = key.partition(":")
        if label != "preflight" or "theoretical" not in state:
            continue
        limits[host] = state["theoretical"]
    return limits


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# pza100 I/O Benchmark Report",
        "",
        f"Run root: `{summary['run_root']}`",
        "",
        "## Environment",
        "",
    ]
    for key, state in sorted(summary["environment"].items()):
        mount = state["mount"]
        lines.append(
            f"- `{key}`: `{mount['filesystem']}` from `{mount['source']}` "
            f"at `{mount['mountpoint']}`"
        )

    lines.extend(["", "## Theoretical ceilings", ""])
    if summary["theoretical"]:
        for host, payload in sorted(summary["theoretical"].items()):
            interface = payload["interface"]
            port = payload["storage_interface_port"]
            lines.append(f"- `{host}`:")
            if port:
                lines.append(
                    f"  - storage NIC: `{port['pci_address']}` "
                    f"{port['rate']} = {float(port['theoretical_gib_s']):.2f} GiB/s; "
                    f"HCA PCIe {port['pcie']['current_link_speed']} x"
                    f"{port['pcie']['current_link_width']} = "
                    f"{float(port['pcie']['theoretical_gib_s']):.2f} GiB/s"
                )
            lines.append(
                f"  - IP interface: MTU {interface['mtu']}, "
                f"IPoIB mode {interface['ipoib_mode']}"
            )
            block = payload["block_device"]
            if block and block.get("pcie"):
                lines.append(
                    f"  - backing device: `{block['model']}` on "
                    f"{block['pcie']['current_link_speed']} x"
                    f"{block['pcie']['current_link_width']} = "
                    f"{float(block['pcie']['theoretical_gib_s']):.2f} GiB/s"
                )
            if payload["binding_limit_gib_s"] is not None:
                lines.append(
                    "  - binding hardware limit: "
                    f"{float(payload['binding_limit_gib_s']):.2f} GiB/s"
                )
    else:
        lines.append("- Not captured.")

    lines.extend(["", "## Cache and O_DIRECT quadrants", ""])
    if summary["cache_matrix"]:
        for name, payload in sorted(summary["cache_matrix"].items()):
            if payload.get("status"):
                lines.append(f"- `{name}`: {payload['status']}")
                continue
            if "quadrants" not in payload:
                lines.append(f"- `{name}`: completed")
                continue
            lines.extend(
                [
                    f"- `{name}`: O_DIRECT page-cache bypass verified="
                    f"{payload['direct_bypassed_page_cache']}",
                    "",
                    "  | quadrant | total GiB/s | file I/O GiB/s | scatter GiB/s |"
                    " file I/O share | scatter share | page read p50 / p99 |",
                    "  |---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for quadrant, item in payload["quadrants"].items():
                scatter = item["scatter_gib_s"]
                latency = item["page_latency_ns"]["read"]
                lines.append(
                    f"  | `{quadrant}` | "
                    f"{float(item['bandwidth_gib_s']['median']):.3f} | "
                    f"{float(item['file_io_gib_s']['median']):.3f} | "
                    + (
                        f"{float(scatter['median']):.3f} | "
                        if scatter
                        else "n/a | "
                    )
                    + f"{float(item['file_io_fraction']['median']):.1%} | "
                    f"{float(item['scatter_fraction']['median']):.1%} | "
                    f"{float(latency['p50_ns']) / 1e6:.3f} / "
                    f"{float(latency['p99_ns']) / 1e6:.3f} ms |"
                )
            lines.append("")
    else:
        lines.append("- Not run.")

    lines.extend(["", "## Raw fio medians", ""])
    for item in summary["raw_series"]:
        if "bandwidth_gib_s" not in item:
            continue
        stats = item["bandwidth_gib_s"]
        warning = " (unstable: CV > 10%)" if float(stats["cv"]) > 0.10 else ""
        lines.append(
            f"- `{item['case']} / {item['phase']} / {item['mode']}`: "
            f"{float(stats['median']):.3f} GiB/s median, "
            f"p95 {float(stats['p95']):.3f}, CV {float(stats['cv']):.1%}{warning}"
        )

    lines.extend(["", "## Dual-node scaling", ""])
    for item in summary["raw_comparisons"]:
        lines.append(
            f"- `{item['case']} / {item['operation']} / {item['mode']}`: "
            f"aggregate scaling {float(item['aggregate_scaling']):.3f}x, "
            f"dual/sum(single) {float(item['dual_to_sum_of_singles']):.1%}, "
            f"g5 retention {float(item['g5_retention']):.1%}, "
            f"g6 retention {float(item['g6_retention']):.1%}"
        )
    if summary["raw_status"]:
        lines.extend(["", "## Unsupported or failed raw phases", ""])
        for item in summary["raw_status"]:
            lines.append(
                f"- `{item['file']}`: {item['status']}"
            )

    lines.extend(["", "## DualPath KV locality", ""])
    if summary["kv_locality"]:
        for name, payload in sorted(summary["kv_locality"].items()):
            if "median_aggregate_scaling" in payload:
                peer_ratios = [
                    float(item["peer_to_writer_ratio"])
                    for item in payload.get("comparisons", [])
                    if "peer_to_writer_ratio" in item
                ]
                peer_text = (
                    f", peer/writer {statistics.median(peer_ratios):.1%}"
                    if peer_ratios
                    else ""
                )
                lines.append(
                    f"- `{name}`: scaling "
                    f"{float(payload['median_aggregate_scaling']):.3f}x, retention "
                    f"{float(payload['median_per_node_retention']):.1%}{peer_text}"
                )
    else:
        lines.append("- Not run.")

    lines.extend(["", "## Decode-style KV writes", ""])
    if summary["kv_write"]:
        for item in summary["kv_write"]:
            buffered = item["buffered_bandwidth_gib_s"]
            durable = item["durable_bandwidth_gib_s"]
            text = (
                f"- `{item['phase']} / {item['sync_mode']}`: buffered "
                f"{float(buffered['median']):.3f} GiB/s"
            )
            if durable is not None:
                text += (
                    ", data+directory durable "
                    f"{float(durable['median']):.3f} GiB/s"
                )
            else:
                text += ", durability not requested"
            lines.append(text)
    else:
        lines.append("- Not run.")

    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- Direct/durable results estimate storage or NFS-path bandwidth.",
            "- Buffered-hot results estimate client page-cache bandwidth, not storage.",
            "- On g6 a cold phase only evicts the g6 client cache: unless the run "
            "set `MATRIX_DROP_PEER_CACHE=1`, the files stay in g5's NFS server "
            "cache, so g6 cold numbers describe the storage network path rather "
            "than the NVMe.",
            "- `file I/O` covers open/readinto/close; `scatter` is the "
            "page_first -> layer_first staging copy. They are timed separately "
            "and their shares are of the same wall-clock phase, so the two "
            "components overlap only through per-rank scheduling.",
            "- The IB link rate is not the storage ceiling here: IPoIB runs in "
            "datagram mode with a 2044-byte MTU and NFS uses proto=tcp, so the "
            "achievable rate is set by kernel TCP/IP over IPoIB.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result_root = args.run_root / "results"

    raw_repetitions, raw_combined = summarize_raw(result_root)
    raw_series = [item for item in raw_combined if "bandwidth_gib_s" in item]
    raw_comparisons = [item for item in raw_combined if "aggregate_scaling" in item]
    raw_status = [
        {"file": str(path.relative_to(result_root)), **json.loads(path.read_text())}
        for path in sorted((result_root / "raw").glob("*/*.status.json"))
    ]
    environment, telemetry = summarize_telemetry(result_root)
    telemetry_index = {
        (item["label"], item["host"]): item for item in telemetry
    }
    for item in raw_repetitions:
        telemetry_phase = (
            "mixed-g5-read-g6-write"
            if item["phase"] == "mixed"
            else item["phase"]
        )
        label = (
            f"raw.{item['case']}.{telemetry_phase}.{item['mode']}."
            f"r{item['repetition']}"
        )
        wire_bytes = 0
        per_host_wire_bytes = {}
        for host, direction in item["directions"].items():
            if host != "g6":
                continue
            delta = telemetry_index.get((label, host), {})
            interface = delta.get("interface", {})
            key = "rx_bytes" if direction == "read" else "tx_bytes"
            value = int(interface.get(key, 0))
            per_host_wire_bytes[host] = value
            wire_bytes += value
        item["network"] = {
            "per_host_directional_bytes": per_host_wire_bytes,
            "directional_bytes": wire_bytes,
            "wire_to_io_ratio": wire_bytes / int(item["bytes"]) if item["bytes"] else 0.0,
        }

    kv_locality = {}
    for path in sorted(result_root.glob("kv-*/summary.json")):
        kv_locality[path.parent.name] = json.loads(path.read_text())
    cache_matrix = {}
    for name in ("cache-matrix-g5", "cache-matrix-g6", "direct-g5", "direct-g6"):
        payload = load_optional_json(result_root / name / "summary.json")
        status = load_optional_json(result_root / f"{name}.status.json")
        if payload or status:
            cache_matrix[name] = payload or status

    summary = {
        "schema_version": 1,
        "run_root": str(args.run_root),
        "config": load_optional_json(result_root / "config.json"),
        "environment": environment,
        "raw_repetitions": raw_repetitions,
        "raw_series": raw_series,
        "raw_comparisons": raw_comparisons,
        "raw_status": raw_status,
        "kv_locality": kv_locality,
        "cache_matrix": cache_matrix,
        "theoretical": theoretical_limits(environment),
        "kv_write": summarize_writes(result_root),
        "telemetry_deltas": telemetry,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.report.write_text(markdown_report(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
