#!/usr/bin/env python3
"""Summarize one-node versus two-node KV page read results."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


RESULT_PREFIX = "KVIO_RESULT "


def load_results(path: Path) -> list[dict]:
    results = []
    for line in path.read_text().splitlines():
        marker = line.find(RESULT_PREFIX)
        if marker >= 0:
            results.append(json.loads(line[marker + len(RESULT_PREFIX) :]))
    if not results:
        raise RuntimeError(f"No {RESULT_PREFIX.strip()} records found in {path}")
    return results


def aggregate_phase(items: list[dict]) -> dict:
    total_bytes = sum(item["bytes"] for item in items)
    start_ns = min(item["wall_start_ns"] for item in items)
    end_ns = max(item["wall_end_ns"] for item in items)
    seconds = (end_ns - start_ns) / 1e9
    return {
        "nodes": len(items),
        "hosts": [item["host"] for item in items],
        "bytes": total_bytes,
        "seconds": seconds,
        "aggregate_gib_s": total_bytes / seconds / 2**30,
        "per_node_gib_s": {item["host"]: item["gib_s"] for item in items},
        "per_node_mean_gib_s": statistics.mean(item["gib_s"] for item in items),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--output")
    args = parser.parse_args()

    records = load_results(Path(args.log))
    by_phase: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        by_phase[(record["phase"], record["repetition"])].append(record)

    repetitions: dict[int, dict[str, dict]] = defaultdict(dict)
    for (phase, repetition), items in sorted(by_phase.items()):
        expected = 1 if phase == "single" else 2
        if len(items) != expected:
            raise RuntimeError(
                f"phase={phase} repetition={repetition} has {len(items)} nodes, expected {expected}"
            )
        repetitions[repetition][phase] = aggregate_phase(items)

    comparisons = []
    for repetition, phases in sorted(repetitions.items()):
        if set(phases) != {"single", "dual"}:
            raise RuntimeError(f"repetition={repetition} is missing single or dual phase")
        single = phases["single"]
        dual = phases["dual"]
        comparisons.append(
            {
                "repetition": repetition,
                "single_gib_s": single["aggregate_gib_s"],
                "dual_aggregate_gib_s": dual["aggregate_gib_s"],
                "dual_per_node_mean_gib_s": dual["per_node_mean_gib_s"],
                "aggregate_scaling": dual["aggregate_gib_s"] / single["aggregate_gib_s"],
                "per_node_retention": dual["per_node_mean_gib_s"] / single["aggregate_gib_s"],
                "dual_per_node_gib_s": dual["per_node_gib_s"],
            }
        )

    scaling = statistics.median(item["aggregate_scaling"] for item in comparisons)
    retention = statistics.median(item["per_node_retention"] for item in comparisons)
    drops_cache = {record["drop_cache"] for record in records}
    if drops_cache == {False}:
        diagnosis = (
            "cache-preserving run: cache residency changes between phases, so use the "
            "per-phase values to study locality rather than aggregate scaling"
        )
    elif scaling < 1.25 and retention < 0.70:
        diagnosis = (
            "strong shared-backend contention: two readers mostly divide one node's throughput"
        )
    elif scaling < 1.60 or retention < 0.80:
        diagnosis = "moderate contention: two-node aggregate bandwidth scales sublinearly"
    else:
        diagnosis = "no strong contention in this test: aggregate bandwidth scales reasonably"

    summary = {
        "comparisons": comparisons,
        "median_aggregate_scaling": scaling,
        "median_per_node_retention": retention,
        "drop_cache": sorted(drops_cache),
        "diagnosis": diagnosis,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n")


if __name__ == "__main__":
    main()
