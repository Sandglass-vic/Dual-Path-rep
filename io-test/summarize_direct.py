#!/usr/bin/env python3
"""Summarize buffered versus O_DIRECT KV read mini-benchmark results."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


RESULT_PREFIX = "KVIO_RESULT "
CACHE_PROBE_PREFIX = "KVIO_CACHE_PROBE "


def records_with_prefix(path: Path, prefix: str) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        marker = line.find(prefix)
        if marker >= 0:
            records.append(json.loads(line[marker + len(prefix) :]))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = records_with_prefix(args.log, RESULT_PREFIX)
    probes = records_with_prefix(args.log, CACHE_PROBE_PREFIX)
    if not records:
        raise RuntimeError(f"No benchmark records found in {args.log}")
    if len(probes) != 1:
        raise RuntimeError(f"Expected one cache probe, found {len(probes)}")

    by_mode: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_mode[str(record["io_mode"])].append(float(record["gib_s"]))
    if set(by_mode) != {"buffered", "direct"}:
        raise RuntimeError(f"Expected buffered and direct records, found {sorted(by_mode)}")

    probe = probes[0]
    direct_bypassed_page_cache = (
        probe["resident_after_direct"] <= probe["resident_before"]
        and probe["resident_after_buffered"] > probe["resident_before_buffered"]
    )
    summary = {
        "cache_probe": probe,
        "direct_bypassed_page_cache": direct_bypassed_page_cache,
        "bandwidth_gib_s": {
            mode: {
                "values": values,
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
            for mode, values in sorted(by_mode.items())
        },
        "direct_to_buffered_median_ratio": (
            statistics.median(by_mode["direct"])
            / statistics.median(by_mode["buffered"])
        ),
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    args.output.write_text(rendered + "\n")
    if not direct_bypassed_page_cache:
        raise RuntimeError("cache-residency probe did not confirm page-cache bypass")


if __name__ == "__main__":
    main()
