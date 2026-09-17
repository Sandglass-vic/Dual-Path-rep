#!/usr/bin/env python3
"""Summarize the buffered-hot versus O_DIRECT-over-hot-cache experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


RESULT_PREFIX = "KVIO_RESULT "


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for line in args.log.read_text().splitlines():
        marker = line.find(RESULT_PREFIX)
        if marker >= 0:
            records.append(json.loads(line[marker + len(RESULT_PREFIX) :]))
    by_phase: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_phase[str(record["phase"])].append(float(record["gib_s"]))

    expected = {
        "buffered-populate",
        "buffered-hot",
        "direct-over-hot-cache",
        "buffered-hot-after-direct",
    }
    if set(by_phase) != expected:
        raise RuntimeError(f"Expected {sorted(expected)}, found {sorted(by_phase)}")

    phases = {
        phase: {
            "values": values,
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        for phase, values in by_phase.items()
    }
    hot = phases["buffered-hot"]["median"]
    direct = phases["direct-over-hot-cache"]["median"]
    populated = phases["buffered-populate"]["median"]
    summary = {
        "bandwidth_gib_s": phases,
        "hot_buffered_to_populate_ratio": hot / populated,
        "direct_over_hot_to_hot_buffered_ratio": direct / hot,
        "direct_over_hot_to_populate_ratio": direct / populated,
        "interpretation": (
            "O_DIRECT did not consume the hot client page cache"
            if direct < hot * 0.8
            else "O_DIRECT was not clearly separated from hot buffered bandwidth"
        ),
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
