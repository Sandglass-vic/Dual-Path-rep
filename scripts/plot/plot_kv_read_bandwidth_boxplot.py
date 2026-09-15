#!/usr/bin/env python3
"""Draw a compact SVG box plot of request-level Lustre KV read bandwidth."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_bandwidth(path: Path) -> tuple[list[float], dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"No rounds found in {path}")
    read_path = str(rows[0]["l3_read_path"])
    side = "decode" if read_path == "decode" else "prefill"
    bytes_key = f"{side}_l3_read_bytes"
    seconds_key = f"{side}_l3_read_seconds"
    values = [
        float(row[bytes_key]) / float(row[seconds_key]) / 1e9
        for row in rows
        if float(row.get(bytes_key, 0)) > 0 and float(row.get(seconds_key, 0)) > 0
    ]
    if not values:
        raise RuntimeError(f"No valid {side} KV reads found in {path}")
    return values, rows[0]


def box_statistics(values: list[float]) -> dict:
    q1 = percentile(values, 0.25)
    median = percentile(values, 0.50)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    low_limit = q1 - 1.5 * iqr
    high_limit = q3 + 1.5 * iqr
    inside = [value for value in values if low_limit <= value <= high_limit]
    return {
        "count": len(values),
        "min": min(values),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": max(values),
        "whisker_low": min(inside),
        "whisker_high": max(inside),
        "mean": sum(values) / len(values),
        "outliers": [value for value in values if value < low_limit or value > high_limit],
    }


def text(x: float, y: float, value: str, size: int = 11, anchor: str = "middle", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="#202124">{html.escape(value)}</text>'
    )


def draw(cases: list[dict], output: Path) -> None:
    width = height = 600
    left, right, top, bottom = 76.0, 580.0, 58.0, 505.0
    plot_width, plot_height = right - left, bottom - top
    observed_max = max(case["stats"]["max"] for case in cases)
    y_max = max(5.0, math.ceil(observed_max / 5.0) * 5.0)

    elements = [f'<rect width="{width}" height="{height}" fill="white"/>']
    elements.append(text(width / 2, 28, "KV Read Bandwidth Distribution - 32K - 8 Agents", 16, weight="bold"))

    for tick in range(0, int(y_max) + 1, 5):
        y = bottom - tick / y_max * plot_height
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            f'stroke="#e5e7eb" stroke-width="1"/>'
        )
        elements.append(text(left - 10, y + 4, str(tick), 11, "end"))
    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#374151"/>')
    elements.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#374151"/>')
    elements.append(
        f'<text x="20" y="{(top + bottom) / 2:.1f}" font-family="Arial, sans-serif" '
        f'font-size="13" text-anchor="middle" fill="#202124" '
        f'transform="rotate(-90 20 {(top + bottom) / 2:.1f})">Bandwidth [GB/s]</text>'
    )

    spacing = plot_width / len(cases)
    box_width = 58.0
    cap_width = 32.0

    def y_pos(value: float) -> float:
        return bottom - value / y_max * plot_height

    for index, case in enumerate(cases):
        center = left + spacing * (index + 0.5)
        stats = case["stats"]
        color = case["color"]
        q1_y, median_y, q3_y = map(y_pos, (stats["q1"], stats["median"], stats["q3"]))
        low_y, high_y = map(y_pos, (stats["whisker_low"], stats["whisker_high"]))

        elements.append(f'<line x1="{center:.1f}" y1="{high_y:.1f}" x2="{center:.1f}" y2="{q3_y:.1f}" stroke="#374151" stroke-width="1.5"/>')
        elements.append(f'<line x1="{center:.1f}" y1="{q1_y:.1f}" x2="{center:.1f}" y2="{low_y:.1f}" stroke="#374151" stroke-width="1.5"/>')
        elements.append(f'<line x1="{center-cap_width/2:.1f}" y1="{high_y:.1f}" x2="{center+cap_width/2:.1f}" y2="{high_y:.1f}" stroke="#374151" stroke-width="1.5"/>')
        elements.append(f'<line x1="{center-cap_width/2:.1f}" y1="{low_y:.1f}" x2="{center+cap_width/2:.1f}" y2="{low_y:.1f}" stroke="#374151" stroke-width="1.5"/>')
        elements.append(
            f'<rect x="{center-box_width/2:.1f}" y="{q3_y:.1f}" width="{box_width:.1f}" '
            f'height="{q1_y-q3_y:.1f}" fill="{color}" fill-opacity="0.72" '
            f'stroke="{color}" stroke-width="1.5"/>'
        )
        elements.append(f'<line x1="{center-box_width/2:.1f}" y1="{median_y:.1f}" x2="{center+box_width/2:.1f}" y2="{median_y:.1f}" stroke="#111827" stroke-width="2.2"/>')
        for outlier_index, value in enumerate(stats["outliers"]):
            jitter = ((outlier_index % 5) - 2) * 2.2
            elements.append(
                f'<circle cx="{center+jitter:.1f}" cy="{y_pos(value):.1f}" r="2.1" '
                f'fill="none" stroke="{color}" stroke-width="1"/>'
            )
        topology, path_name = case["label"].split("-", 1)
        elements.append(text(center, bottom + 23, topology, 11, weight="bold"))
        elements.append(text(center, bottom + 39, path_name, 10))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n' + "\n".join(elements) + "\n</svg>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", nargs=2, metavar=("LABEL", "ROUNDS"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    cases = []
    for index, (label, path_string) in enumerate(args.case):
        values, metadata = load_bandwidth(Path(path_string))
        cases.append(
            {
                "label": label,
                "path": path_string,
                "task_name": metadata.get("task_name"),
                "allocation_job_id": metadata.get("allocation_job_id"),
                "read_path": metadata.get("l3_read_path"),
                "color": COLORS[index % len(COLORS)],
                "stats": box_statistics(values),
            }
        )
    draw(cases, args.output)
    summary = args.summary or args.output.with_suffix(".json")
    summary.write_text(json.dumps(cases, indent=2) + "\n")
    print(f"KV_READ_BANDWIDTH_BOXPLOT_COMPLETE output={args.output} summary={summary}")


if __name__ == "__main__":
    main()
