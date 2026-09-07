#!/usr/bin/env python3
"""Plot all-trajectory latency distributions and mean TTFT breakdown for one run."""

from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


FULLCHAIN_COMPONENTS = (
    ("bootstrap_and_scheduling_seconds", "Bootstrap & scheduling", "#64748b"),
    ("l3_kv_read_seconds", "L3 KV read", "#f59e0b"),
    ("prefill_logits_sampling_seconds", "Prefill + sampling", "#22c55e"),
    ("pd_transfer_seconds", "P→D KV transfer", "#f97316"),
    ("decode_host_to_device_seconds", "Decode Host→GPU", "#ef4444"),
    ("decode_dispatch_seconds", "Decode dispatch", "#8b5cf6"),
    ("sse_egress_seconds", "SSE egress", "#06b6d4"),
    ("router_client_runtime_seconds", "Router/client gaps", "#cbd5e1"),
)


def text(
    x: float,
    y: float,
    value: str,
    size: int = 12,
    anchor: str = "middle",
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
        f'fill="#1f2937">{html.escape(value)}</text>'
    )


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    weight = position - left
    return ordered[left] * (1 - weight) + ordered[right] * weight


def stats(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "q1": quantile(ordered, 0.25),
        "median": quantile(ordered, 0.5),
        "q3": quantile(ordered, 0.75),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def agent_rows(run_dir: Path) -> list[dict[str, float | str]]:
    trajectories = [
        json.loads(line)
        for line in (run_dir / "trajectories.jsonl").read_text().splitlines()
    ]
    rounds = [
        json.loads(line)
        for line in (run_dir / "rounds.jsonl").read_text().splitlines()
    ]
    ttft: dict[str, list[float]] = defaultdict(list)
    tpot: dict[str, list[float]] = defaultdict(list)
    for row in rounds:
        agent = str(row["trajectory_id"])
        ttft[agent].append(float(row["ttft_seconds"]))
        if row.get("tpot_seconds") is not None:
            tpot[agent].append(float(row["tpot_seconds"]))
    output = []
    for trajectory in trajectories:
        agent = str(trajectory["trajectory_id"])
        output.append(
            {
                "trajectory_id": agent,
                "jct_seconds": float(trajectory["elapsed_seconds"]),
                "mean_ttft_seconds": statistics.fmean(ttft[agent]),
                "mean_tpot_seconds": statistics.fmean(tpot[agent]),
            }
        )
    return output


def draw_box_panel(
    elements: list[str],
    values: list[float],
    left: float,
    right: float,
    top: float,
    bottom: float,
    title: str,
    ylabel: str,
    scale: float,
) -> dict[str, float | int]:
    scaled = [value * scale for value in values]
    display_summary = stats(scaled)
    y_max = float(display_summary["max"]) * 1.18 or 1.0

    def y_pos(value: float) -> float:
        return bottom - value / y_max * (bottom - top)

    elements.append(text((left + right) / 2, top - 20, title, 13, weight="bold"))
    for tick in range(6):
        value = y_max * tick / 5
        y = y_pos(value)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            'stroke="#e5e7eb"/>'
        )
        elements.append(text(left - 8, y + 4, f"{value:.1f}", 10, "end"))
    x = (left + right) / 2
    box_width = min(90.0, (right - left) * 0.32)
    y_min = y_pos(float(display_summary["min"]))
    y_q1 = y_pos(float(display_summary["q1"]))
    y_med = y_pos(float(display_summary["median"]))
    y_q3 = y_pos(float(display_summary["q3"]))
    y_max_v = y_pos(float(display_summary["max"]))
    y_mean = y_pos(float(display_summary["mean"]))
    elements.extend(
        [
            f'<line x1="{x:.1f}" y1="{y_max_v:.1f}" x2="{x:.1f}" '
            f'y2="{y_min:.1f}" stroke="#111"/>',
            f'<line x1="{x - 12:.1f}" y1="{y_max_v:.1f}" '
            f'x2="{x + 12:.1f}" y2="{y_max_v:.1f}" stroke="#111"/>',
            f'<line x1="{x - 12:.1f}" y1="{y_min:.1f}" '
            f'x2="{x + 12:.1f}" y2="{y_min:.1f}" stroke="#111"/>',
            f'<rect x="{x - box_width / 2:.1f}" y="{y_q3:.1f}" '
            f'width="{box_width:.1f}" height="{max(1.0, y_q1 - y_q3):.1f}" '
            'fill="#fdba74" stroke="#9a3412"/>',
            f'<line x1="{x - box_width / 2:.1f}" y1="{y_med:.1f}" '
            f'x2="{x + box_width / 2:.1f}" y2="{y_med:.1f}" '
            'stroke="#9a3412" stroke-width="2"/>',
            f'<circle cx="{x:.1f}" cy="{y_mean:.1f}" r="4" fill="#1d4ed8"/>',
            text(x + box_width / 2 + 12, y_max_v + 4, f"max {display_summary['max']:.1f}", 10, "start"),
            text(x + box_width / 2 + 12, y_min + 4, f"min {display_summary['min']:.1f}", 10, "start"),
            text(x, bottom + 24, f"n={display_summary['count']}", 10),
        ]
    )
    elements.append(
        f'<text x="{left - 45:.1f}" y="{(top + bottom) / 2:.1f}" '
        'font-family="sans-serif" font-size="11" text-anchor="middle" '
        f'transform="rotate(-90 {left - 45:.1f} {(top + bottom) / 2:.1f})">'
        f"{html.escape(ylabel)}</text>"
    )
    return stats(values)


def write_latency(
    rows: list[dict[str, float | str]], output: Path
) -> dict[str, dict[str, float | int]]:
    width, height = 1280, 520
    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(width / 2, 28, "All trajectories · per-agent distributions", 16, weight="bold"),
        text(
            width / 2,
            48,
            "Whiskers=min/max, box=Q1–Q3, line=median, blue dot=mean",
            11,
        ),
    ]
    fields = (
        ("jct_seconds", "JCT", "Seconds", 1.0),
        ("mean_ttft_seconds", "Per-agent mean TTFT", "Milliseconds", 1000.0),
        ("mean_tpot_seconds", "Per-agent mean TPOT", "Milliseconds", 1000.0),
    )
    summaries = {}
    for index, (field, title, ylabel, scale) in enumerate(fields):
        left = 80.0 + index * 410
        summaries[field] = draw_box_panel(
            elements,
            [float(row[field]) for row in rows],
            left,
            left + 350,
            95,
            430,
            title,
            ylabel,
            scale,
        )
    output.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )
    return summaries


def write_breakdown(rounds_path: Path, output: Path) -> dict[str, float]:
    rows = [json.loads(line) for line in rounds_path.read_text().splitlines()]
    means = {
        field: statistics.fmean(float(row[field]) for row in rows)
        for field, _, _ in FULLCHAIN_COMPONENTS
    }
    means["mean_ttft_seconds"] = statistics.fmean(
        float(row["ttft_seconds"]) for row in rows
    )
    width, height = 980, 500
    left, right, top, bottom = 100.0, 500.0, 75.0, 420.0
    y_max = means["mean_ttft_seconds"] * 1000 * 1.15 or 1.0
    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(width / 2, 28, "Mean TTFT full-chain breakdown", 16, weight="bold"),
        text(width / 2, 48, f"Mean over all {len(rows)} requests", 11),
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = bottom - value / y_max * (bottom - top)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            'stroke="#e5e7eb"/>'
        )
        elements.append(text(left - 8, y + 4, f"{value:.0f}", 10, "end"))
    x, bar_width, cursor = 300.0, 120.0, bottom
    for field, _, color in FULLCHAIN_COMPONENTS:
        segment = means[field] * 1000 / y_max * (bottom - top)
        cursor -= segment
        elements.append(
            f'<rect x="{x - bar_width / 2:.1f}" y="{cursor:.1f}" '
            f'width="{bar_width:.1f}" height="{max(segment, 0.0):.1f}" '
            f'fill="{color}"/>'
        )
    elements.append(text(x, cursor - 10, f"{means['mean_ttft_seconds'] * 1000:.1f} ms", 11))
    for index, (_, label, color) in enumerate(FULLCHAIN_COMPONENTS):
        y = 100 + index * 38
        elements.append(f'<rect x="570" y="{y - 12}" width="16" height="16" fill="{color}"/>')
        elements.append(text(594, y + 1, label, 11, "start"))
    output.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )
    return means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fullchain-rounds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    rows = agent_rows(args.run_dir)
    with args.csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "latency": write_latency(rows, args.output_dir / "latency.svg"),
        "ttft_breakdown": write_breakdown(
            args.fullchain_rounds, args.output_dir / "ttft-breakdown.svg"
        ),
    }
    args.json_output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
