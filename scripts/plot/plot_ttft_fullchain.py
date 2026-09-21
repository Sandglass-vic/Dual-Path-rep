#!/usr/bin/env python3
"""Render a dependency-free SVG for the minimal full-chain TTFT run."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from plot_style import add_title_suffix
from ttft_components import components_for_rows, normalize_ttft_row


def text(x: float, y: float, value: str, size: int = 12, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#1f2937">'
        f"{html.escape(value)}</text>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title-suffix", default="")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rounds.read_text().splitlines()]
    if not rows:
        raise RuntimeError("No full-chain rounds to plot")
    for row in rows:
        normalize_ttft_row(row)
        if int(row.get("ttft_timing_schema_version", 0)) < 3:
            row["misc_seconds"] = sum(
                float(row.get(field, 0.0))
                for field in (
                    "bootstrap_and_scheduling_seconds",
                    "decode_l3_bootstrap_handoff_seconds",
                    "sse_egress_seconds",
                    "server_runtime_residual_seconds",
                    "router_client_runtime_seconds",
                    "kv_read_overhead_seconds",
                )
            )
    components = components_for_rows(rows)

    width, height = 600, 470
    plot_left, plot_right = 75.0, 540.0
    plot_top, plot_bottom = 145.0, 390.0
    max_ttft = max(float(row["ttft_seconds"]) for row in rows) * 1.12
    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(width / 2, 24, add_title_suffix("TTFT Breakdown", args.title_suffix), 15),
        text(
            width / 2,
            43,
            "First token is sampled on Prefill; Decode dispatches the prebuilt token",
            11,
        ),
    ]
    # Keep the original component order and render the legend inside the
    # figure, above the chart, as two horizontal rows.
    legend_columns = 3
    legend_x_positions = (40, 280, 450)
    for index, (_, label, color) in enumerate(components):
        column = index % legend_columns
        row = index // legend_columns
        legend_x = legend_x_positions[column]
        legend_y = 54 + row * 21
        elements.append(
            f'<rect x="{legend_x}" y="{legend_y - 8:.1f}" width="10" height="10" '
            f'fill="{color}"/>'
        )
        elements.append(text(legend_x + 14, legend_y + 1, label, 9, "start"))
    for tick in range(6):
        value = max_ttft * tick / 5
        y = plot_bottom - (plot_bottom - plot_top) * tick / 5
        elements.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" '
            f'y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        elements.append(text(plot_left - 10, y + 4, f"{value * 1000:.0f}", 10, "end"))

    bar_width = min(110.0, 500.0 / len(rows))
    spacing = (plot_right - plot_left) / len(rows)
    label_step = max(1, len(rows) // 10)
    for index, row in enumerate(rows):
        x = plot_left + spacing * (index + 0.5)
        bottom = plot_bottom
        for field, _, color in components:
            value = float(row[field])
            segment_height = value / max_ttft * (plot_bottom - plot_top)
            bottom -= segment_height
            elements.append(
                f'<rect x="{x - bar_width / 2:.1f}" y="{bottom:.1f}" '
                f'width="{bar_width:.1f}" height="{segment_height:.1f}" '
                f'fill="{color}"/>'
            )
        if index % label_step == 0:
            elements.append(text(x, 378, f"turn {row['turn_id']}", 9))
        if len(rows) <= 16:
            elements.append(
                text(
                    x,
                    bottom - 7,
                    f"{float(row['ttft_seconds']) * 1000:.1f} ms",
                    10,
                )
            )

    elements.append(
        '<text x="20" y="220" font-family="sans-serif" font-size="11" '
        'text-anchor="middle" transform="rotate(-90 20 220)">Time [ms]</text>'
    )
    elements.append(
        text(
            width / 2,
            425,
            "Components are additive; cross-node overlap is reported separately in JSON.",
            10,
        )
    )
    document = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        *elements,
        "</svg>",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(document) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
