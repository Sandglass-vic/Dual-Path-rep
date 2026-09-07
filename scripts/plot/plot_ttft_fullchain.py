#!/usr/bin/env python3
"""Render a dependency-free SVG for the minimal full-chain TTFT run."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


COMPONENTS = (
    ("bootstrap_and_scheduling_seconds", "Prefill Bootstrap & Scheduling", "#64748b"),
    ("l3_kv_read_seconds", "L3 KV Read (exclusive)", "#f59e0b"),
    (
        "prefill_logits_sampling_seconds",
        "Prefill Forward + Logits + Sampling",
        "#22c55e",
    ),
    ("pd_transfer_seconds", "Prefill-to-Decode KV Transfer", "#f97316"),
    ("decode_host_to_device_seconds", "Decode Host-to-GPU Copy", "#ef4444"),
    ("decode_dispatch_seconds", "Decode Prebuilt-token Dispatch", "#8b5cf6"),
    ("sse_egress_seconds", "SSE Serialization & Egress", "#06b6d4"),
    ("router_client_runtime_seconds", "Router, Client & Runtime Gaps", "#cbd5e1"),
)


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
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rounds.read_text().splitlines()]
    if not rows:
        raise RuntimeError("No full-chain rounds to plot")

    width, height = 1280, 520
    plot_left, plot_right = 90.0, 820.0
    plot_top, plot_bottom = 70.0, 440.0
    max_ttft = max(float(row["ttft_seconds"]) for row in rows) * 1.12
    elements = [
        '<rect width="1280" height="520" fill="white"/>',
        text(640, 26, "Per-turn TTFT Critical-path Breakdown", 17),
        text(
            640,
            48,
            "First token is sampled on Prefill; Decode dispatches the prebuilt token",
            11,
        ),
    ]
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
        for field, _, color in COMPONENTS:
            value = float(row[field])
            segment_height = value / max_ttft * (plot_bottom - plot_top)
            bottom -= segment_height
            elements.append(
                f'<rect x="{x - bar_width / 2:.1f}" y="{bottom:.1f}" '
                f'width="{bar_width:.1f}" height="{segment_height:.1f}" '
                f'fill="{color}"/>'
            )
        if index % label_step == 0:
            elements.append(text(x, 461, f"turn {row['turn_id']}", 10))
        if len(rows) <= 16:
            elements.append(
                text(
                    x,
                    bottom - 7,
                    f"{float(row['ttft_seconds']) * 1000:.1f} ms",
                    10,
                )
            )

    legend_x, legend_y = 875.0, 92.0
    for index, (_, label, color) in enumerate(COMPONENTS):
        y = legend_y + index * 35
        elements.append(
            f'<rect x="{legend_x}" y="{y - 12:.1f}" width="16" height="16" '
            f'fill="{color}"/>'
        )
        elements.append(text(legend_x + 25, y + 1, label, 11, "start"))
    elements.append(
        '<text x="22" y="260" font-family="sans-serif" font-size="12" '
        'text-anchor="middle" transform="rotate(-90 22 260)">TTFT (ms)</text>'
    )
    elements.append(
        text(
            640,
            502,
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
