#!/usr/bin/env python3
"""Plot per-turn latency metrics and a trajectory-JCT decomposition."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


JCT_COMPONENTS = (
    ("latency_seconds", "Model request", "#2563eb"),
    ("turn_client_runtime_seconds", "Client runtime", "#cbd5e1"),
)


def text(
    x: float, y: float, value: str, size: int = 12, anchor: str = "middle"
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#1f2937">'
        f"{html.escape(value)}</text>"
    )


def write_svg(path: Path, elements: list[str], width: int, height: int) -> None:
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )


def plot_lines(
    rows: list[dict[str, object]],
    series: tuple[tuple[str, str, str], ...],
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    width, height = 1280, 460
    left, right, top, bottom = 85.0, 1240.0, 60.0, 390.0
    turns = [int(row["turn_id"]) for row in rows]
    y_max = max(
        float(row[field]) for row in rows for field, _, _ in series
    ) * 1.12
    y_max = y_max or 1.0

    def x_pos(turn: int) -> float:
        return left + (turn - turns[0]) / max(1, turns[-1] - turns[0]) * (
            right - left
        )

    def y_pos(value: float) -> float:
        return bottom - value / y_max * (bottom - top)

    elements = [
        '<rect width="1280" height="460" fill="white"/>',
        text(640, 25, title, 16),
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = y_pos(value)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            'stroke="#e5e7eb"/>'
        )
        elements.append(text(left - 8, y + 4, f"{value * 1000:.1f}", 10, "end"))
    for index, (field, label, color) in enumerate(series):
        points = " ".join(
            f"{x_pos(int(row['turn_id'])):.1f},"
            f"{y_pos(float(row[field])):.1f}"
            for row in rows
        )
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2"/>'
        )
        x = 130 + index * 300
        elements.append(
            f'<line x1="{x}" y1="44" x2="{x + 24}" y2="44" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        elements.append(text(x + 31, 48, label, 11, "start"))
    tick_step = max(1, len(turns) // 10)
    for index in range(0, len(turns), tick_step):
        elements.append(text(x_pos(turns[index]), 410, str(turns[index]), 10))
    elements.extend(
        [
            text(640, 438, "Turn", 12),
            (
                '<text x="20" y="225" font-family="sans-serif" font-size="12" '
                'text-anchor="middle" transform="rotate(-90 20 225)">'
                f"{html.escape(ylabel)} (ms)</text>"
            ),
        ]
    )
    write_svg(output, elements, width, height)


def plot_jct_breakdown(
    rows: list[dict[str, object]], trajectory_jct: float, output: Path
) -> None:
    width, height = 1480, 560
    left, right, top, bottom = 90.0, 1120.0, 75.0, 465.0
    y_max = max(float(row["turn_wall_seconds"]) for row in rows) * 1.12 or 1.0
    spacing = (right - left) / len(rows)
    bar_width = max(2.0, min(18.0, spacing * 0.78))
    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(740, 25, "Per-turn Latency Breakdown", 17),
        text(
            740,
            48,
            f"Sum of all turn bars = trajectory JCT = {trajectory_jct:.3f} s",
            11,
        ),
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = bottom - value / y_max * (bottom - top)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            'stroke="#e5e7eb"/>'
        )
        elements.append(text(left - 8, y + 4, f"{value:.1f}", 10, "end"))
    for index, row in enumerate(rows):
        x = left + spacing * (index + 0.5)
        y = bottom
        for field, _, color in JCT_COMPONENTS:
            value = float(row[field])
            segment = value / y_max * (bottom - top)
            y -= segment
            elements.append(
                f'<rect x="{x - bar_width / 2:.1f}" y="{y:.1f}" '
                f'width="{bar_width:.1f}" height="{segment:.1f}" '
                f'fill="{color}"/>'
            )
    tick_step = max(1, len(rows) // 10)
    for index in range(0, len(rows), tick_step):
        x = left + spacing * (index + 0.5)
        elements.append(text(x, 485, str(rows[index]["turn_id"]), 10))
    legend_x = 1160.0
    for index, (_, label, color) in enumerate(JCT_COMPONENTS):
        y = 100 + index * 40
        elements.append(
            f'<rect x="{legend_x}" y="{y - 12}" width="16" height="16" '
            f'fill="{color}"/>'
        )
        elements.append(text(legend_x + 24, y, label, 11, "start"))
    elements.extend(
        [
            text((left + right) / 2, 520, "Turn", 12),
            (
                '<text x="22" y="270" font-family="sans-serif" font-size="12" '
                'text-anchor="middle" transform="rotate(-90 22 270)">'
                "Turn wall time (s)</text>"
            ),
        ]
    )
    write_svg(output, elements, width, height)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.rounds.read_text().splitlines()]
    trajectories = [
        json.loads(line) for line in args.trajectories.read_text().splitlines()
    ]
    if len(trajectories) != 1:
        raise RuntimeError("Per-turn JCT plot currently requires one trajectory")
    required = (
        "ttft_seconds",
        "ttst_seconds",
        "tpot_seconds",
        "turn_wall_seconds",
        *(field for field, _, _ in JCT_COMPONENTS),
    )
    for row in rows:
        missing = [field for field in required if field not in row]
        if missing:
            raise RuntimeError(f"turn {row.get('turn_id')} missing {missing}")
        component_sum = sum(float(row[field]) for field, _, _ in JCT_COMPONENTS)
        error = float(row["turn_wall_seconds"]) - component_sum
        if abs(error) > 1e-6:
            raise RuntimeError(
                f"turn {row['turn_id']} JCT breakdown error is {error}"
            )
    trajectory_jct = float(trajectories[0]["elapsed_seconds"])
    turn_sum = sum(float(row["turn_wall_seconds"]) for row in rows)
    if abs(trajectory_jct - turn_sum) > 1e-6:
        raise RuntimeError("turn wall times do not close to trajectory JCT")

    plot_lines(
        rows,
        (
            ("turn_wall_seconds", "Turn wall time", "#dc2626"),
            ("latency_seconds", "Model request", "#2563eb"),
        ),
        "Per-turn Wall and Request Latency",
        "Latency",
        args.output_dir / "trajectory_turn_latency.svg",
    )
    for field, name, color in (
        ("ttft_seconds", "TTFT", "#f97316"),
        ("ttst_seconds", "TTST", "#8b5cf6"),
        ("tpot_seconds", "TPOT", "#16a34a"),
    ):
        plot_lines(
            rows,
            ((field, name, color),),
            f"Per-turn {name}",
            name,
            args.output_dir / f"trajectory_{name.lower()}.svg",
        )
    plot_jct_breakdown(
        rows,
        trajectory_jct,
        args.output_dir / "trajectory_jct_breakdown.svg",
    )
    print(f"Wrote per-turn latency plots for {len(rows)} turns")


if __name__ == "__main__":
    main()
