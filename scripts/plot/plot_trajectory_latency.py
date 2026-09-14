#!/usr/bin/env python3
"""Plot latency metrics for the longest trajectory in one benchmark run."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from plot_style import add_title_suffix


TURN_COMPONENTS = (
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
    title_suffix: str = "",
) -> None:
    width, height = 1100, 400
    left, right, top, bottom = 70.0, 1060.0, 52.0, 335.0
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
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(width / 2, 22, add_title_suffix(title, title_suffix), 14),
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
        x = 100 + index * 250
        elements.append(
            f'<line x1="{x}" y1="38" x2="{x + 24}" y2="38" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        elements.append(text(x + 31, 42, label, 10, "start"))
    tick_step = max(1, len(turns) // 10)
    for index in range(0, len(turns), tick_step):
        elements.append(text(x_pos(turns[index]), 352, str(turns[index]), 9))
    elements.extend(
        [
            text(width / 2, 378, "Turn", 11),
            (
                '<text x="16" y="195" font-family="sans-serif" font-size="11" '
                'text-anchor="middle" transform="rotate(-90 16 195)">'
                f"{html.escape(ylabel)} (ms)</text>"
            ),
        ]
    )
    write_svg(output, elements, width, height)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title-suffix", default="")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.rounds.read_text().splitlines()]
    required = (
        "ttft_seconds",
        "ttst_seconds",
        "tpot_seconds",
        "turn_wall_seconds",
        *(field for field, _, _ in TURN_COMPONENTS),
    )
    for row in rows:
        missing = [field for field in required if field not in row]
        if missing:
            raise RuntimeError(f"turn {row.get('turn_id')} missing {missing}")
        component_sum = sum(float(row[field]) for field, _, _ in TURN_COMPONENTS)
        error = float(row["turn_wall_seconds"]) - component_sum
        if abs(error) > 1e-6:
            raise RuntimeError(
                f"turn {row['turn_id']} trajectory_jct component error is {error}"
            )
    plot_lines(
        rows,
        (
            ("turn_wall_seconds", "Turn wall time", "#dc2626"),
            ("latency_seconds", "Model request", "#2563eb"),
        ),
        "Per-turn Wall and Request Latency",
        "Latency",
        args.output_dir / "turn.svg",
        args.title_suffix,
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
            args.output_dir / f"{name.lower()}.svg",
            args.title_suffix,
        )
    print(f"Wrote per-turn latency plots for {len(rows)} turns")


if __name__ == "__main__":
    main()
