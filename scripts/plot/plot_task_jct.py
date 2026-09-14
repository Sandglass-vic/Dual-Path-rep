#!/usr/bin/env python3
"""Compare worst-case trajectory JCT across datasets and agent counts."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path

from plot_style import add_title_suffix


def text(
    x: float, y: float, value: str, size: int = 12, anchor: str = "middle"
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#1f2937">'
        f"{html.escape(value)}</text>"
    )


def discover(result_dir: Path) -> dict[str, dict[int, dict[str, object]]]:
    repetitions: dict[tuple[str, int], list[dict[str, float | int]]] = defaultdict(list)
    for path in sorted(result_dir.glob("*/agents-*/rep-*/trajectories.jsonl")):
        run_dir = path.parent
        metadata = json.loads((run_dir / "run.json").read_text())
        dataset = str(
            metadata.get("dataset", f"{int(metadata['max_len_k'])}K")
        )
        agents = int(metadata["agents"])
        repetition = int(metadata["repetition"])
        trajectories = [
            json.loads(line) for line in path.read_text().splitlines()
        ]
        repetitions[(dataset, agents)].append(
            {
                "repetition": repetition,
                "max_jct_seconds": max(
                    float(row["elapsed_seconds"]) for row in trajectories
                ),
            }
        )
    if not repetitions:
        raise RuntimeError(f"No completed runs under {result_dir}")
    output: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for (dataset, agents), reps in repetitions.items():
        output[dataset][agents] = {
            "max_jct_seconds": max(float(row["max_jct_seconds"]) for row in reps),
            "repetitions": sorted(reps, key=lambda row: int(row["repetition"])),
        }
    return dict(output)


def write_svg(data: dict[str, dict[int, dict[str, object]]], output: Path, title_suffix: str) -> None:
    datasets = sorted(data.items())
    columns = min(3, len(datasets))
    rows = math.ceil(len(datasets) / columns)
    panel_width, panel_height = 330, 275
    width = 55 + columns * panel_width
    height = 80 + rows * panel_height
    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(width / 2, 22, add_title_suffix("Worst-case trajectory JCT", title_suffix), 14),
        text(
            width / 2,
            41,
            "For each dataset × agents: max agent JCT, then max across repetitions",
            11,
        ),
    ]
    for index, (dataset, points) in enumerate(datasets):
        row, column = divmod(index, columns)
        panel_x = 55 + column * panel_width
        panel_y = 65 + row * panel_height
        left, right = panel_x + 50, panel_x + panel_width - 18
        top, bottom = panel_y + 35, panel_y + panel_height - 55
        ordered = sorted(points.items())
        y_max = (
            max(float(point["max_jct_seconds"]) for _, point in ordered) * 1.18
            or 1.0
        )

        def y_pos(value: float) -> float:
            return bottom - value / y_max * (bottom - top)

        elements.append(
            text((left + right) / 2, panel_y + 18, f"Dataset = {dataset}", 14)
        )
        for tick in range(5):
            value = y_max * tick / 4
            y = y_pos(value)
            elements.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
                'stroke="#e5e7eb"/>'
            )
            elements.append(text(left - 7, y + 4, f"{value:.0f}", 9, "end"))
        spacing = (right - left) / len(ordered)
        bar_width = min(62.0, spacing * 0.58)
        for bar_index, (agents, point) in enumerate(ordered):
            value = float(point["max_jct_seconds"])
            x = left + spacing * (bar_index + 0.5)
            y = y_pos(value)
            elements.append(
                f'<rect x="{x - bar_width / 2:.1f}" y="{y:.1f}" '
                f'width="{bar_width:.1f}" height="{bottom - y:.1f}" '
                'fill="#f6c77c" stroke="#b7791f"/>'
            )
            elements.append(text(x, y - 7, f"{value:.1f}", 9))
            elements.append(text(x, bottom + 18, str(agents), 10))
        elements.append(text((left + right) / 2, bottom + 39, "#Agents", 11))
        axis_x = panel_x + 13
        axis_y = (top + bottom) / 2
        elements.append(
            f'<text x="{axis_x:.1f}" y="{axis_y:.1f}" '
            'font-family="sans-serif" font-size="11" text-anchor="middle" '
            f'transform="rotate(-90 {axis_x:.1f} {axis_y:.1f})">'
            "Max JCT (s)</text>"
        )
    output.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title-suffix", default="")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = discover(args.result_dir)
    write_svg(data, args.output_dir / "JCT.svg", args.title_suffix)
    (args.output_dir / "JCT.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
