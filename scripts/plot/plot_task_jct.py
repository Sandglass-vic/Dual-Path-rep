#!/usr/bin/env python3
"""Compare worst-case trajectory JCT across datasets and agent counts."""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path


COLORS = ("#2563eb", "#f97316", "#16a34a", "#9333ea", "#0891b2", "#db2777")


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


def write_svg(data: dict[str, dict[int, dict[str, object]]], output: Path) -> None:
    width, height = 1120, 560
    left, right, top, bottom = 90.0, 850.0, 75.0, 470.0
    agent_counts = sorted(
        {agents for datasets in data.values() for agents in datasets}
    )
    y_max = (
        max(
            float(point["max_jct_seconds"])
            for datasets in data.values()
            for point in datasets.values()
        )
        * 1.15
        or 1.0
    )

    def x_pos(agents: int) -> float:
        index = agent_counts.index(agents)
        return left + (index + 0.5) / len(agent_counts) * (right - left)

    def y_pos(value: float) -> float:
        return bottom - value / y_max * (bottom - top)

    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(width / 2, 28, "Worst-case trajectory JCT by dataset and agents", 16),
        text(
            width / 2,
            50,
            "Each point is max(agent JCT) within a rep, then max across repetitions",
            11,
        ),
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = y_pos(value)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            'stroke="#e5e7eb"/>'
        )
        elements.append(text(left - 8, y + 4, f"{value:.1f}", 10, "end"))
    for agents in agent_counts:
        elements.append(text(x_pos(agents), bottom + 24, str(agents), 11))
    for index, (dataset, points) in enumerate(sorted(data.items())):
        color = COLORS[index % len(COLORS)]
        ordered = sorted(points.items())
        polyline = " ".join(
            f"{x_pos(agents):.1f},{y_pos(float(point['max_jct_seconds'])):.1f}"
            for agents, point in ordered
        )
        elements.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" '
            'stroke-width="2.5"/>'
        )
        for agents, point in ordered:
            x = x_pos(agents)
            y = y_pos(float(point["max_jct_seconds"]))
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        legend_y = 105 + index * 34
        elements.append(
            f'<line x1="900" y1="{legend_y}" x2="925" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        elements.append(text(935, legend_y + 4, dataset, 11, "start"))
    elements.extend(
        [
            text((left + right) / 2, 525, "#Agents", 12),
            (
                '<text x="24" y="275" font-family="sans-serif" font-size="12" '
                'text-anchor="middle" transform="rotate(-90 24 275)">'
                "Max JCT (seconds)</text>"
            ),
        ]
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
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = discover(args.result_dir)
    write_svg(data, args.output_dir / "JCT.svg")
    (args.output_dir / "JCT.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
