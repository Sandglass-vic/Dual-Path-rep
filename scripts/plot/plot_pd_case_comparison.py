#!/usr/bin/env python3
"""Plot TTFT breakdown and worst-agent JCT for several completed PD cases."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from plot_style import COLORS
from ttft_components import components_for_path, normalize_ttft_row
CASE_COLORS = (COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"])


def svg_text(x: float, y: float, value: str, size: int = 12, anchor: str = "middle", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
        f'fill="#1f2937">{html.escape(value)}</text>'
    )


def find_file(task_dir: Path, name: str) -> Path:
    candidates = sorted(task_dir.glob(f"**/{name}"))
    if not candidates:
        raise FileNotFoundError(f"Cannot find {name} under {task_dir}")
    preferred = [path for path in candidates if "longest-trajectory" in str(path)]
    return preferred[0] if preferred else candidates[0]


def find_fullchain_rounds(task_dir: Path) -> Path:
    candidates = sorted(task_dir.glob("**/data/fullchain/fullchain_rounds.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"Cannot find fullchain_rounds.jsonl under {task_dir}")
    return candidates[0]


def find_all_trajectories(task_dir: Path) -> Path:
    candidates = sorted(task_dir.glob("**/data/trajectories.jsonl"))
    if not candidates:
        return find_file(task_dir, "trajectories.jsonl")
    return candidates[0]


def find_run_metadata(path: Path) -> dict[str, object]:
    for parent in (path.parent, *path.parents):
        candidate = parent / "run.json"
        if candidate.is_file():
            return json.loads(candidate.read_text())
    return {}


def infer_path(metadata: dict[str, object], task_dir: Path) -> str:
    for key in ("l3_read_path", "dualpath_l3_read_path", "l3_read"):
        value = metadata.get(key)
        if value:
            value = str(value).lower()
            if "decode" in value:
                return "Decode-Path"
            if "prefill" in value:
                return "Prefill-Path"
    text = " ".join(path.name.lower() for path in task_dir.glob("**/run.json"))
    return "Decode-Path" if "decode" in text else "Prefill-Path"


def topology(task: str) -> str:
    match = re.search(r"(\d+p\d+d)", task.lower())
    return match.group(1).upper() if match else task


def load_case(code_home: Path, job_id: str, task: str) -> dict[str, object]:
    task_dir = code_home / "results" / "latency-benchmark" / task
    rounds_path = find_fullchain_rounds(task_dir)
    trajectories_path = find_all_trajectories(task_dir)
    metadata = find_run_metadata(rounds_path)
    rounds = [json.loads(line) for line in rounds_path.read_text().splitlines() if line.strip()]
    trajectories = [
        json.loads(line) for line in trajectories_path.read_text().splitlines() if line.strip()
    ]
    if not rounds or not trajectories:
        raise RuntimeError(f"Incomplete case data under {task_dir}")

    path = infer_path(metadata, task_dir)
    for row in rounds:
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
    components = components_for_path(path)
    values = {
        field: sum(float(row.get(field, 0.0)) for row in rounds) / len(rounds)
        for field, _, _ in components
    }
    ttft = sum(float(row.get("ttft_seconds", 0.0)) for row in rounds) / len(rounds)
    jct = max(float(row.get("elapsed_seconds", 0.0)) for row in trajectories)
    label = f"{topology(task)}-{path}"
    return {
        "job_id": job_id,
        "task": task,
        "label": label,
        "path": path,
        "ttft_seconds": ttft,
        "jct_seconds": jct,
        "components": values,
        "component_fields": [field for field, _, _ in components],
        "rounds": len(rounds),
        "trajectories": len(trajectories),
    }


def write_svg(cases: list[dict[str, object]], output: Path) -> None:
    width, height = 980, 720
    left, right = 105.0, 925.0
    elements = [f'<rect width="{width}" height="{height}" fill="white"/>']
    elements.append(svg_text(width / 2, 28, "TTFT Breakdown and JCT comparison", 17, weight="bold"))

    legend_columns = 3
    legend_x_positions = (35.0, 350.0, 680.0)
    legend_components = []
    seen_fields = set()
    for case in cases:
        for component in components_for_path(str(case["path"])):
            if component[0] not in seen_fields:
                seen_fields.add(component[0])
                legend_components.append(component)
    for index, (_, label, color) in enumerate(legend_components):
        column = index % legend_columns
        row = index // legend_columns
        x = legend_x_positions[column]
        y = 50.0 + row * 20.0
        elements.append(f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="11" height="11" fill="{color}"/>')
        elements.append(svg_text(x + 16, y, label, 9, "start"))

    panels = ((175.0, 400.0, "Mean TTFT Breakdown", "Time [ms]"), (460.0, 685.0, "Worst-agent JCT", "JCT [s]"))
    max_ttft = max(float(case["ttft_seconds"]) for case in cases) * 1000.0 * 1.18 or 1.0
    max_jct = max(float(case["jct_seconds"]) for case in cases) * 1.18 or 1.0
    bar_width = min(115.0, (right - left) / len(cases) * 0.52)
    spacing = (right - left) / len(cases)

    for top, bottom, title, ylabel in panels:
        max_value = max_ttft if "TTFT" in title else max_jct
        elements.append(svg_text(width / 2, top - 25, title, 14, weight="bold"))
        for tick in range(6):
            value = max_value * tick / 5
            y = bottom - (bottom - top) * tick / 5
            elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
            elements.append(svg_text(left - 10, y + 4, f"{value:.0f}", 10, "end"))
        elements.append(
            f'<text x="25" y="{(top + bottom) / 2:.1f}" font-family="sans-serif" '
            f'font-size="11" text-anchor="middle" transform="rotate(-90 25 {(top + bottom) / 2:.1f})">'
            f"{html.escape(ylabel)}</text>"
        )
        for index, case in enumerate(cases):
            x = left + spacing * (index + 0.5)
            if "TTFT" in title:
                cursor = bottom
                for field, _, color in components_for_path(str(case["path"])):
                    segment = float(case["components"][field]) * 1000.0 / max_value * (bottom - top)
                    cursor -= segment
                    elements.append(f'<rect x="{x - bar_width / 2:.1f}" y="{cursor:.1f}" width="{bar_width:.1f}" height="{max(segment, 0.0):.1f}" fill="{color}"/>')
                elements.append(svg_text(x, cursor - 8, f"{float(case['ttft_seconds']) * 1000:.1f}", 10))
            else:
                value = float(case["jct_seconds"])
                y = bottom - value / max_value * (bottom - top)
                elements.append(f'<rect x="{x - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bottom - y:.1f}" fill="#94a3b8"/>')
                elements.append(svg_text(x, y - 8, f"{value:.1f}", 10))
            if "JCT" in title:
                elements.append(svg_text(x, bottom + 19, str(case["label"]), 13, weight="bold"))
    output.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-home", type=Path, default=Path("/labhome/sezhou/dual-path"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case", action="append", nargs=2, metavar=("JOB_ID", "TASK_NAME"), required=True)
    args = parser.parse_args()
    if len(args.case) != 4:
        parser.error("exactly four --case JOB_ID TASK_NAME arguments are required")
    cases = [load_case(args.code_home, job_id, task) for job_id, task in args.case]
    output_dir = args.output_dir or args.code_home / "results" / "pd-case-comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_svg(cases, output_dir / "pd-case-comparison.svg")
    (output_dir / "pd-case-comparison.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Wrote {output_dir / 'pd-case-comparison.svg'}")


if __name__ == "__main__":
    main()
