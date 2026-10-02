#!/usr/bin/env python3
"""Compare TTFT breakdown, JCT, L3 read bandwidth and TPOT across PD cases."""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
from pathlib import Path

from plot_style import COLORS
from ttft_components import components_for_rows, normalize_ttft_row
CASE_COLORS = (COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"])
KV_BYTES_PER_TOKEN = 64 * 8 * 128 * 2 * 2  # layers * KV heads * head dim * K/V * BF16


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


def find_summary(path: Path) -> dict[str, object]:
    for parent in (path.parent, *path.parents):
        candidate = parent / "summary.json"
        if candidate.is_file():
            return json.loads(candidate.read_text())
    return {}


def infer_path(metadata: dict[str, object], task_dir: Path) -> str:
    for key in ("l3_read_path", "dualpath_l3_read_path", "l3_read"):
        value = metadata.get(key)
        if value:
            value = str(value).lower()
            if "dynamic" in value:
                return "Dynamic-Path"
            if "decode" in value:
                return "Decode-Path"
            if "prefill" in value:
                return "Prefill-Path"
    text = " ".join(path.name.lower() for path in task_dir.glob("**/run.json"))
    if "dynamic" in text:
        return "Dynamic-Path"
    return "Decode-Path" if "decode" in text else "Prefill-Path"


def read_bandwidth(rounds: list[dict[str, object]]) -> dict[str, float]:
    """Median per-request GB/s on each side, plus the two-sided aggregate.

    A dynamic run resolves one reader per request, so each side only serves
    half of them and its total read time covers half the run. Summing the two
    medians gives the throughput the run actually sustained; summing tokens
    over summed seconds would instead average the two sides.
    """
    out: dict[str, float] = {}
    aggregate = 0.0
    for side in ("prefill", "decode"):
        values = [
            int(row.get(f"{side}_read_tokens_diagnostic", 0) or 0)
            * KV_BYTES_PER_TOKEN
            / float(row[f"{side}_lustre_read_seconds"])
            / 1e9
            for row in rounds
            if int(row.get(f"{side}_read_tokens_diagnostic", 0) or 0) > 0
            and float(row.get(f"{side}_lustre_read_seconds", 0) or 0) > 0
        ]
        median = statistics.median(values) if values else 0.0
        out[side] = median
        aggregate += median
    out["aggregate"] = aggregate
    return out


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
    components = components_for_rows(rounds)
    values = {
        field: sum(float(row.get(field, 0.0)) for row in rounds) / len(rounds)
        for field, _, _ in components
    }
    ttft = sum(float(row.get("ttft_seconds", 0.0)) for row in rounds) / len(rounds)
    jct = max(float(row.get("elapsed_seconds", 0.0)) for row in trajectories)
    summary = find_summary(rounds_path)
    bandwidth = read_bandwidth(rounds)
    label = f"{topology(task)}-{path}"
    return {
        "job_id": job_id,
        "task": task,
        "label": label,
        "path": path,
        "ttft_seconds": ttft,
        "jct_seconds": jct,
        "tpot_ms": float(summary.get("tpot_seconds", {}).get("mean", 0.0)) * 1000.0,
        "read_bandwidth_gbps": bandwidth,
        "components": values,
        "component_fields": [field for field, _, _ in components],
        "rounds": len(rounds),
        "trajectories": len(trajectories),
    }


def short_label(case: dict[str, object]) -> str:
    return str(case["path"]).replace("-Path", "")


def write_svg(cases: list[dict[str, object]], output: Path, title: str) -> None:
    width, height = 1080, 400
    elements = [f'<rect width="{width}" height="{height}" fill="white"/>']
    elements.append(svg_text(width / 2, 25, title, 16, weight="bold"))

    def bandwidth(case, key):
        return float(case["read_bandwidth_gbps"][key])

    # One row, panels ordered outcome first then the mechanism behind it.
    panels = [
        ("Worst-agent JCT [s]", "jct"),
        ("Mean TTFT [ms]", "ttft"),
        ("Lustre Read [GB/s]", "bandwidth"),
        ("Mean TPOT [ms]", "tpot"),
    ]
    maxima = {
        "jct": max(float(c["jct_seconds"]) for c in cases) * 1.2,
        "ttft": max(float(c["ttft_seconds"]) for c in cases) * 1000.0 * 1.2,
        "bandwidth": max(bandwidth(c, "aggregate") for c in cases) * 1.25,
        "tpot": max(float(c["tpot_ms"]) for c in cases) * 1.25,
    }
    panel_width, gap, margin = 224.0, 40.0, 68.0
    top, bottom = 62.0, 268.0

    for pindex, (panel_title, kind) in enumerate(panels):
        left = margin + pindex * (panel_width + gap)
        right = left + panel_width
        max_value = maxima[kind] or 1.0
        spacing = panel_width / len(cases)
        bar_width = min(58.0, spacing * 0.56)
        elements.append(svg_text((left + right) / 2, top - 14, panel_title, 13, weight="bold"))
        for tick in range(5):
            value = max_value * tick / 4
            y = bottom - (bottom - top) * tick / 4
            elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
            fmt = f"{value:.1f}" if max_value < 30 else f"{value:.0f}"
            elements.append(svg_text(left - 7, y + 4, fmt, 11, "end"))
        for index, case in enumerate(cases):
            x = left + spacing * (index + 0.5)
            if kind == "ttft":
                cursor = bottom
                for field, _, color in components_for_rows([{"path": case["path"]}]):
                    value = float(case["components"].get(field, 0.0))
                    segment = value * 1000.0 / max_value * (bottom - top)
                    cursor -= segment
                    elements.append(
                        f'<rect x="{x - bar_width / 2:.1f}" y="{cursor:.1f}" '
                        f'width="{bar_width:.1f}" height="{max(segment, 0.0):.1f}" fill="{color}"/>'
                    )
                elements.append(svg_text(x, cursor - 7, f"{float(case['ttft_seconds']) * 1000:.0f}", 11))
            elif kind == "bandwidth":
                # Stack the two sides: a static path has one of them at zero,
                # so the bar height is the aggregate either way.
                cursor = bottom
                for side, color in (("prefill", COLORS["orange"]), ("decode", COLORS["darkorange"])):
                    segment = bandwidth(case, side) / max_value * (bottom - top)
                    cursor -= segment
                    elements.append(
                        f'<rect x="{x - bar_width / 2:.1f}" y="{cursor:.1f}" '
                        f'width="{bar_width:.1f}" height="{max(segment, 0.0):.1f}" fill="{color}"/>'
                    )
                elements.append(svg_text(x, cursor - 7, f"{bandwidth(case, 'aggregate'):.1f}", 11))
            else:
                value = float(case["jct_seconds"]) if kind == "jct" else float(case["tpot_ms"])
                y = bottom - value / max_value * (bottom - top)
                fill = "#94a3b8" if kind == "jct" else COLORS["cyan"]
                elements.append(
                    f'<rect x="{x - bar_width / 2:.1f}" y="{y:.1f}" '
                    f'width="{bar_width:.1f}" height="{bottom - y:.1f}" fill="{fill}"/>'
                )
                elements.append(svg_text(x, y - 7, f"{value:.2f}" if kind == "tpot" else f"{value:.1f}", 11))
            elements.append(svg_text(x, bottom + 18, short_label(case), 12, weight="bold"))

    # TTFT stage legend. The dynamic table is a superset, so collecting across
    # cases keeps one entry per stage whichever paths were passed in.
    legend_components = []
    seen_fields = set()
    for case in cases:
        for component in components_for_rows([{"path": case["path"]}]):
            if component[0] not in seen_fields:
                seen_fields.add(component[0])
                legend_components.append(component)
    columns = 5
    column_width = (width - 2 * margin) / columns
    for index, (_, label, color) in enumerate(legend_components):
        x = margin + (index % columns) * column_width
        y = 320.0 + (index // columns) * 17.0
        elements.append(f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="10" height="10" fill="{color}"/>')
        elements.append(svg_text(x + 14, y, label, 10, "start"))

    note = (
        "TTFT bars are stage means; a dynamic request reads on exactly one side, "
        "so its two read stages each cover half the requests. "
        "Lustre Read is the median per-request GB/s per side, stacked to the aggregate."
    )
    elements.append(svg_text(width / 2, height - 10, note, 10))
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
    parser.add_argument(
        "--title", default="JCT, TTFT, Lustre read bandwidth and TPOT comparison"
    )
    parser.add_argument("--name", default="pd-case-comparison")
    args = parser.parse_args()
    if not 2 <= len(args.case) <= 6:
        parser.error("pass between two and six --case JOB_ID TASK_NAME arguments")
    cases = [load_case(args.code_home, job_id, task) for job_id, task in args.case]
    output_dir = args.output_dir or args.code_home / "results" / "pd-case-comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_svg(cases, output_dir / f"{args.name}.svg", args.title)
    (output_dir / f"{args.name}.json").write_text(json.dumps(cases, indent=2) + "\n")
    print(f"Wrote {output_dir / f'{args.name}.svg'}")


if __name__ == "__main__":
    main()
