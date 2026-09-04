#!/usr/bin/env python3
"""Compare agent concurrency with per-agent distributions and full-chain TTFT."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


FULLCHAIN_COMPONENTS = (
    ("bootstrap_and_scheduling_seconds", "Prefill Bootstrap & Scheduling", "#64748b"),
    ("l3_kv_read_seconds", "L3 KV Read", "#f59e0b"),
    (
        "prefill_logits_sampling_seconds",
        "Prefill Forward + Logits + Sampling",
        "#22c55e",
    ),
    ("pd_transfer_seconds", "Prefill-to-Decode KV Transfer", "#f97316"),
    ("decode_dispatch_seconds", "Decode Prebuilt-token Dispatch", "#8b5cf6"),
    ("sse_egress_seconds", "SSE Serialization & Egress", "#06b6d4"),
    ("router_client_runtime_seconds", "Router, Client & Runtime Gaps", "#cbd5e1"),
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


def quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    left = int(position)
    right = min(left + 1, len(sorted_values) - 1)
    weight = position - left
    return sorted_values[left] * (1.0 - weight) + sorted_values[right] * weight


def box_stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "min": ordered[0],
        "q1": quantile(ordered, 0.25),
        "median": quantile(ordered, 0.5),
        "q3": quantile(ordered, 0.75),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "count": float(len(ordered)),
    }


def load_agent_series(run_dir: Path) -> dict[str, list[float]]:
    trajectories = [
        json.loads(line)
        for line in (run_dir / "trajectories.jsonl").read_text().splitlines()
    ]
    rounds = [
        json.loads(line)
        for line in (run_dir / "rounds.jsonl").read_text().splitlines()
    ]
    ttft_by_agent: dict[str, list[float]] = defaultdict(list)
    tpot_by_agent: dict[str, list[float]] = defaultdict(list)
    for item in rounds:
        agent = str(item["trajectory_id"])
        ttft_by_agent[agent].append(float(item["ttft_seconds"]))
        if item.get("tpot_seconds") is not None:
            tpot_by_agent[agent].append(float(item["tpot_seconds"]))
    return {
        "trajectory_jct_seconds": [
            float(item["elapsed_seconds"]) for item in trajectories
        ],
        "agent_mean_ttft_seconds": [
            statistics.fmean(values) for values in ttft_by_agent.values()
        ],
        "agent_mean_tpot_seconds": [
            statistics.fmean(values) for values in tpot_by_agent.values()
        ],
        "request_ttft_seconds": [
            float(item["ttft_seconds"]) for item in rounds
        ],
    }


def discover_runs(result_dir: Path) -> list[tuple[int, int, Path]]:
    runs = []
    for summary in sorted(result_dir.glob("*k/agents-*/rep-*/summary.json")):
        run_dir = summary.parent
        metadata = json.loads((run_dir / "run.json").read_text())
        runs.append(
            (
                int(metadata["max_len_k"]),
                int(metadata["agents"]),
                run_dir,
            )
        )
    if not runs:
        raise RuntimeError(f"No completed runs under {result_dir}")
    return runs


def draw_boxplot_panel(
    elements: list[str],
    series_by_agents: dict[int, list[float]],
    left: float,
    right: float,
    top: float,
    bottom: float,
    title: str,
    ylabel: str,
    scale: float = 1.0,
    y_digits: int = 2,
) -> None:
    agent_counts = sorted(series_by_agents)
    stats = {
        agents: box_stats([value * scale for value in values])
        for agents, values in series_by_agents.items()
    }
    y_max = max(item["max"] for item in stats.values()) * 1.15 or 1.0
    elements.append(text((left + right) / 2, top - 18, title, 13, weight="bold"))
    for tick in range(6):
        value = y_max * tick / 5
        y = bottom - (bottom - top) * tick / 5
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            'stroke="#e5e7eb"/>'
        )
        elements.append(
            text(left - 8, y + 4, f"{value:.{y_digits}f}", 10, "end")
        )
    span = right - left
    for index, agents in enumerate(agent_counts):
        item = stats[agents]
        x = left + span * (index + 0.5) / len(agent_counts)
        box_width = min(54.0, span / max(len(agent_counts), 1) * 0.35)

        def y_of(value: float) -> float:
            return bottom - value / y_max * (bottom - top)

        y_min = y_of(item["min"])
        y_q1 = y_of(item["q1"])
        y_med = y_of(item["median"])
        y_q3 = y_of(item["q3"])
        y_max_v = y_of(item["max"])
        y_mean = y_of(item["mean"])
        elements.extend(
            [
                f'<line x1="{x:.1f}" y1="{y_max_v:.1f}" x2="{x:.1f}" '
                f'y2="{y_min:.1f}" stroke="#111"/>',
                f'<line x1="{x - 8:.1f}" y1="{y_max_v:.1f}" x2="{x + 8:.1f}" '
                f'y2="{y_max_v:.1f}" stroke="#111"/>',
                f'<line x1="{x - 8:.1f}" y1="{y_min:.1f}" x2="{x + 8:.1f}" '
                f'y2="{y_min:.1f}" stroke="#111"/>',
                f'<rect x="{x - box_width / 2:.1f}" y="{y_q3:.1f}" '
                f'width="{box_width:.1f}" height="{max(1.0, y_q1 - y_q3):.1f}" '
                'fill="#fdba74" stroke="#9a3412"/>',
                f'<line x1="{x - box_width / 2:.1f}" y1="{y_med:.1f}" '
                f'x2="{x + box_width / 2:.1f}" y2="{y_med:.1f}" '
                'stroke="#9a3412" stroke-width="2"/>',
                f'<circle cx="{x:.1f}" cy="{y_mean:.1f}" r="3.5" fill="#1d4ed8"/>',
                text(x, bottom + 18, str(agents), 11),
                text(
                    x,
                    bottom + 34,
                    f"n={int(item['count'])}",
                    9,
                ),
            ]
        )
    elements.append(
        f'<text x="{left - 42:.1f}" y="{(top + bottom) / 2:.1f}" '
        'font-family="sans-serif" font-size="11" text-anchor="middle" '
        f'transform="rotate(-90 {left - 42:.1f} {(top + bottom) / 2:.1f})">'
        f"{html.escape(ylabel)}</text>"
    )


def write_agent_boxplots(
    runs: list[tuple[int, int, Path]], output: Path
) -> dict[str, Any]:
    by_agents: dict[int, dict[str, list[float]]] = {}
    max_len = {max_len for max_len, _, _ in runs}
    if len(max_len) != 1:
        raise RuntimeError(f"Expected one MaxLen in comparison, got {max_len}")
    max_len_k = next(iter(max_len))
    for _, agents, run_dir in runs:
        by_agents[agents] = load_agent_series(run_dir)

    width, height = 1280, 520
    panels = [
        (
            "trajectory_jct_seconds",
            "trajectory_jct across agents",
            "trajectory_jct (s)",
            1.0,
            1,
        ),
        (
            "agent_mean_ttft_seconds",
            "Per-agent mean TTFT",
            "Mean TTFT (ms)",
            1000.0,
            0,
        ),
        (
            "agent_mean_tpot_seconds",
            "Per-agent mean TPOT",
            "Mean TPOT (ms)",
            1000.0,
            1,
        ),
    ]
    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(
            width / 2,
            28,
            f"Qwen2.5-32B 1P2D · MaxLen={max_len_k}K · per-agent distributions",
            16,
            weight="bold",
        ),
        text(
            width / 2,
            48,
            "Each box is the distribution over agents in that concurrency "
            "(whiskers=min/max, box=Q1–Q3, line=median, blue dot=mean)",
            11,
        ),
    ]
    panel_width = 360.0
    for panel_index, (field, title, ylabel, scale, digits) in enumerate(panels):
        left = 80.0 + panel_index * 400.0
        right = left + panel_width
        series = {
            agents: values[field] for agents, values in sorted(by_agents.items())
        }
        draw_boxplot_panel(
            elements,
            series,
            left,
            right,
            90.0,
            430.0,
            title,
            ylabel,
            scale=scale,
            y_digits=digits,
        )
        elements.append(text((left + right) / 2, 470, "#Agents", 12))
    elements.append(
        text(
            width / 2,
            505,
            "Not a cross-repetition plot: with one rep, comparing agents inside "
            "each concurrency setting.",
            10,
        )
    )
    output.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )
    summary = {
        agents: {
            field: box_stats(values[field])
            for field in (
                "trajectory_jct_seconds",
                "agent_mean_ttft_seconds",
                "agent_mean_tpot_seconds",
            )
        }
        for agents, values in by_agents.items()
    }
    return summary, by_agents, max_len_k


def write_jct_max_by_agents(
    runs: list[tuple[int, int, Path]],
    by_agents: dict[int, dict[str, list[float]]],
    max_len_k: int,
    output: Path,
) -> dict[str, Any]:
    """Bar chart of max trajectory_jct vs #Agents (batch critical-path JCT)."""
    points: dict[int, dict[str, float]] = {}
    for _, agents, run_dir in runs:
        jcts = by_agents[agents]["trajectory_jct_seconds"]
        summary = json.loads((run_dir / "summary.json").read_text())
        points[agents] = {
            "trajectory_jct_max_seconds": max(jcts),
            "trajectory_jct_min_seconds": min(jcts),
            "trajectory_jct_mean_seconds": statistics.fmean(jcts),
            "batch_wall_seconds": float(summary["wall_seconds"]),
            "n_agents": float(len(jcts)),
        }

    width, height = 900, 480
    plot_left, plot_right = 90.0, 720.0
    plot_top, plot_bottom = 80.0, 400.0
    agent_counts = sorted(points)
    y_max = (
        max(points[a]["trajectory_jct_max_seconds"] for a in agent_counts) * 1.15
        or 1.0
    )
    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(
            width / 2,
            28,
            f"Batch JCT · max trajectory_jct · MaxLen={max_len_k}K",
            16,
            weight="bold",
        ),
        text(
            width / 2,
            48,
            "Each bar = max agent trajectory_jct (critical path); "
            "marker = summary.json wall_seconds",
            11,
        ),
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = plot_bottom - (plot_bottom - plot_top) * tick / 5
        elements.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" '
            f'y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        elements.append(text(plot_left - 8, y + 4, f"{value:.0f}", 10, "end"))

    bar_width = min(100.0, 380.0 / max(len(agent_counts), 1))
    span = plot_right - plot_left
    for index, agents in enumerate(agent_counts):
        item = points[agents]
        x = plot_left + span * (index + 0.5) / len(agent_counts)
        height_px = item["trajectory_jct_max_seconds"] / y_max * (
            plot_bottom - plot_top
        )
        y = plot_bottom - height_px
        elements.append(
            f'<rect x="{x - bar_width / 2:.1f}" y="{y:.1f}" '
            f'width="{bar_width:.1f}" height="{max(height_px, 0.0):.1f}" '
            'fill="#fb923c" stroke="#9a3412"/>'
        )
        wall_y = plot_bottom - item["batch_wall_seconds"] / y_max * (
            plot_bottom - plot_top
        )
        elements.append(
            f'<circle cx="{x:.1f}" cy="{wall_y:.1f}" r="5" fill="#1d4ed8"/>'
        )
        elements.append(text(x, 425, str(agents), 12))
        elements.append(
            text(
                x,
                y - 10,
                f"{item['trajectory_jct_max_seconds']:.1f}s",
                11,
            )
        )

    elements.append(text((plot_left + plot_right) / 2, 450, "#Agents", 12))
    elements.append(
        '<text x="28" y="240" font-family="sans-serif" font-size="12" '
        'text-anchor="middle" transform="rotate(-90 28 240)">'
        "Seconds</text>"
    )
    elements.append(
        f'<rect x="740" y="100" width="14" height="14" fill="#fb923c" '
        'stroke="#9a3412"/>'
    )
    elements.append(text(760, 112, "max trajectory_jct", 11, "start"))
    elements.append(f'<circle cx="747" cy="140" r="5" fill="#1d4ed8"/>')
    elements.append(text(760, 144, "batch wall_seconds", 11, "start"))
    elements.append(
        text(
            width / 2,
            468,
            "Equals batch critical-path JCT when all agents share one barrier; "
            "not the sum of agent JCTs.",
            10,
        )
    )
    output.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )
    return points


def write_fullchain_by_agents(
    fullchain_dirs: dict[int, Path], output: Path, max_len_k: int
) -> dict[str, Any]:
    means: dict[int, dict[str, float]] = {}
    for agents, directory in sorted(fullchain_dirs.items()):
        rounds = [
            json.loads(line)
            for line in (directory / "fullchain_rounds.jsonl").read_text().splitlines()
        ]
        if not rounds:
            raise RuntimeError(f"No full-chain rounds in {directory}")
        means[agents] = {
            field: statistics.fmean(float(row[field]) for row in rounds)
            for field, _, _ in FULLCHAIN_COMPONENTS
        }
        means[agents]["ttft_seconds"] = statistics.fmean(
            float(row["ttft_seconds"]) for row in rounds
        )
        means[agents]["accounting_error_max_abs"] = max(
            abs(float(row["accounting_error_seconds"])) for row in rounds
        )
        means[agents]["n_requests"] = float(len(rounds))

    width, height = 1280, 520
    plot_left, plot_right = 90.0, 760.0
    plot_top, plot_bottom = 80.0, 430.0
    agent_counts = sorted(means)
    y_max = (
        max(means[agents]["ttft_seconds"] for agents in agent_counts) * 1000 * 1.15
        or 1.0
    )
    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(
            width / 2,
            28,
            f"Full-chain additive TTFT breakdown · MaxLen={max_len_k}K",
            16,
            weight="bold",
        ),
        text(
            width / 2,
            48,
            "Stacked means over all requests; components sum to client TTFT",
            11,
        ),
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = plot_bottom - (plot_bottom - plot_top) * tick / 5
        elements.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" '
            f'y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        elements.append(text(plot_left - 8, y + 4, f"{value:.0f}", 10, "end"))

    bar_width = min(120.0, 420.0 / max(len(agent_counts), 1))
    span = plot_right - plot_left
    for index, agents in enumerate(agent_counts):
        x = plot_left + span * (index + 0.5) / len(agent_counts)
        bottom = plot_bottom
        for field, _, color in FULLCHAIN_COMPONENTS:
            value_ms = means[agents][field] * 1000.0
            height_px = value_ms / y_max * (plot_bottom - plot_top)
            bottom -= height_px
            elements.append(
                f'<rect x="{x - bar_width / 2:.1f}" y="{bottom:.1f}" '
                f'width="{bar_width:.1f}" height="{max(height_px, 0.0):.1f}" '
                f'fill="{color}"/>'
            )
        elements.append(text(x, 455, str(agents), 12))
        elements.append(
            text(
                x,
                bottom - 8,
                f"{means[agents]['ttft_seconds'] * 1000:.0f} ms",
                11,
            )
        )
    elements.append(text((plot_left + plot_right) / 2, 480, "#Agents", 12))
    elements.append(
        '<text x="28" y="255" font-family="sans-serif" font-size="12" '
        'text-anchor="middle" transform="rotate(-90 28 255)">Mean TTFT (ms)</text>'
    )

    legend_x, legend_y = 820.0, 100.0
    for index, (_, label, color) in enumerate(FULLCHAIN_COMPONENTS):
        y = legend_y + index * 36
        elements.append(
            f'<rect x="{legend_x}" y="{y - 12:.1f}" width="16" height="16" '
            f'fill="{color}"/>'
        )
        elements.append(text(legend_x + 24, y + 1, label, 11, "start"))
    elements.append(
        text(
            width / 2,
            508,
            "Full-chain exclusive buckets (see docs/metrics.md); not the legacy proxy.",
            10,
        )
    )
    output.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )
    return means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fullchain-root",
        type=Path,
        help="Directory containing agents-<n>/fullchain_rounds.jsonl",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.result_dir)
    box_summary, by_agents, max_len_k = write_agent_boxplots(
        runs, args.output_dir / "agent_latency_boxplots.svg"
    )
    (args.output_dir / "agent_latency_boxplots.json").write_text(
        json.dumps(box_summary, indent=2) + "\n"
    )
    jct_max = write_jct_max_by_agents(
        runs,
        by_agents,
        max_len_k,
        args.output_dir / "trajectory_jct_max_by_agents.svg",
    )
    (args.output_dir / "trajectory_jct_max_by_agents.json").write_text(
        json.dumps(jct_max, indent=2) + "\n"
    )

    if args.fullchain_root is not None:
        fullchain_dirs: dict[int, Path] = {}
        for agents_dir in sorted(args.fullchain_root.glob("agents-*")):
            agents = int(agents_dir.name.split("-", 1)[1])
            rounds = agents_dir / "fullchain_rounds.jsonl"
            if rounds.exists():
                fullchain_dirs[agents] = agents_dir
        if not fullchain_dirs:
            raise RuntimeError(f"No full-chain outputs under {args.fullchain_root}")
        means = write_fullchain_by_agents(
            fullchain_dirs,
            args.output_dir / "ttft_fullchain_by_agents.svg",
            max_len_k,
        )
        (args.output_dir / "ttft_fullchain_by_agents.json").write_text(
            json.dumps(means, indent=2) + "\n"
        )

    print(f"Wrote agent comparison plots to {args.output_dir}")


if __name__ == "__main__":
    main()
