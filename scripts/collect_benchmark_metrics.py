#!/usr/bin/env python3
"""Collect JCT, full-chain TTFT, trajectory metrics, and agent-compare plots.

Writes summarized charts/CSV under the labhome repo result directory
(REPO_RESULT_DIR / --repo-result-dir), which is distinct from the Lustre
runtime RESULT_DIR used for raw rounds/logs.

Full-chain TTFT is retained only as all-request aggregates under
fullchain/agents-*/ (fullchain_rounds.jsonl + fullchain_summary.json) and
the cross-agent chart ttft_fullchain_by_agents.svg. Per-trajectory
fullchain slices are not produced.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def completed_runs(result_dir: Path) -> list[Path]:
    runs = []
    for summary in sorted(result_dir.glob("*k/agents-*/rep-*/summary.json")):
        run_dir = summary.parent
        if (run_dir / "rounds.jsonl").exists() and (run_dir / "run.json").exists():
            runs.append(run_dir)
    return runs


def run_id_from_dir(run_dir: Path) -> str:
    metadata = json.loads((run_dir / "run.json").read_text())
    return (
        f"{metadata['max_len_k']}k-a{metadata['agents']}-r{metadata['repetition']}"
    )


def pick_trajectory_id(trajectories_path: Path) -> str:
    trajectories = [
        json.loads(line) for line in trajectories_path.read_text().splitlines()
    ]
    if not trajectories:
        raise RuntimeError(f"No trajectories in {trajectories_path}")
    best = max(trajectories, key=lambda item: int(item["turns_completed"]))
    return str(best["trajectory_id"])


def filter_rounds(
    rounds_path: Path, trajectories_path: Path, trajectory_id: str, output_dir: Path
) -> tuple[Path, Path]:
    """Write one representative trajectory's rounds for trajectory-* plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rounds_out = output_dir / "rounds.jsonl"
    traj_out = output_dir / "trajectories.jsonl"
    with rounds_out.open("w") as stream:
        for line in rounds_path.read_text().splitlines():
            item = json.loads(line)
            if item["trajectory_id"] == trajectory_id:
                stream.write(json.dumps(item, separators=(",", ":")) + "\n")
    with traj_out.open("w") as stream:
        for line in trajectories_path.read_text().splitlines():
            item = json.loads(line)
            if item["trajectory_id"] == trajectory_id:
                stream.write(json.dumps(item, separators=(",", ":")) + "\n")
    return rounds_out, traj_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Lustre runtime results (raw rounds/summary)",
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-result-dir",
        type=Path,
        required=True,
        help="Labhome repo-side charts/aggregates under CODE_HOME/results/",
    )
    parser.add_argument(
        "--code-home", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()

    scripts = args.code_home / "scripts"
    plot_dir = scripts / "plot"
    args.repo_result_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = args.result_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            str(scripts / "summarize_latency_benchmark.py"),
            "--result-dir",
            str(args.result_dir),
            "--log-dir",
            str(args.log_dir),
            "--output-dir",
            str(summary_dir),
        ]
    )

    fullchain_root = args.repo_result_dir / "fullchain"
    runs = completed_runs(args.result_dir)
    if not runs:
        raise SystemExit(f"No completed runs under {args.result_dir}")

    for run_dir in runs:
        metadata = json.loads((run_dir / "run.json").read_text())
        agents = int(metadata["agents"])
        run_id = run_id_from_dir(run_dir)
        run_out = (
            args.repo_result_dir
            / f"agents-{agents}"
            / f"rep-{metadata['repetition']}"
        )
        run_out.mkdir(parents=True, exist_ok=True)
        fc_dir = fullchain_root / f"agents-{agents}"
        # Prefer rep-1; only regenerate if fullchain for this #Agents is missing.
        if int(metadata.get("repetition", 1)) == 1 or not (
            fc_dir / "fullchain_rounds.jsonl"
        ).exists():
            fc_dir.mkdir(parents=True, exist_ok=True)
            run(
                [
                    sys.executable,
                    str(scripts / "summarize_ttft_fullchain.py"),
                    "--rounds",
                    str(run_dir / "rounds.jsonl"),
                    "--log-dir",
                    str(args.log_dir),
                    "--log-glob",
                    f"*{run_id}*",
                    "--output-dir",
                    str(fc_dir),
                ]
            )

        # Representative trajectory plots only (not a fullchain slice).
        trajectory_id = pick_trajectory_id(run_dir / "trajectories.jsonl")
        filtered_dir = run_out / "trajectory-source"
        rounds_one, traj_one = filter_rounds(
            run_dir / "rounds.jsonl",
            run_dir / "trajectories.jsonl",
            trajectory_id,
            filtered_dir,
        )
        # Drop any leftover single-trajectory fullchain slices from older collects.
        leftover_fc = filtered_dir / "fullchain_rounds.jsonl"
        if leftover_fc.exists():
            leftover_fc.unlink()
        leftover_svg = run_out / "trajectory_ttft_breakdown.svg"
        if leftover_svg.exists():
            leftover_svg.unlink()

        run(
            [
                sys.executable,
                str(plot_dir / "plot_trajectory_metrics.py"),
                "--rounds",
                str(rounds_one),
                "--log-dir",
                str(args.log_dir),
                "--log-glob",
                f"*{run_id}*",
                "--output-dir",
                str(run_out / "trajectory-metrics"),
            ]
        )
        run(
            [
                sys.executable,
                str(plot_dir / "plot_trajectory_latency.py"),
                "--rounds",
                str(rounds_one),
                "--trajectories",
                str(traj_one),
                "--output-dir",
                str(run_out / "trajectory-latency"),
            ]
        )

    run(
        [
            sys.executable,
            str(plot_dir / "plot_agent_compare.py"),
            "--result-dir",
            str(args.result_dir),
            "--output-dir",
            str(args.repo_result_dir),
            "--fullchain-root",
            str(fullchain_root),
        ]
    )

    for name in (
        "aggregate.csv",
        "runs.csv",
        "validation.json",
        "merged_rounds.jsonl",
    ):
        source = summary_dir / name
        if source.exists():
            shutil.copy2(source, args.repo_result_dir / name)

    print(
        "COLLECT_BENCHMARK_METRICS_COMPLETE "
        f"repo_result_dir={args.repo_result_dir}"
    )


if __name__ == "__main__":
    main()
