#!/usr/bin/env python3
"""Collect one task into dataset/agents/rep result directories."""

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
    for summary in sorted(result_dir.glob("*/agents-*/rep-*/summary.json")):
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
    best = max(
        trajectories,
        key=lambda item: (
            float(item["elapsed_seconds"]),
            int(item["turns_completed"]),
            str(item["trajectory_id"]),
        ),
    )
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
        help="Shared runtime results (raw rounds/summary)",
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-result-dir",
        type=Path,
        required=True,
        help="Labhome repo-side charts/aggregates under CODE_HOME/results/",
    )
    parser.add_argument(
        "--code-home", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()

    scripts = args.code_home / "scripts"
    benchmark_dir = scripts / "benchmark"
    plot_dir = scripts / "plot"
    sys.path.insert(0, str(plot_dir))
    from plot_style import metadata_title_suffix
    args.repo_result_dir.mkdir(parents=True, exist_ok=True)
    # Keep all derived summaries and plots on the repository side. The shared
    # result directory is the immutable source of replay output.
    summary_dir = args.repo_result_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            str(benchmark_dir / "summarize_latency_benchmark.py"),
            "--result-dir",
            str(args.result_dir),
            "--log-dir",
            str(args.log_dir),
            "--output-dir",
            str(summary_dir),
        ]
    )

    runs = completed_runs(args.result_dir)
    if not runs:
        raise SystemExit(f"No completed runs under {args.result_dir}")

    for run_dir in runs:
        metadata = json.loads((run_dir / "run.json").read_text())
        title_suffix = metadata_title_suffix(metadata)
        dataset = str(
            metadata.get("dataset", f"{int(metadata['max_len_k'])}K")
        )
        agents = int(metadata["agents"])
        run_id = run_id_from_dir(run_dir)
        run_out = (
            args.repo_result_dir
            / dataset
            / f"agents-{agents}"
            / f"rep-{metadata['repetition']}"
        )
        data_dir = run_out / "data"
        csv_dir = run_out / "csv"
        longest_dir = run_out / "longest-trajectory"
        all_dir = run_out / "all-trajectories"
        for directory in (data_dir, csv_dir, longest_dir, all_dir):
            directory.mkdir(parents=True, exist_ok=True)

        for name in ("run.json", "summary.json", "rounds.jsonl", "trajectories.jsonl"):
            shutil.copy2(run_dir / name, data_dir / name)

        fullchain_dir = data_dir / "fullchain"
        fullchain_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                str(benchmark_dir / "summarize_ttft_fullchain.py"),
                "--rounds",
                str(run_dir / "rounds.jsonl"),
                "--log-dir",
                str(args.log_dir),
                "--log-glob",
                "*.log",
                "--output-dir",
                str(fullchain_dir),
            ]
        )

        trajectory_id = pick_trajectory_id(run_dir / "trajectories.jsonl")
        rounds_one, _ = filter_rounds(
            run_dir / "rounds.jsonl",
            run_dir / "trajectories.jsonl",
            trajectory_id,
            data_dir / "longest-trajectory",
        )
        run(
            [
                sys.executable,
                str(plot_dir / "plot_trajectory_metrics.py"),
                "--rounds",
                str(rounds_one),
                "--log-dir",
                str(args.log_dir),
                "--log-glob",
                "*.log",
                "--l3-read-path",
                str(metadata.get("l3_read_path", "prefill")),
                "--output-dir",
                str(longest_dir),
                "--csv-output",
                str(csv_dir / "longest-trajectory.csv"),
                "--summary-output",
                str(data_dir / "longest-trajectory-summary.json"),
                f"--title-suffix={title_suffix}",
            ]
        )
        run(
            [
                sys.executable,
                str(plot_dir / "plot_trajectory_latency.py"),
                "--rounds",
                str(rounds_one),
                "--output-dir",
                str(longest_dir),
                f"--title-suffix={title_suffix}",
            ]
        )
        run(
            [
                sys.executable,
                str(plot_dir / "plot_run_summary.py"),
                "--run-dir",
                str(run_dir),
                "--fullchain-rounds",
                str(fullchain_dir / "fullchain_rounds.jsonl"),
                "--output-dir",
                str(all_dir),
                "--csv-output",
                str(csv_dir / "all-trajectories.csv"),
                "--json-output",
                str(data_dir / "all-trajectories-summary.json"),
                f"--title-suffix={title_suffix}",
            ]
        )

    run(
        [
            sys.executable,
            str(plot_dir / "plot_task_jct.py"),
            "--result-dir",
            str(args.result_dir),
            "--output-dir",
            str(args.repo_result_dir),
            f"--title-suffix={metadata_title_suffix(json.loads((runs[0] / 'run.json').read_text()))}",
        ]
    )

    task_data = args.repo_result_dir / "data"
    task_csv = args.repo_result_dir / "csv"
    task_data.mkdir(exist_ok=True)
    task_csv.mkdir(exist_ok=True)
    for name in ("aggregate.csv", "runs.csv"):
        source = summary_dir / name
        if source.exists():
            shutil.copy2(source, task_csv / name)
    for name in ("validation.json", "merged_rounds.jsonl"):
        source = summary_dir / name
        if source.exists():
            shutil.copy2(source, task_data / name)

    print(
        "COLLECT_BENCHMARK_METRICS_COMPLETE "
        f"repo_result_dir={args.repo_result_dir}"
    )


if __name__ == "__main__":
    main()
