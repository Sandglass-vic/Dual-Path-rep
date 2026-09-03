#!/usr/bin/env python3
"""Generate length-only agent trajectories calibrated to DualPath Table 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean


@dataclass(frozen=True)
class Target:
    name: str
    max_len: int
    turns: int
    append: int
    gen: int
    total: int
    context: int
    low_turns: int


TARGETS = (
    Target("32k", 32 * 1024, 60, 608, 148, 28639, 17183, 26),
    Target("48k", 48 * 1024, 106, 474, 172, 42607, 25120, 45),
    Target("64k", 64 * 1024, 157, 429, 176, 55958, 32721, 63),
)


def balanced_round(value: float, count: int) -> list[int]:
    """Round a repeated real value while preserving its aggregate."""
    lower = math.floor(value)
    upper_count = round(value * count) - lower * count
    values = [lower] * count
    for index in range(upper_count):
        values[(index * 137) % count] += 1
    return values


def allocate_weighted(total: int, weights: list[float]) -> list[int]:
    raw = [total * weight / sum(weights) for weight in weights]
    values = [math.floor(value) for value in raw]
    remainder = total - sum(values)
    order = sorted(
        range(len(raw)), key=lambda index: raw[index] - values[index], reverse=True
    )
    for index in order[:remainder]:
        values[index] += 1
    return values


def trajectory_totals(target: Target, count: int) -> list[tuple[int, int, int]]:
    """Create two equally sized, negatively correlated turn/length groups."""
    if count % 2:
        raise ValueError("trajectory count must be even")

    high_turns = 2 * target.turns - target.low_turns
    mean_step = target.append + target.gen
    low_step = (high_turns * mean_step - target.total) / (
        0.5 * (high_turns - target.low_turns)
    )
    high_step = 2 * mean_step - low_step
    append_fraction = target.append / mean_step

    half = count // 2
    low_append = balanced_round(target.low_turns * low_step * append_fraction, half)
    low_gen = balanced_round(target.low_turns * low_step * (1 - append_fraction), half)
    high_append = balanced_round(
        high_turns * high_step * append_fraction, half
    )
    high_gen = balanced_round(high_turns * high_step * (1 - append_fraction), half)

    result = [
        (target.low_turns, low_append[index], low_gen[index])
        for index in range(half)
    ]
    result.extend(
        (high_turns, high_append[index], high_gen[index])
        for index in range(half)
    )

    expected_total = target.total * count
    total_error = expected_total - sum(append + gen for _, append, gen in result)
    step = 1 if total_error > 0 else -1
    for index in range(abs(total_error)):
        turns, append, gen = result[index % count]
        result[index % count] = (turns, append, gen + step)

    if max(append + gen for _, append, gen in result) > target.max_len:
        raise ValueError(f"{target.name} generated a trajectory beyond MaxLen")
    return result


def make_turns(
    turn_count: int,
    append_total: int,
    gen_total: int,
    decay: float,
    noise: list[float],
) -> list[dict[str, int]]:
    denominator = max(turn_count - 1, 1)
    weights = [
        math.exp(-decay * index / denominator) * noise[index]
        for index in range(turn_count)
    ]
    appends = allocate_weighted(append_total, weights)
    generations = allocate_weighted(gen_total, weights)

    context = 0
    turns = []
    for index, (append, generation) in enumerate(zip(appends, generations), start=1):
        prompt_context = context + append
        turns.append(
            {
                "turn_id": index,
                "append_tokens": append,
                "generate_tokens": generation,
                "prompt_context_tokens": prompt_context,
            }
        )
        context += append + generation
    return turns


def build_dataset(
    target: Target, count: int, seed: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    totals = trajectory_totals(target, count)
    noises = []
    for trajectory_id, (turn_count, _, _) in enumerate(totals):
        rng = random.Random(seed + trajectory_id * 1009)
        noises.append([math.exp(rng.gauss(0, 0.08)) for _ in range(turn_count)])

    def generate(decay: float) -> list[dict[str, object]]:
        trajectories = []
        for trajectory_id, ((turn_count, append, gen), noise) in enumerate(
            zip(totals, noises)
        ):
            turns = make_turns(turn_count, append, gen, decay, noise)
            trajectories.append(
                {
                    "schema_version": 1,
                    "dataset": f"dualpath-fake-{target.name}",
                    "trajectory_id": f"{target.name}-{trajectory_id:04d}",
                    "max_agent_length": target.max_len,
                    "turn_count": turn_count,
                    "total_tokens": append + gen,
                    "mean_prompt_context_tokens": fmean(
                        turn["prompt_context_tokens"] for turn in turns
                    ),
                    "turns": turns,
                }
            )
        return trajectories

    best: tuple[float, float, list[dict[str, object]]] | None = None
    lower, upper = 0.0, 5.0
    for _ in range(32):
        decay = (lower + upper) / 2
        trajectories = generate(decay)
        observed = fmean(
            float(item["mean_prompt_context_tokens"]) for item in trajectories
        )
        error = abs(observed - target.context)
        if best is None or error < best[0]:
            best = (error, decay, trajectories)
        if observed < target.context:
            lower = decay
        else:
            upper = decay

    assert best is not None
    _, decay, trajectories = best
    observed = {
        "trajectories": count,
        "mean_turns": fmean(int(item["turn_count"]) for item in trajectories),
        "mean_append_tokens_per_turn_per_trajectory": fmean(
            sum(turn["append_tokens"] for turn in item["turns"])
            / int(item["turn_count"])
            for item in trajectories
        ),
        "mean_generate_tokens_per_turn_per_trajectory": fmean(
            sum(turn["generate_tokens"] for turn in item["turns"])
            / int(item["turn_count"])
            for item in trajectories
        ),
        "mean_total_tokens": fmean(
            int(item["total_tokens"]) for item in trajectories
        ),
        "mean_prompt_context_tokens": fmean(
            float(item["mean_prompt_context_tokens"]) for item in trajectories
        ),
        "max_total_tokens": max(int(item["total_tokens"]) for item in trajectories),
    }
    metadata = {
        "target": asdict(target),
        "observed": observed,
        "context_frontload_decay": decay,
        "seed": seed,
        "interpretation": (
            "Append/Gen are means of each trajectory's per-turn mean; Context is "
            "the mean of each trajectory's mean prompt context."
        ),
    }
    return trajectories, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectories", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260226)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "source": "DualPath paper Table 2 (arXiv:2602.21548v2)",
        "length_only": True,
        "datasets": [],
    }
    for offset, target in enumerate(TARGETS):
        trajectories, metadata = build_dataset(
            target, args.trajectories, args.seed + offset * 1_000_003
        )
        output = args.output_dir / f"dualpath_fake_{target.name}.jsonl"
        with output.open("w", encoding="utf-8") as stream:
            for trajectory in trajectories:
                stream.write(json.dumps(trajectory, separators=(",", ":")) + "\n")
        metadata["file"] = output.name
        metadata["bytes"] = output.stat().st_size
        metadata["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        manifest["datasets"].append(metadata)

    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    readme = """# DualPath fake agent traces

These are deterministic, length-only synthetic trajectories calibrated to
Table 2 of the DualPath paper. They do not contain production prompts or model
outputs. Each JSONL line is one trajectory; `turns` records append length,
forced generation length, and cumulative prompt-context length.

The five paper averages cannot be reproduced by fixed-size turns because
`mean(Turns) * (mean(Append) + mean(Gen)) != mean(Total)`. The generator uses
two negatively correlated trajectory groups and front-loads tokens to match
the reported Context mean. See `manifest.json` for targets, observed values,
seed, and the exact interpretation.

A replay client must synthesize exactly `append_tokens` input token IDs, append
the model's actual prior output IDs to the next prompt, set `max_new_tokens` to
`generate_tokens`, and ignore EOS so generation length is exact.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
