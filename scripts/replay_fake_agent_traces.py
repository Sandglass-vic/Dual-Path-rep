#!/usr/bin/env python3
"""Replay deterministic length-only agent trajectories against SGLang."""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReplayConfig:
    endpoint: str
    timeout: float
    server_selection_timeout: float
    stream: bool
    request_id_prefix: str
    seed: int
    token_id_min: int
    token_id_max: int
    max_context_tokens: int | None


def synthetic_token_ids(
    count: int, config: ReplayConfig, trajectory_id: str, turn_id: int
) -> list[int]:
    """Return deterministic ordinary-vocabulary token IDs for one append."""
    mixed_seed = config.seed
    for char in f"{trajectory_id}:{turn_id}":
        mixed_seed = (mixed_seed * 131 + ord(char)) & ((1 << 64) - 1)
    rng = random.Random(mixed_seed)
    return [
        rng.randrange(config.token_id_min, config.token_id_max) for _ in range(count)
    ]


def post_generate(
    config: ReplayConfig,
    input_ids: list[int],
    generate_tokens: int,
    request_id: str,
) -> tuple[dict[str, Any], float, dict[str, Any], str, int]:
    start = time.perf_counter()
    selection_deadline = time.monotonic() + config.server_selection_timeout
    while True:
        payload = {
            "rid": request_id,
            "input_ids": input_ids,
            "stream": config.stream,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": generate_tokens,
                "ignore_eos": True,
                "stream_interval": 1,
            },
        }
        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            config.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.timeout) as response:
                if config.stream:
                    parsed, stream_metrics = read_streaming_response(
                        response, start
                    )
                    response_body = None
                else:
                    response_body = response.read()
                    stream_metrics = {}
            break
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            if (
                error.code == 503
                and "server_selection_failed" in detail
                and time.monotonic() < selection_deadline
            ):
                time.sleep(1)
                continue
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    latency = time.perf_counter() - start
    if not config.stream:
        parsed = json.loads(response_body)
        if isinstance(parsed, list):
            if len(parsed) != 1:
                raise RuntimeError(f"Expected one response, got {len(parsed)}")
            parsed = parsed[0]
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected response type: {type(parsed).__name__}")
    return (
        parsed,
        latency,
        stream_metrics,
        request_id,
        0,
    )


def read_streaming_response(
    response: Any, request_start: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    final_response: dict[str, Any] | None = None
    token_times: list[float] = []
    previous_output_count = 0
    batched_token_chunks = 0

    for raw_line in response:
        line = raw_line.decode(errors="replace").strip()
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        item = json.loads(payload)
        if "error" in item:
            raise RuntimeError(f"Streaming response error: {item['error']}")
        output_ids = extract_output_ids(item)
        new_tokens = len(output_ids) - previous_output_count
        if new_tokens < 0:
            raise RuntimeError("Streaming output_ids length decreased")
        if new_tokens:
            now = time.perf_counter()
            token_times.extend([now] * new_tokens)
            batched_token_chunks += max(0, new_tokens - 1)
            previous_output_count = len(output_ids)
        final_response = item

    if final_response is None:
        raise RuntimeError("Streaming response contained no JSON payload")
    if not token_times:
        raise RuntimeError("Streaming response contained no output tokens")

    itls = [
        token_times[index] - token_times[index - 1]
        for index in range(1, len(token_times))
    ]
    metrics = {
        "ttft_seconds": token_times[0] - request_start,
        "ttst_seconds": token_times[1] - request_start
        if len(token_times) >= 2
        else None,
        "tpot_seconds": statistics.fmean(itls) if itls else None,
        "itl_seconds": itls,
        "streamed_token_count": len(token_times),
        "batched_token_chunks": batched_token_chunks,
    }
    return final_response, metrics


def extract_output_ids(response: dict[str, Any]) -> list[int]:
    output_ids = response.get("output_ids")
    if output_ids is None:
        output_ids = response.get("meta_info", {}).get("output_ids")
    if not isinstance(output_ids, list) or not all(
        isinstance(token, int) for token in output_ids
    ):
        raise RuntimeError(f"Response has no integer output_ids: {response}")
    return output_ids


def replay_trajectory(
    trajectory: dict[str, Any],
    config: ReplayConfig,
    start_event: threading.Event,
    max_turns: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trajectory_id = str(trajectory["trajectory_id"])
    context_ids: list[int] = []
    round_results = []
    start_event.wait()

    turns = trajectory["turns"]
    if max_turns is not None:
        turns = turns[:max_turns]

    for turn in turns:
        turn_started = time.perf_counter()
        turn_id = int(turn["turn_id"])
        append_tokens = int(turn["append_tokens"])
        generate_tokens = int(turn["generate_tokens"])
        append_ids = synthetic_token_ids(
            append_tokens, config, trajectory_id, turn_id
        )
        prompt_ids = context_ids + append_ids
        if (
            config.max_context_tokens is not None
            and len(prompt_ids) + generate_tokens > config.max_context_tokens
        ):
            raise RuntimeError(
                f"{trajectory_id} turn {turn_id} requires "
                f"{len(prompt_ids) + generate_tokens} tokens, exceeding "
                f"--max-context-tokens={config.max_context_tokens}"
            )

        request_id = (
            f"{config.request_id_prefix}{trajectory_id}-turn-{turn_id}"
        )
        (
            response,
            latency,
            stream_metrics,
            effective_request_id,
            stream_retry_count,
        ) = post_generate(
            config, prompt_ids, generate_tokens, request_id
        )
        output_ids = extract_output_ids(response)
        if len(output_ids) != generate_tokens:
            raise RuntimeError(
                f"{trajectory_id} turn {turn_id} generated {len(output_ids)} "
                f"tokens; expected {generate_tokens}"
            )

        meta = response.get("meta_info", {})
        round_results.append(
            {
                "trajectory_id": trajectory_id,
                "turn_id": turn_id,
                "request_id": effective_request_id,
                "base_request_id": request_id,
                "stream_retry_count": stream_retry_count,
                "append_tokens": append_tokens,
                "prompt_tokens": len(prompt_ids),
                "target_generate_tokens": generate_tokens,
                "actual_generate_tokens": len(output_ids),
                "cached_tokens": int(meta.get("cached_tokens", 0)),
                "latency_seconds": latency,
                "finish_reason": meta.get("finish_reason"),
                **stream_metrics,
            }
        )
        context_ids = prompt_ids + output_ids
        turn_ended = time.perf_counter()
        turn_wall_seconds = turn_ended - turn_started
        client_runtime_seconds = max(0.0, turn_wall_seconds - latency)
        round_results[-1].update(
            {
                "turn_wall_seconds": turn_wall_seconds,
                "turn_client_runtime_seconds": client_runtime_seconds,
                "turn_accounting_error_seconds": (
                    turn_wall_seconds - latency - client_runtime_seconds
                ),
            }
        )

    elapsed_seconds = sum(
        float(item["turn_wall_seconds"]) for item in round_results
    )
    summary = {
        "trajectory_id": trajectory_id,
        "turns_completed": len(round_results),
        "final_context_tokens": len(context_ids),
        "elapsed_seconds": elapsed_seconds,
        "turn_wall_sum_seconds": elapsed_seconds,
        "request_seconds": sum(
            float(item["latency_seconds"]) for item in round_results
        ),
        "turn_client_runtime_seconds": sum(
            float(item["turn_client_runtime_seconds"]) for item in round_results
        ),
        "jct_accounting_error_seconds": sum(
            float(item["turn_accounting_error_seconds"]) for item in round_results
        ),
        "total_cached_tokens": sum(item["cached_tokens"] for item in round_results),
    }
    return round_results, summary


def load_trajectories(
    path: Path, offset: int, limit: int | None
) -> list[dict[str, Any]]:
    trajectories = []
    with path.open() as stream:
        for index, line in enumerate(stream):
            if index < offset:
                continue
            if limit is not None and len(trajectories) >= limit:
                break
            trajectories.append(json.loads(line))
    if not trajectories:
        raise ValueError("No trajectories selected")
    return trajectories


def expand_trajectories_like_paper(
    trajectories: list[dict[str, Any]],
    agent_count: int,
    seed: int,
    synthetic_append_tokens: int,
) -> list[dict[str, Any]]:
    """Expand 500 base traces using the synthetic-prefix method in Appendix A.3."""
    if agent_count <= 0:
        raise ValueError("paper agent count must be positive")
    if synthetic_append_tokens <= 0:
        raise ValueError("synthetic append length must be positive")
    rng = random.Random(seed)
    if agent_count <= len(trajectories):
        return [
            copy.deepcopy(item)
            for item in rng.sample(trajectories, agent_count)
        ]

    result = [copy.deepcopy(item) for item in trajectories]
    for clone_index in range(agent_count - len(trajectories)):
        source = trajectories[rng.randrange(len(trajectories))]
        clone = copy.deepcopy(source)
        source_id = str(source["trajectory_id"])
        clone_id = f"{source_id}-paper-aug-{clone_index:05d}"
        prefix_total = synthetic_append_tokens + 1
        shifted_turns = [
            {
                **turn,
                "turn_id": int(turn["turn_id"]) + 1,
                "prompt_context_tokens": (
                    int(turn["prompt_context_tokens"]) + prefix_total
                ),
            }
            for turn in clone["turns"]
        ]
        clone.update(
            {
                "trajectory_id": clone_id,
                "turn_count": int(source["turn_count"]) + 1,
                "total_tokens": int(source["total_tokens"]) + prefix_total,
                "mean_prompt_context_tokens": statistics.fmean(
                    [
                        synthetic_append_tokens,
                        *(
                            int(turn["prompt_context_tokens"])
                            for turn in shifted_turns
                        ),
                    ]
                ),
                "paper_augmentation": {
                    "source_trajectory_id": source_id,
                    "synthetic_append_tokens": synthetic_append_tokens,
                    "synthetic_generate_tokens": 1,
                    "sampling": "uniform_with_replacement",
                    "seed": seed,
                },
                "turns": [
                    {
                        "turn_id": 1,
                        "append_tokens": synthetic_append_tokens,
                        "generate_tokens": 1,
                        "prompt_context_tokens": synthetic_append_tokens,
                    },
                    *shifted_turns,
                ],
            }
        )
        if int(clone["total_tokens"]) > int(clone["max_agent_length"]):
            raise ValueError(
                f"{clone_id} exceeds MaxLen after paper-style augmentation"
            )
        result.append(clone)
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def metric_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50) if values else None,
        "p90": percentile(values, 0.90) if values else None,
        "p99": percentile(values, 0.99) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trajectory-offset", type=int, default=0)
    parser.add_argument("--num-trajectories", type=int)
    parser.add_argument("--paper-expand-agents", action="store_true")
    parser.add_argument("--paper-synthetic-append-tokens", type=int, default=1)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--server-selection-timeout", type=float, default=0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--request-id-prefix", default="")
    parser.add_argument("--seed", type=int, default=20260226)
    parser.add_argument("--token-id-min", type=int, default=1000)
    parser.add_argument("--token-id-max", type=int, default=30000)
    parser.add_argument("--max-context-tokens", type=int)
    args = parser.parse_args()

    if args.token_id_min < 0 or args.token_id_max <= args.token_id_min:
        parser.error("invalid token ID range")
    if args.paper_expand_agents:
        if args.trajectory_offset != 0:
            parser.error("--paper-expand-agents requires --trajectory-offset=0")
        if args.num_trajectories is None:
            parser.error("--paper-expand-agents requires --num-trajectories")
        base_trajectories = load_trajectories(args.dataset, 0, None)
        trajectories = expand_trajectories_like_paper(
            base_trajectories,
            args.num_trajectories,
            args.seed,
            args.paper_synthetic_append_tokens,
        )
    else:
        trajectories = load_trajectories(
            args.dataset, args.trajectory_offset, args.num_trajectories
        )
        if (
            args.num_trajectories is not None
            and len(trajectories) != args.num_trajectories
        ):
            parser.error(
                f"requested {args.num_trajectories} trajectories, but only "
                f"{len(trajectories)} are available; use --paper-expand-agents "
                "for Appendix A.3 expansion"
            )
    workers = args.max_workers or len(trajectories)
    workers = min(workers, len(trajectories))
    config = ReplayConfig(
        endpoint=args.base_url.rstrip("/") + "/generate",
        timeout=args.timeout,
        server_selection_timeout=args.server_selection_timeout,
        stream=args.stream,
        request_id_prefix=args.request_id_prefix,
        seed=args.seed,
        token_id_min=args.token_id_min,
        token_id_max=args.token_id_max,
        max_context_tokens=args.max_context_tokens,
    )
    start_event = threading.Event()
    all_rounds: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    errors = []
    wall_start = 0.0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                replay_trajectory,
                trajectory,
                config,
                start_event,
                args.max_turns,
            ): trajectory["trajectory_id"]
            for trajectory in trajectories
        }
        wall_start = time.perf_counter()
        start_event.set()
        for future in as_completed(futures):
            trajectory_id = futures[future]
            try:
                rounds, summary = future.result()
                all_rounds.extend(rounds)
                summaries.append(summary)
            except Exception as error:  # Preserve all agent failures.
                errors.append(
                    {
                        "trajectory_id": trajectory_id,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rounds.sort(key=lambda item: (item["trajectory_id"], item["turn_id"]))
    summaries.sort(key=lambda item: item["trajectory_id"])
    with (args.output_dir / "rounds.jsonl").open("w") as stream:
        for item in all_rounds:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
    with (args.output_dir / "trajectories.jsonl").open("w") as stream:
        for item in summaries:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")

    latencies = [float(item["latency_seconds"]) for item in all_rounds]
    summary = {
        "dataset": str(args.dataset),
        "endpoint": config.endpoint,
        "seed": args.seed,
        "trajectory_offset": args.trajectory_offset,
        "paper_expand_agents": args.paper_expand_agents,
        "paper_synthetic_append_tokens": (
            args.paper_synthetic_append_tokens
            if args.paper_expand_agents
            else None
        ),
        "paper_augmented_trajectories": sum(
            1 for item in trajectories if "paper_augmentation" in item
        ),
        "trajectories_requested": len(trajectories),
        "trajectories_completed": len(summaries),
        "turns_completed": len(all_rounds),
        "errors": errors,
        "wall_seconds": time.perf_counter() - wall_start,
        "request_latency_seconds": metric_summary(latencies),
    }
    if args.stream:
        for name in ("ttft_seconds", "ttst_seconds", "tpot_seconds"):
            values = [
                float(item[name]) for item in all_rounds if item.get(name) is not None
            ]
            summary[name] = metric_summary(values)
        all_itls = [
            float(value)
            for item in all_rounds
            for value in item.get("itl_seconds", [])
        ]
        summary["itl_seconds"] = metric_summary(all_itls)
        summary["batched_token_chunks"] = sum(
            int(item.get("batched_token_chunks", 0)) for item in all_rounds
        )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
