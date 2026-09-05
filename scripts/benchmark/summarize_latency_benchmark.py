#!/usr/bin/env python3
"""Join client streaming metrics with SGLang request timing logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUEST_PATTERN = re.compile(
    r"DUALPATH_REQ_TIMING rid=(?P<rid>\S+) type=(?P<type>\S+) "
    r"queue_s=(?P<queue>[0-9.]+) alloc_s=(?P<alloc>[0-9.]+) "
    r"prefill_s=(?P<prefill>[0-9.]+) forward_s=(?P<forward>[0-9.]+)"
)
L3_PATTERN = re.compile(
    r"DUALPATH_L3_TIMING rid=(?P<rid>\S+) "
    r"(?:prefill_engine=(?P<engine>\S+) )?tp_rank=(?P<rank>\d+) "
    r"read_start=(?P<start>[0-9.]+) read_end=(?P<end>[0-9.]+) "
    r"read_s=(?P<read>[0-9.]+) tokens=(?P<tokens>\d+)"
)
P_TRANSFER_PATTERN = re.compile(
    r"DUALPATH_TTFT_P_TRANSFER(?:_V2)? rid=(?P<rid>\S+) .*?"
    r"transfer_s=(?P<seconds>[0-9.]+) logical_tokens=(?P<tokens>\d+)"
    r"(?: page_aligned_tokens=(?P<page_tokens>\d+) "
    r"page_count=(?P<page_count>\d+) page_size=(?P<page_size>\d+))?"
)
PREFILL_RELEASE_PATTERN = re.compile(
    r"DUALPATH_PREFILL_CACHE_RELEASE rid=(?P<rid>\S+) "
    r"evictable_before=(?P<before>\d+) evictable_after=(?P<after>\d+)"
)
KV_BYTES_PER_TOKEN = 64 * 8 * 128 * 2 * 2


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def parse_timing_logs(
    log_dir: Path,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    request_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    l3_values: dict[
        str, dict[str, dict[int, dict[str, list[float]]]]
    ] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    transfer_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    release_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in log_dir.glob("prefill*.log"):
        for line in path.read_text(errors="replace").splitlines():
            match = REQUEST_PATTERN.search(line)
            if match and match.group("type").lower() == "prefill":
                rid = match.group("rid")
                for name in ("queue", "alloc", "prefill", "forward"):
                    request_values[rid][name].append(float(match.group(name)))
            match = L3_PATTERN.search(line)
            if match:
                rid = match.group("rid")
                engine = match.group("engine") or "0"
                rank = int(match.group("rank"))
                l3_values[rid][engine][rank]["read"].append(
                    float(match.group("read"))
                )
                l3_values[rid][engine][rank]["start"].append(
                    float(match.group("start"))
                )
                l3_values[rid][engine][rank]["end"].append(
                    float(match.group("end"))
                )
                l3_values[rid][engine][rank]["tokens"].append(
                    float(match.group("tokens"))
                )
            match = P_TRANSFER_PATTERN.search(line)
            if match:
                rid = match.group("rid")
                transfer_values[rid]["seconds"].append(
                    float(match.group("seconds"))
                )
                transfer_values[rid]["tokens"].append(
                    float(match.group("tokens"))
                )
                transfer_values[rid]["page_tokens"].append(
                    float(match.group("page_tokens") or 0)
                )
                transfer_values[rid]["page_count"].append(
                    float(match.group("page_count") or 0)
                )
                transfer_values[rid]["page_size"].append(
                    float(match.group("page_size") or 0)
                )
            match = PREFILL_RELEASE_PATTERN.search(line)
            if match:
                rid = match.group("rid")
                release_values[rid]["before"].append(
                    float(match.group("before"))
                )
                release_values[rid]["after"].append(
                    float(match.group("after"))
                )

    requests = {
        rid: {name: statistics.median(values) for name, values in fields.items()}
        for rid, fields in request_values.items()
    }
    l3 = {}
    for rid, engines in l3_values.items():
        if len(engines) != 1:
            raise RuntimeError(
                f"{rid} has L3 timing from Prefill Engines {sorted(engines)}"
            )
        engine, ranks = next(iter(engines.items()))
        if len(ranks) != 8:
            raise RuntimeError(
                f"{rid} on Prefill Engine {engine} has {len(ranks)}/8 "
                "TP rank timings"
            )
        reduced = {
            rank: {
                name: statistics.median(values)
                for name, values in fields.items()
            }
            for rank, fields in ranks.items()
        }
        slowest_rank, slowest = max(
            reduced.items(), key=lambda item: item[1]["read"]
        )
        l3[rid] = {
            "read": max(item["end"] for item in reduced.values())
            - min(item["start"] for item in reduced.values()),
            "tokens": min(item["tokens"] for item in reduced.values()),
            "engine": engine,
            "slowest_rank": slowest_rank,
            "rank_count": len(reduced),
        }
    transfers = {
        rid: {
            name: statistics.median(values)
            for name, values in fields.items()
        }
        for rid, fields in transfer_values.items()
    }
    releases = {
        rid: {
            name: statistics.median(values)
            for name, values in fields.items()
        }
        for rid, fields in release_values.items()
    }
    return requests, l3, transfers, releases


def load_runs(result_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for run_path in sorted(result_dir.glob("*/agents-*/rep-*/run.json")):
        run_dir = run_path.parent
        metadata = json.loads(run_path.read_text())
        summary_path = run_dir / "summary.json"
        rounds_path = run_dir / "rounds.jsonl"
        if not summary_path.exists() or not rounds_path.exists():
            print(f"Skipping incomplete run directory: {run_dir}")
            continue
        summary = json.loads(summary_path.read_text())
        rounds = [
            json.loads(line)
            for line in rounds_path.read_text().splitlines()
        ]
        configured_turns = int(metadata["turns"])
        expected_rounds = (
            int(metadata["agents"]) * configured_turns
            if configured_turns > 0
            else int(summary["turns_completed"])
        )
        if (
            summary["errors"]
            or int(summary["trajectories_completed"]) != int(metadata["agents"])
            or summary["turns_completed"] != expected_rounds
        ):
            raise RuntimeError(f"Incomplete run {run_dir}: {summary}")
        if len(rounds) != expected_rounds:
            raise RuntimeError(
                f"Run {run_dir} has {len(rounds)} rounds, expected {expected_rounds}"
            )
        if any(
            int(item["batched_token_chunks"]) != 0
            or int(item["streamed_token_count"])
            != int(item["actual_generate_tokens"])
            for item in rounds
        ):
            raise RuntimeError(f"Run {run_dir} has invalid streaming token chunks")
        runs.append(
            {
                "directory": run_dir,
                "metadata": metadata,
                "summary": summary,
                "rounds": rounds,
            }
        )
    if not runs:
        raise RuntimeError(f"No measured runs found under {result_dir}")
    return runs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    request_timing, l3_timing, pd_timing, release_timing = parse_timing_logs(
        args.log_dir
    )
    runs = load_runs(args.result_dir)
    merged_rounds = []
    run_rows = []

    for run in runs:
        metadata = run["metadata"]
        rounds = run["rounds"]
        for item in rounds:
            rid = item["request_id"]
            request = request_timing.get(rid, {})
            l3 = l3_timing.get(rid, {})
            pd = pd_timing.get(rid, {})
            release = release_timing.get(rid, {})
            pd_payload_tokens = int(
                pd.get("page_tokens", 0) or pd.get("tokens", 0)
            )
            sch = float(request.get("queue", 0.0))
            alloc = float(request.get("alloc", 0.0))
            read = float(l3.get("read", 0.0))
            prefill = float(request.get("prefill", 0.0))
            ttft = float(item["ttft_seconds"])
            stage_sum = sch + alloc + read + prefill
            merged_rounds.append(
                {
                    **metadata,
                    **item,
                    "sch_seconds": sch,
                    "alloc_seconds": alloc,
                    "read_seconds": read,
                    "prefill_seconds": prefill,
                    "residual_seconds": max(0.0, ttft - stage_sum),
                    "overlap_excess_seconds": max(0.0, stage_sum - ttft),
                    "l3_prefetched_tokens_tp0": int(l3.get("tokens", 0)),
                    "l3_prefetched_tokens_logical": int(l3.get("tokens", 0)),
                    "l3_prefill_engine_id": l3.get("engine", ""),
                    "l3_slowest_tp_rank": int(l3.get("slowest_rank", -1)),
                    "l3_tp_rank_count": int(l3.get("rank_count", 0)),
                    "has_request_timing": bool(request),
                    "has_l3_timing": bool(l3),
                    "pd_transfer_seconds": float(pd.get("seconds", 0.0)),
                    "pd_transfer_tokens": int(pd.get("tokens", 0)),
                    "pd_transfer_page_aligned_tokens": int(
                        pd.get("page_tokens", 0)
                    ),
                    "pd_transfer_page_count": int(pd.get("page_count", 0)),
                    "pd_transfer_page_size": int(pd.get("page_size", 0)),
                    "pd_transfer_payload_bytes": (
                        pd_payload_tokens * KV_BYTES_PER_TOKEN
                    ),
                    "pd_effective_bandwidth_gbps": (
                        float(pd_payload_tokens)
                        * KV_BYTES_PER_TOKEN
                        / float(pd["seconds"])
                        / 1e9
                        if float(pd.get("seconds", 0.0)) > 0
                        else 0.0
                    ),
                    "cache_hit_rate": (
                        int(item["cached_tokens"]) / int(item["prompt_tokens"])
                        if int(item["prompt_tokens"]) > 0
                        else 0.0
                    ),
                    "has_pd_timing": bool(pd),
                    "has_prefill_cache_release": bool(release),
                    "prefill_evictable_before_release": int(
                        release.get("before", 0)
                    ),
                    "prefill_evictable_after_release": int(
                        release.get("after", 0)
                    ),
                }
            )

        def values(name: str) -> list[float]:
            return [float(item[name]) for item in merged_rounds if (
                item["max_len_k"] == metadata["max_len_k"]
                and item["agents"] == metadata["agents"]
                and item["repetition"] == metadata["repetition"]
            ) and item.get(name) is not None]

        run_rows.append(
            {
                **metadata,
                "jct_seconds": float(run["summary"]["wall_seconds"]),
                "ttft_mean_seconds": mean(values("ttft_seconds")),
                "ttst_mean_seconds": mean(values("ttst_seconds")),
                "tpot_mean_seconds": mean(values("tpot_seconds")),
                "sch_mean_seconds": mean(values("sch_seconds")),
                "alloc_mean_seconds": mean(values("alloc_seconds")),
                "read_mean_seconds": mean(values("read_seconds")),
                "prefill_mean_seconds": mean(values("prefill_seconds")),
                "residual_mean_seconds": mean(values("residual_seconds")),
                "prompt_tokens_min": min(int(item["prompt_tokens"]) for item in rounds),
                "prompt_tokens_max": max(int(item["prompt_tokens"]) for item in rounds),
                "cached_tokens_min": min(int(item["cached_tokens"]) for item in rounds),
                "cached_tokens_max": max(int(item["cached_tokens"]) for item in rounds),
                "stream_retry_count": sum(
                    int(item.get("stream_retry_count", 0)) for item in rounds
                ),
                "request_timing_coverage": mean(
                    [1.0 if item["has_request_timing"] else 0.0 for item in merged_rounds
                     if item["max_len_k"] == metadata["max_len_k"]
                     and item["agents"] == metadata["agents"]
                     and item["repetition"] == metadata["repetition"]]
                ),
                "l3_timing_coverage": mean(
                    [1.0 if item["has_l3_timing"] else 0.0 for item in merged_rounds
                     if item["max_len_k"] == metadata["max_len_k"]
                     and item["agents"] == metadata["agents"]
                     and item["repetition"] == metadata["repetition"]]
                ),
                "pd_timing_coverage": mean(
                    [1.0 if item["has_pd_timing"] else 0.0 for item in merged_rounds
                     if item["max_len_k"] == metadata["max_len_k"]
                     and item["agents"] == metadata["agents"]
                     and item["repetition"] == metadata["repetition"]]
                ),
                "prefill_cache_release_coverage": mean(
                    [1.0 if item["has_prefill_cache_release"] else 0.0
                     for item in merged_rounds
                     if item["max_len_k"] == metadata["max_len_k"]
                     and item["agents"] == metadata["agents"]
                     and item["repetition"] == metadata["repetition"]]
                ),
                "prefill_evictable_after_release_max": max(
                    int(item["prefill_evictable_after_release"])
                    for item in merged_rounds
                    if item["max_len_k"] == metadata["max_len_k"]
                    and item["agents"] == metadata["agents"]
                    and item["repetition"] == metadata["repetition"]
                ),
                "prefill_evictable_before_release_max": max(
                    int(item["prefill_evictable_before_release"])
                    for item in merged_rounds
                    if item["max_len_k"] == metadata["max_len_k"]
                    and item["agents"] == metadata["agents"]
                    and item["repetition"] == metadata["repetition"]
                ),
                "pd_transfer_mean_seconds": mean(values("pd_transfer_seconds")),
                "pd_effective_bandwidth_mean_gbps": mean(
                    values("pd_effective_bandwidth_gbps")
                ),
                "cache_hit_rate_mean": mean(values("cache_hit_rate")),
            }
        )

    with (args.output_dir / "merged_rounds.jsonl").open("w") as stream:
        for item in merged_rounds:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
    write_csv(args.output_dir / "runs.csv", run_rows)

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["max_len_k"], row["agents"])].append(row)
    aggregate_rows = []
    metric_names = [
        name for name in run_rows[0]
        if name.endswith("_seconds")
        or name.endswith("_coverage")
        or name.endswith("_gbps")
        or name.endswith("_rate_mean")
    ]
    for (max_len, agents), rows in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "max_len_k": max_len,
            "agents": agents,
            "ablation_profile": rows[0].get("ablation_profile", "legacy-full"),
            "repetitions": len(rows),
            "prompt_tokens_min": min(row["prompt_tokens_min"] for row in rows),
            "prompt_tokens_max": max(row["prompt_tokens_max"] for row in rows),
        }
        for name in metric_names:
            vals = [float(row[name]) for row in rows]
            aggregate[f"{name}_median"] = statistics.median(vals)
            aggregate[f"{name}_min"] = min(vals)
            aggregate[f"{name}_max"] = max(vals)
        aggregate_rows.append(aggregate)
    write_csv(args.output_dir / "aggregate.csv", aggregate_rows)

    expects_l3_timing = any(
        bool(run["metadata"].get("prefill_l3", True)) for run in runs
    )
    expects_prefill_release = any(
        bool(
            run["metadata"].get(
                "prefill_release_local_cache_after_pd_ack", False
            )
        )
        for run in runs
    )
    expects_l3_reuse = any(
        bool(run["metadata"].get("prefill_l3", True))
        and bool(run["metadata"].get("decode_offload", True))
        and len(run["rounds"]) > 1
        for run in runs
    )
    validation = {
        "runs": len(run_rows),
        "rounds": len(merged_rounds),
        "expects_l3_timing": expects_l3_timing,
        "expects_l3_reuse": expects_l3_reuse,
        "request_timing_coverage": mean(
            [1.0 if item["has_request_timing"] else 0.0 for item in merged_rounds]
        ),
        "l3_timing_coverage": mean(
            [1.0 if item["has_l3_timing"] else 0.0 for item in merged_rounds]
        ),
        "pd_timing_coverage": mean(
            [1.0 if item["has_pd_timing"] else 0.0 for item in merged_rounds]
        ),
        "expects_prefill_cache_release": expects_prefill_release,
        "prefill_cache_release_coverage": mean(
            [1.0 if item["has_prefill_cache_release"] else 0.0
             for item in merged_rounds]
        ),
        "prefill_evictable_after_release_max": max(
            int(item["prefill_evictable_after_release"])
            for item in merged_rounds
        ),
        "prefill_evictable_before_release_max": max(
            int(item["prefill_evictable_before_release"])
            for item in merged_rounds
        ),
        "batched_token_chunks": sum(
            int(item.get("batched_token_chunks", 0)) for item in merged_rounds
        ),
        "stream_retry_count": sum(
            int(item.get("stream_retry_count", 0)) for item in merged_rounds
        ),
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    if validation["request_timing_coverage"] < 0.95:
        raise RuntimeError(f"Low request timing coverage: {validation}")
    # Dynamic full rollout does not pre-populate or wait for Store publication.
    # Short traces may therefore have zero L3 reads even when the backend is
    # enabled; coverage is an observed metric, not a correctness threshold.
    if not expects_l3_timing and validation["l3_timing_coverage"] != 0:
        raise RuntimeError(f"Unexpected L3 timing for no-L3 profile: {validation}")
    if validation["pd_timing_coverage"] < 0.95:
        raise RuntimeError(f"Low P-to-D timing coverage: {validation}")
    if (
        expects_prefill_release
        and validation["prefill_cache_release_coverage"] < 0.95
    ):
        raise RuntimeError(f"Low Prefill cache release coverage: {validation}")
    if (
        expects_prefill_release
        and validation["prefill_evictable_after_release_max"] != 0
    ):
        raise RuntimeError(f"Incomplete Prefill cache eviction: {validation}")
    if (
        expects_prefill_release
        and validation["prefill_evictable_before_release_max"] == 0
    ):
        raise RuntimeError(f"Prefill cache release evicted no pages: {validation}")
    if expects_l3_reuse and validation["l3_timing_coverage"] == 0:
        raise RuntimeError(f"Expected cross-turn L3 reuse was not observed: {validation}")
    if validation["batched_token_chunks"]:
        raise RuntimeError(f"Streaming chunks contained multiple tokens: {validation}")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
