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
    r"(?:prefill_engine=(?P<engine>\S+) )?"
    r"(?:node=(?P<node>\S+) )?tp_rank=(?P<rank>\d+) "
    r"read_start=(?P<start>[0-9.]+) read_end=(?P<end>[0-9.]+) "
    r"read_s=(?P<read>[0-9.]+) tokens=(?P<tokens>\d+)"
)
DECODE_L3_PATTERN = re.compile(
    r"DUALPATH_DECODE_L3_READ rid=(?P<rid>\S+) "
    r"decode_engine=(?P<engine>\S+) "
    r"(?:node=(?P<node>\S+) )?tp_rank=(?P<rank>\d+) "
    r"read_start=(?P<start>[0-9.]+) read_end=(?P<end>[0-9.]+) "
    r"read_s=(?P<read>[0-9.]+) tokens=(?P<tokens>\d+)"
)
P_TRANSFER_PATTERN = re.compile(
    r"DUALPATH_TTFT_P_TRANSFER(?:_V2)? rid=(?P<rid>\S+) "
    r"(?:prefill_engine=(?P<engine>\S+) )?"
    r"(?:node=(?P<node>\S+) )?.*?"
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


def l3_cached_token_coverage(rows: list[dict[str, Any]]) -> float:
    reusable = [item for item in rows if int(item["cached_tokens"]) > 0]
    cached_tokens = sum(int(item["cached_tokens"]) for item in reusable)
    if cached_tokens == 0:
        return 1.0
    l3_tokens = sum(
        min(
            int(item["cached_tokens"]),
            int(item["l3_read_tokens_logical"]),
        )
        for item in reusable
    )
    return l3_tokens / cached_tokens


def parse_timing_logs(
    log_dir: Path,
) -> tuple[
    dict[str, dict[str, float]],
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
    decode_l3_values: dict[
        str, dict[str, dict[int, dict[str, list[float]]]]
    ] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    l3_nodes: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    decode_l3_nodes: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    transfer_values: dict[
        str, dict[tuple[str, str], dict[str, list[float]]]
    ] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    release_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in log_dir.rglob("*.log"):
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
                l3_nodes[rid][engine].add(match.group("node") or "unknown")
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
            match = DECODE_L3_PATTERN.search(line)
            if match:
                rid = match.group("rid")
                engine = match.group("engine")
                decode_l3_nodes[rid][engine].add(match.group("node") or "unknown")
                rank = int(match.group("rank"))
                for name in ("read", "start", "end", "tokens"):
                    decode_l3_values[rid][engine][rank][name].append(
                        float(match.group(name))
                    )
            match = P_TRANSFER_PATTERN.search(line)
            if match:
                rid = match.group("rid")
                key = (
                    match.group("engine") or "unknown",
                    match.group("node") or "unknown",
                )
                for name in ("seconds", "tokens", "page_tokens", "page_count", "page_size"):
                    transfer_values[rid][key][name].append(
                        float(match.group(name) or 0)
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
    def reduce_l3(
        values: dict[str, dict[str, dict[int, dict[str, list[float]]]]],
        nodes: dict[str, dict[str, set[str]]],
        side: str,
    ) -> dict[str, dict[str, float]]:
        result = {}
        for rid, engines in values.items():
            if len(engines) != 1:
                raise RuntimeError(
                    f"{rid} has L3 timing from {side} Engines {sorted(engines)}"
                )
            engine, ranks = next(iter(engines.items()))
            if len(ranks) != 8:
                raise RuntimeError(
                    f"{rid} on {side} Engine {engine} has {len(ranks)}/8 "
                    "TP rank timings"
                )
            reduced = {
                rank: {
                    name: statistics.median(field_values)
                    for name, field_values in fields.items()
                }
                for rank, fields in ranks.items()
            }
            slowest_rank, _ = max(
                reduced.items(), key=lambda item: item[1]["read"]
            )
            result[rid] = {
                "read": max(item["end"] for item in reduced.values())
                - min(item["start"] for item in reduced.values()),
                "tokens": min(item["tokens"] for item in reduced.values()),
                "engine": engine,
                "node": ",".join(sorted(nodes[rid][engine])) or "unknown",
                "slowest_rank": slowest_rank,
                "rank_count": len(reduced),
            }
        return result

    l3 = reduce_l3(l3_values, l3_nodes, "Prefill")
    decode_l3 = reduce_l3(decode_l3_values, decode_l3_nodes, "Decode")
    transfers = {}
    for rid, engines in transfer_values.items():
        reduced = {
            key: {
                name: statistics.median(values)
                for name, values in fields.items()
            }
            for key, fields in engines.items()
        }
        transfers[rid] = {
            name: sum(item[name] for item in reduced.values())
            for name in ("tokens", "page_tokens", "page_count")
        }
        transfers[rid]["seconds"] = max(
            item["seconds"] for item in reduced.values()
        )
        transfers[rid]["engine"] = ",".join(sorted(key[0] for key in reduced))
        transfers[rid]["node"] = ",".join(sorted(key[1] for key in reduced))
        transfers[rid]["by_engine"] = reduced
    releases = {
        rid: {
            name: statistics.median(values)
            for name, values in fields.items()
        }
        for rid, fields in release_values.items()
    }
    return requests, l3, decode_l3, transfers, releases


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
            int(item["streamed_token_count"])
            != int(item["actual_generate_tokens"])
            for item in rounds
        ):
            raise RuntimeError(f"Run {run_dir} has mismatched streaming token counts")
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

    (
        request_timing,
        prefill_l3_timing,
        decode_l3_timing,
        pd_timing,
        release_timing,
    ) = parse_timing_logs(args.log_dir)
    runs = load_runs(args.result_dir)
    merged_rounds = []
    run_rows = []

    for run in runs:
        metadata = run["metadata"]
        rounds = run["rounds"]
        for item in rounds:
            rid = item["request_id"]
            request = request_timing.get(rid, {})
            prefill_l3 = prefill_l3_timing.get(rid, {})
            decode_l3 = decode_l3_timing.get(rid, {})
            pd = pd_timing.get(rid, {})
            release = release_timing.get(rid, {})
            pd_payload_tokens = int(
                pd.get("page_tokens", 0) or pd.get("tokens", 0)
            )
            sch = float(request.get("queue", 0.0))
            alloc = float(request.get("alloc", 0.0))
            prefill_read = float(prefill_l3.get("read", 0.0))
            decode_read = float(decode_l3.get("read", 0.0))
            read = prefill_read + decode_read
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
                    "prefill_l3_read_seconds": prefill_read,
                    "prefill_l3_read_tokens": int(prefill_l3.get("tokens", 0)),
                    "prefill_l3_read_bytes": int(prefill_l3.get("tokens", 0))
                    * KV_BYTES_PER_TOKEN,
                    "prefill_l3_effective_bandwidth_gbps": (
                        int(prefill_l3.get("tokens", 0))
                        * KV_BYTES_PER_TOKEN
                        / prefill_read
                        / 1e9
                        if prefill_read > 0
                        else 0.0
                    ),
                    "decode_l3_read_seconds": decode_read,
                    "decode_l3_read_tokens": int(decode_l3.get("tokens", 0)),
                    "decode_l3_read_bytes": int(decode_l3.get("tokens", 0))
                    * KV_BYTES_PER_TOKEN,
                    "decode_l3_effective_bandwidth_gbps": (
                        int(decode_l3.get("tokens", 0))
                        * KV_BYTES_PER_TOKEN
                        / decode_read
                        / 1e9
                        if decode_read > 0
                        else 0.0
                    ),
                    "l3_read_tokens_logical": int(prefill_l3.get("tokens", 0))
                    + int(decode_l3.get("tokens", 0)),
                    "l3_prefetched_tokens_tp0": int(prefill_l3.get("tokens", 0)),
                    "l3_prefetched_tokens_logical": int(prefill_l3.get("tokens", 0)),
                    "l3_prefill_engine_id": prefill_l3.get("engine", ""),
                    "l3_decode_engine_id": decode_l3.get("engine", ""),
                    "prefill_l3_node": prefill_l3.get("node", "unknown"),
                    "decode_l3_node": decode_l3.get("node", "unknown"),
                    "l3_slowest_tp_rank": int(prefill_l3.get("slowest_rank", -1)),
                    "l3_tp_rank_count": int(prefill_l3.get("rank_count", 0)),
                    "decode_l3_slowest_tp_rank": int(
                        decode_l3.get("slowest_rank", -1)
                    ),
                    "decode_l3_tp_rank_count": int(decode_l3.get("rank_count", 0)),
                    "has_request_timing": bool(request),
                    "has_l3_timing": bool(prefill_l3) or bool(decode_l3),
                    "has_prefill_l3_timing": bool(prefill_l3),
                    "has_decode_l3_timing": bool(decode_l3),
                    "pd_transfer_seconds": float(pd.get("seconds", 0.0)),
                    "pd_transfer_tokens": int(pd.get("tokens", 0)),
                    "pd_transfer_engine_id": pd.get("engine", "unknown"),
                    "pd_transfer_node": pd.get("node", "unknown"),
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
                "prefill_l3_read_mean_seconds": mean(
                    values("prefill_l3_read_seconds")
                ),
                "decode_l3_read_mean_seconds": mean(
                    values("decode_l3_read_seconds")
                ),
                "prefill_l3_effective_bandwidth_mean_gbps": mean(
                    values("prefill_l3_effective_bandwidth_gbps")
                ),
                "decode_l3_effective_bandwidth_mean_gbps": mean(
                    values("decode_l3_effective_bandwidth_gbps")
                ),
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
                "prefill_l3_cached_token_coverage": l3_cached_token_coverage(
                    [
                        item
                        for item in merged_rounds
                        if item["max_len_k"] == metadata["max_len_k"]
                        and item["agents"] == metadata["agents"]
                        and item["repetition"] == metadata["repetition"]
                    ]
                ),
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
    deferred_evict_rows = [
        item
        for item in merged_rounds
        if bool(item.get("prefill_l3", True))
        and bool(item.get("decode_offload", True))
        and bool(item.get("prefill_hicache_evict_unlocked", False))
    ]
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
        "prefill_l3_cached_token_coverage": l3_cached_token_coverage(
            deferred_evict_rows
        ),
        "prefill_local_prefix_reuse_rounds": sum(
            int(item["cached_tokens"]) > 0
            and int(item["l3_read_tokens_logical"])
            < int(item["cached_tokens"])
            for item in deferred_evict_rows
        ),
        "l3_read_paths": sorted(
            {str(item.get("l3_read_path", "prefill")) for item in merged_rounds}
        ),
        "prefill_l3_read_tokens_total": sum(
            int(item["prefill_l3_read_tokens"]) for item in merged_rounds
        ),
        "decode_l3_read_tokens_total": sum(
            int(item["decode_l3_read_tokens"]) for item in merged_rounds
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
    if expects_l3_reuse and validation["l3_timing_coverage"] == 0:
        raise RuntimeError(f"Expected cross-turn L3 reuse was not observed: {validation}")
    if (
        deferred_evict_rows
        and validation["prefill_l3_cached_token_coverage"] < 0.99
    ):
        raise RuntimeError(
            f"Unexpected Prefill local-prefix reuse after deferred eviction: {validation}"
        )
    if validation["l3_read_paths"] == ["prefill"]:
        if validation["decode_l3_read_tokens_total"] != 0:
            raise RuntimeError(
                f"Decode performed L3 reads on the Prefill-read branch: {validation}"
            )
    elif validation["l3_read_paths"] == ["decode"]:
        if validation["prefill_l3_read_tokens_total"] != 0:
            raise RuntimeError(
                f"Prefill performed L3 reads on the Decode-read branch: {validation}"
            )
    else:
        raise RuntimeError(f"Mixed or invalid L3 read paths: {validation}")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
