#!/usr/bin/env python3
"""Summarize and plot per-turn KV and effective-bandwidth metrics."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from plot_style import COLORS, add_title_suffix


KV_BYTES_PER_TOKEN = 64 * 8 * 128 * 2 * 2  # layers * KV heads * head dim * K/V * BF16
GIB = 1024**3
GB = 10**9
TITLE_SUFFIX = ""

P_TRANSFER = re.compile(
    r"DUALPATH_TTFT_P_TRANSFER(?:_V2)? rid=(?P<rid>\S+) "
    r"(?:prefill_engine=(?P<engine>\S+) )?"
    r"(?:node=(?P<node>\S+) )?.*?"
    r"transfer_s=(?P<seconds>[0-9.]+) logical_tokens=(?P<tokens>\d+)"
    r"(?: page_aligned_tokens=(?P<page_tokens>\d+) "
    r"page_count=(?P<page_count>\d+) page_size=(?P<page_size>\d+))?"
)
PREFILL_L3_READ = re.compile(
    r"DUALPATH_L3_TIMING rid=(?P<rid>\S+) "
    r"prefill_engine=(?P<engine>\S+) "
    r"(?:node=(?P<node>\S+) )?tp_rank=(?P<rank>\d+) .*?"
    r"read_start=(?P<start>[0-9.]+) read_end=(?P<end>[0-9.]+) "
    r"read_s=(?P<seconds>[0-9.]+) tokens=(?P<tokens>\d+)"
)
DECODE_L3_READ = re.compile(
    r"DUALPATH_DECODE_L3_READ rid=(?P<rid>\S+) "
    r"decode_engine=(?P<engine>\S+) "
    r"(?:node=(?P<node>\S+) )?tp_rank=(?P<rank>\d+) .*?"
    r"read_start=(?P<start>[0-9.]+) read_end=(?P<end>[0-9.]+) "
    r"read_s=(?P<seconds>[0-9.]+) tokens=(?P<tokens>\d+)"
)


def parse_logs(
    log_dir: Path,
    log_glob: str = "prefill*.log",
) -> tuple[
    dict[str, tuple[float, int, int, int, int]],
    dict[str, dict[str, dict[int, tuple[float, float, float, int, str]]]],
    dict[str, dict[str, dict[int, tuple[float, float, float, int, str]]]],
    dict[str, list[dict[str, float | int | str]]],
]:
    transfer_values: dict[
        str, list[tuple[float, int, int, int, int, str, str]]
    ] = defaultdict(list)
    read_values: dict[
        str, dict[str, dict[int, list[tuple[float, float, float, int, str]]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    decode_read_values: dict[
        str, dict[str, dict[int, list[tuple[float, float, float, int, str]]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in log_dir.rglob(log_glob):
        for line in path.read_text(errors="replace").splitlines():
            transfer_match = P_TRANSFER.search(line)
            if transfer_match:
                transfer_values[transfer_match.group("rid")].append(
                    (
                        float(transfer_match.group("seconds")),
                        int(transfer_match.group("tokens")),
                        int(transfer_match.group("page_tokens") or 0),
                        int(transfer_match.group("page_count") or 0),
                        int(transfer_match.group("page_size") or 0),
                        transfer_match.group("engine") or "unknown",
                        transfer_match.group("node") or "unknown",
                    )
                )
                continue
            read_match = PREFILL_L3_READ.search(line)
            if read_match:
                read_values[read_match.group("rid")][read_match.group("engine")][
                    int(read_match.group("rank"))
                ].append(
                    (
                        float(read_match.group("start")),
                        float(read_match.group("end")),
                        float(read_match.group("seconds")),
                        int(read_match.group("tokens")),
                        read_match.group("node") or "unknown",
                    )
                )
                continue
            read_match = DECODE_L3_READ.search(line)
            if read_match:
                decode_read_values[read_match.group("rid")][
                    read_match.group("engine")
                ][int(read_match.group("rank"))].append(
                    (
                        float(read_match.group("start")),
                        float(read_match.group("end")),
                        float(read_match.group("seconds")),
                        int(read_match.group("tokens")),
                        read_match.group("node") or "unknown",
                    )
                )

    transfer = {
        rid: (
            statistics.median(item[0] for item in items),
            round(statistics.median(item[1] for item in items)),
            round(statistics.median(item[2] for item in items)),
            round(statistics.median(item[3] for item in items)),
            round(statistics.median(item[4] for item in items)),
        )
        for rid, items in transfer_values.items()
    }
    def reduce_reads(values):
        return {
            rid: {
                engine: {
                    rank: (
                        statistics.median(item[0] for item in items),
                        statistics.median(item[1] for item in items),
                        statistics.median(item[2] for item in items),
                        round(statistics.median(item[3] for item in items)),
                        items[0][4],
                    )
                    for rank, items in ranks.items()
                }
                for engine, ranks in engines.items()
            }
            for rid, engines in values.items()
        }

    transfer_details = {
        rid: [
            {
                "seconds": statistics.median(item[0] for item in items),
                "tokens": round(statistics.median(item[1] for item in items)),
                "page_tokens": round(statistics.median(item[2] for item in items)),
                "page_count": round(statistics.median(item[3] for item in items)),
                "page_size": round(statistics.median(item[4] for item in items)),
                "engine": items[0][5],
                "node": items[0][6],
            }
            for items in [
                [item for item in values if item[5] == engine and item[6] == node]
                for engine, node in sorted({(item[5], item[6]) for item in values})
            ]
        ]
        for rid, values in transfer_values.items()
    }
    return (
        transfer,
        reduce_reads(read_values),
        reduce_reads(decode_read_values),
        transfer_details,
    )


def build_rows(
    rounds_path: Path, log_dir: Path, log_glob: str = "*.log"
) -> tuple[
    list[dict[str, float | int | str]],
    dict[str, list[dict[str, float | int | str]]],
]:
    rounds = [json.loads(line) for line in rounds_path.read_text().splitlines()]
    transfer, prefill_l3_read, decode_l3_read, transfer_details = parse_logs(
        log_dir, log_glob=log_glob
    )
    rows = []
    engine_rank_rows: dict[
        str, list[dict[str, float | int | str]]
    ] = defaultdict(list)
    def reduce_side(item, rid, side, request_engines):
        if len(request_engines) > 1:
            raise RuntimeError(
                f"request {rid} has L3 timing from multiple {side} Engines: "
                f"{sorted(request_engines)}"
            )
        engine = next(iter(request_engines), "")
        rank_timings = request_engines.get(engine, {})
        if not rank_timings:
            return engine, "unknown", rank_timings, -1, 0.0, 0.0, 0
        slowest_rank, (_, _, slowest_seconds, _, _) = max(
            rank_timings.items(), key=lambda entry: entry[1][2]
        )
        union_seconds = max(entry[1] for entry in rank_timings.values()) - min(
            entry[0] for entry in rank_timings.values()
        )
        tokens = min(entry[3] for entry in rank_timings.values())
        node = ",".join(sorted({entry[4] for entry in rank_timings.values()}))
        rank_row: dict[str, float | int | str] = {
            "trajectory_id": item["trajectory_id"],
            "turn_id": int(item["turn_id"]),
            "request_id": rid,
            "side": side,
            "engine_id": engine,
            "node": node,
            "l3_read_gib": tokens * KV_BYTES_PER_TOKEN / GIB,
        }
        for rank, (_, _, seconds, _, _) in sorted(rank_timings.items()):
            rank_row[f"tp_rank_{rank}_read_seconds"] = seconds
        rank_row["tp_slowest_read_seconds"] = slowest_seconds
        rank_row["tp_union_read_seconds"] = union_seconds
        engine_rank_rows[f"{side}:{engine}"].append(rank_row)
        return (
            engine,
            node,
            rank_timings,
            slowest_rank,
            slowest_seconds,
            union_seconds,
            tokens,
        )

    for item in rounds:
        rid = item["request_id"]
        prompt_tokens = int(item["prompt_tokens"])
        generated_tokens = int(item["actual_generate_tokens"])
        append_tokens = int(item["append_tokens"])
        cached_tokens = int(item["cached_tokens"])
        (
            transfer_seconds,
            transfer_tokens,
            transfer_page_tokens,
            transfer_page_count,
            transfer_page_size,
        ) = transfer.get(
            rid, (0.0, 0, 0, 0, 0)
        )
        (
            prefill_engine,
            prefill_node,
            prefill_rank_timings,
            prefill_slowest_rank,
            prefill_l3_seconds,
            _,
            prefill_l3_tokens,
        ) = reduce_side(item, rid, "prefill", prefill_l3_read.get(rid, {}))
        (
            decode_engine,
            decode_node,
            decode_rank_timings,
            decode_slowest_rank,
            decode_l3_seconds,
            _,
            decode_l3_tokens,
        ) = reduce_side(item, rid, "decode", decode_l3_read.get(rid, {}))
        # Page-aligned tokens estimate actual KV payload sent by all TP ranks.
        # V1 logs lack page fields and fall back to logical tokens.
        transfer_payload_tokens = transfer_page_tokens or transfer_tokens
        transfer_bytes = transfer_payload_tokens * KV_BYTES_PER_TOKEN
        prefill_l3_bytes = prefill_l3_tokens * KV_BYTES_PER_TOKEN
        decode_l3_bytes = decode_l3_tokens * KV_BYTES_PER_TOKEN
        l3_tokens = prefill_l3_tokens + decode_l3_tokens
        rows.append(
            {
                "trajectory_id": item["trajectory_id"],
                "turn_id": int(item["turn_id"]),
                "request_id": rid,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "cache_hit_rate": cached_tokens / prompt_tokens,
                "l3_read_tokens": l3_tokens,
                "l3_hit_rate": l3_tokens / prompt_tokens,
                "prefill_engine_id": prefill_engine,
                "decode_engine_id": decode_engine,
                "prefill_l3_node": prefill_node,
                "decode_l3_node": decode_node,
                "l3_tp_rank_count": len(prefill_rank_timings),
                "l3_slowest_tp_rank": prefill_slowest_rank,
                "l3_rank_read_seconds_json": json.dumps(
                    {
                        str(rank): seconds
                    for rank, (_, _, seconds, _, _) in sorted(
                        prefill_rank_timings.items()
                    )
                    },
                    separators=(",", ":"),
                ),
                "decode_l3_tp_rank_count": len(decode_rank_timings),
                "decode_l3_slowest_tp_rank": decode_slowest_rank,
                "append_tokens": append_tokens,
                "generated_tokens": generated_tokens,
                "new_logical_kv_gib": (
                    (append_tokens + generated_tokens)
                    * KV_BYTES_PER_TOKEN
                    / GIB
                ),
                "total_logical_kv_gib": (
                    (prompt_tokens + generated_tokens)
                    * KV_BYTES_PER_TOKEN
                    / GIB
                ),
                "pd_transfer_tokens": transfer_tokens,
                "pd_transfer_page_aligned_tokens": transfer_page_tokens,
                "pd_transfer_page_count": transfer_page_count,
                "pd_transfer_page_size": transfer_page_size,
                "pd_transfer_payload_bytes": transfer_bytes,
                "pd_transfer_gib": transfer_bytes / GIB,
                "pd_transfer_seconds": transfer_seconds,
                "pd_transfer_engine_id": ",".join(
                    sorted(
                        {
                            str(detail["engine"])
                            for detail in transfer_details.get(rid, [])
                        }
                    )
                ) or "unknown",
                "pd_transfer_node": ",".join(
                    sorted(
                        {
                            str(detail["node"])
                            for detail in transfer_details.get(rid, [])
                        }
                    )
                ) or "unknown",
                "pd_effective_bandwidth_gbps": (
                    transfer_bytes / transfer_seconds / GB
                    if transfer_seconds > 0
                    else 0.0
                ),
                "l3_read_seconds": prefill_l3_seconds + decode_l3_seconds,
                "prefill_l3_read_tokens": prefill_l3_tokens,
                "prefill_l3_read_gib": prefill_l3_bytes / GIB,
                "prefill_l3_read_seconds": prefill_l3_seconds,
                "prefill_l3_effective_bandwidth_gbps": (
                    prefill_l3_bytes / prefill_l3_seconds / GB
                    if prefill_l3_seconds > 0
                    else 0.0
                ),
                "decode_l3_read_tokens": decode_l3_tokens,
                "decode_l3_read_gib": decode_l3_bytes / GIB,
                "decode_l3_read_seconds": decode_l3_seconds,
                "decode_l3_effective_bandwidth_gbps": (
                    decode_l3_bytes / decode_l3_seconds / GB
                    if decode_l3_seconds > 0
                    else 0.0
                ),
            }
        )
    return rows, dict(engine_rank_rows)


def svg_text(
    x: float, y: float, value: str, size: int = 12, anchor: str = "middle"
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#1f2937">'
        f"{html.escape(value)}</text>"
    )


def plot(
    rows: list[dict[str, float | int | str]],
    series: tuple[tuple[str, str, str], ...],
    ylabel: str,
    title: str,
    output: Path,
    y_limit: float | None = None,
    canvas_size: tuple[int, int] = (1120, 410),
    legend_columns_override: int | None = None,
    legend_row_gap: float = 16.0,
    legend_x_start: float = 90.0,
    legend_y_start: float = 38.0,
    legend_column_gap: float | None = None,
    reserve_legend_space: bool = True,
) -> None:
    width, height = canvas_size
    legend_columns = legend_columns_override or min(4, len(series))
    legend_rows = (len(series) + legend_columns - 1) // legend_columns
    left, right = 70.0, width - 40.0
    top = (
        52.0 + max(0, legend_rows - 1) * legend_row_gap
        if reserve_legend_space
        else 52.0
    )
    bottom = height - 65.0
    turns = [int(row["turn_id"]) for row in rows]
    values = [
        float(row[field])
        for row in rows
        for field, _, _ in series
    ]
    y_max = y_limit if y_limit is not None else (max(values) * 1.12 or 1.0)
    x_min, x_max = min(turns), max(turns)

    def x_pos(turn: int) -> float:
        return left + (turn - x_min) / max(1, x_max - x_min) * (right - left)

    def y_pos(value: float) -> float:
        return bottom - value / y_max * (bottom - top)

    elements = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        svg_text(width / 2, 23, add_title_suffix(title, TITLE_SUFFIX), 14),
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = y_pos(value)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            'stroke="#e5e7eb"/>'
        )
        elements.append(svg_text(left - 8, y + 4, f"{value:.2f}", 10, "end"))
    for tick in range(0, len(turns), max(1, len(turns) // 10)):
        turn = turns[tick]
        elements.append(svg_text(x_pos(turn), bottom + 18, str(turn), 9))
    for index, (field, label, color) in enumerate(series):
        points = " ".join(
            f"{x_pos(int(row['turn_id'])):.1f},{y_pos(float(row[field])):.1f}"
            for row in rows
        )
        dash = ' stroke-dasharray="7 4"' if index else ""
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="2"{dash}/>'
        )
        legend_column = index % legend_columns
        legend_row = index // legend_columns
        column_gap = legend_column_gap or (
            (width - 180) / max(1, legend_columns - 1)
        )
        legend_x = legend_x_start + legend_column * column_gap
        legend_y = legend_y_start + legend_row * legend_row_gap
        elements.append(
            f'<line x1="{legend_x}" y1="{legend_y}" '
            f'x2="{legend_x + 24}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        elements.append(
            svg_text(legend_x + 31, legend_y + 4, label, 11, "start")
        )
    elements.append(svg_text(width / 2, height - 20, "Turn", 11))
    elements.append(
        f'<text x="15" y="{height / 2:.1f}" font-family="sans-serif" font-size="11" '
        f'text-anchor="middle" transform="rotate(-90 15 {height / 2:.1f})">'
        f"{html.escape(ylabel)}</text>"
    )
    output.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rounds", type=Path)
    source.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--log-glob", default="*.log")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--title-suffix", default="")
    args = parser.parse_args()
    global TITLE_SUFFIX
    TITLE_SUFFIX = args.title_suffix
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.metrics_csv:
        with args.metrics_csv.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row.setdefault("prefill_l3_node", "unknown")
            row.setdefault("decode_l3_node", "unknown")
            row.setdefault("pd_transfer_engine_id", "unknown")
            row.setdefault("pd_transfer_node", "unknown")
            row.setdefault(
                "pd_transfer_gib",
                float(row.get("pd_transfer_payload_bytes", 0.0)) / GIB,
            )
            for field in (
                "prefill_l3_read_gib",
                "decode_l3_read_gib",
                "prefill_l3_read_tokens",
                "decode_l3_read_tokens",
            ):
                row.setdefault(field, 0)
        engine_rank_rows: dict[
            str, list[dict[str, float | int | str]]
        ] = {}
    else:
        if args.log_dir is None:
            parser.error("--log-dir is required with --rounds")
        rows, engine_rank_rows = build_rows(
            args.rounds, args.log_dir, log_glob=args.log_glob
        )
    transfer_coverage = sum(
        float(row["pd_transfer_seconds"]) > 0 for row in rows
    ) / len(rows)
    if transfer_coverage < 1.0:
        raise RuntimeError(
            f"P-to-D timing coverage is only {transfer_coverage:.1%}"
        )
    csv_path = args.csv_output or args.output_dir / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plot(
        rows,
        (
            ("cache_hit_rate", "Reusable-prefix hit rate", "#2563eb"),
            ("l3_hit_rate", "KV Read (Lustre) hit rate", COLORS["orange"]),
        ),
        "Hit rate",
        "KV-cache Hit Rate by Turn",
        args.output_dir / "cache-hit.svg",
        y_limit=1.0,
        canvas_size=(600, 600),
        legend_x_start=60.0,
        legend_y_start=72.0,
        legend_column_gap=270.0,
        reserve_legend_space=False,
    )
    plot(
        rows,
        (("new_logical_kv_gib", "New logical KV", "#16a34a"),),
        "GiB",
        "New KV Data by Turn",
        args.output_dir / "kv-new.svg",
    )
    plot(
        rows,
        (("total_logical_kv_gib", "Total logical context KV", "#0f766e"),),
        "GiB",
        "Total KV Data by Turn",
        args.output_dir / "kv-total.svg",
    )
    plot(
        rows,
        (
            ("total_logical_kv_gib", "Total KV", "#0f766e"),
            ("new_logical_kv_gib", "New KV", "#16a34a"),
            ("prefill_l3_read_gib", "Prefill KV Read (Lustre)", COLORS["orange"]),
            ("decode_l3_read_gib", "Decode KV Read (Lustre)", COLORS["blue"]),
            ("pd_transfer_gib", "P→D transfer", "#dc2626"),
        ),
        "Volume [GB]",
        "KV Cache Monitor",
        args.output_dir / "kv-cache-monitor.svg",
        canvas_size=(400, 600),
        legend_columns_override=1,
        legend_row_gap=14.0,
        legend_x_start=80.0,
        legend_y_start=78.0,
        reserve_legend_space=False,
    )

    def scoped_totals(
        field: str, engine_field: str, node_field: str
    ) -> dict[str, dict[str, float | int]]:
        totals: dict[str, dict[str, float | int]] = {}
        for row in rows:
            if float(row[field]) <= 0:
                continue
            engine = str(row[engine_field])
            node = str(row[node_field])
            engine = engine or "unknown"
            node = node or "unknown"
            key = f"{node}/{engine}"
            item = totals.setdefault(key, {"node": node, "engine": engine, "turns": 0})
            item["turns"] = int(item["turns"]) + 1
            item["gib"] = float(item.get("gib", 0.0)) + float(row[field])
            item["tokens"] = int(item.get("tokens", 0)) + int(
                row[field.replace("_gib", "_tokens")]
            )
        return totals

    def with_aggregate(scopes):
        aggregate = {
            "node": "all",
            "engine": "all",
            "turns": sum(int(item["turns"]) for item in scopes.values()),
            "gib": sum(float(item.get("gib", 0.0)) for item in scopes.values()),
            "tokens": sum(int(item.get("tokens", 0)) for item in scopes.values()),
        }
        return {"aggregate": aggregate, "scopes": scopes}

    scope_summary = {
        "prefill_l3": with_aggregate(
            scoped_totals(
                "prefill_l3_read_gib", "prefill_engine_id", "prefill_l3_node"
            )
        ),
        "decode_l3": with_aggregate(
            scoped_totals(
                "decode_l3_read_gib", "decode_engine_id", "decode_l3_node"
            )
        ),
        "pd_transfer": with_aggregate(
            scoped_totals(
                "pd_transfer_gib", "pd_transfer_engine_id", "pd_transfer_node"
            )
        ),
    }
    (args.output_dir / "kv-cache-monitor-scopes.json").write_text(
        json.dumps(scope_summary, indent=2) + "\n"
    )
    for side, field, token_field, engine_field, node_field, color in (
        (
            "prefill",
            "prefill_l3_read_gib",
            "prefill_l3_read_tokens",
            "prefill_engine_id",
            "prefill_l3_node",
            COLORS["orange"],
        ),
        (
            "decode",
            "decode_l3_read_gib",
            "decode_l3_read_tokens",
            "decode_engine_id",
            "decode_l3_node",
            COLORS["blue"],
        ),
    ):
        scopes = sorted({(str(row[node_field]), str(row[engine_field])) for row in rows})
        for node, engine in scopes:
            scoped_rows = [
                row
                for row in rows
                if str(row[node_field]) == node and str(row[engine_field]) == engine
            ]
            if not any(int(row[token_field]) > 0 for row in scoped_rows):
                continue
            safe_scope = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{node}-{engine}")
            plot(
                scoped_rows,
                ((field, f"{side.title()} KV Read (Lustre)", color),),
                "Volume [GB]",
                f"{side.title()} KV Read (Lustre): {node}/{engine}",
                args.output_dir / f"l3-read-data-{side}-{safe_scope}.svg",
            )
    plot(
        rows,
        (
            (
                "pd_effective_bandwidth_gbps",
                "Completion-window logical bandwidth",
                "#dc2626",
            ),
        ),
        "GB/s",
        "Prefill-to-Decode KV Completion-window Bandwidth by Turn",
        args.output_dir / "pd-bandwidth.svg",
    )
    plot(
        rows,
        (
            (
                "prefill_l3_effective_bandwidth_gbps",
                "Prefill KV Read (Lustre)",
                COLORS["orange"],
            ),
            (
                "decode_l3_effective_bandwidth_gbps",
                "Decode KV Read (Lustre)",
                COLORS["blue"],
            ),
        ),
        "GB/s",
        "KV Read (Lustre) Bandwidth by Turn",
        args.output_dir / "l3-bandwidth.svg",
    )
    rank_colors = (
        COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"],
        COLORS["purple"], COLORS["cyan"], COLORS["brown"], COLORS["pink"],
    )
    plot(
        rows,
        (
            ("prefill_l3_read_gib", "Prefill KV Read (Lustre)", COLORS["orange"]),
            ("decode_l3_read_gib", "Decode KV Read (Lustre)", COLORS["blue"]),
        ),
        "GiB",
        "KV Read (Lustre) Data by Turn",
        args.output_dir / "l3-read-data.svg",
    )
    for side_engine, rank_rows in sorted(engine_rank_rows.items()):
        side, engine = side_engine.split(":", 1)
        for row in rank_rows:
            missing = [
                rank
                for rank in range(8)
                if f"tp_rank_{rank}_read_seconds" not in row
            ]
            if missing:
                raise RuntimeError(
                    f"{side.title()} Engine {engine} request {row['request_id']} "
                    f"is missing TP rank timings: {missing}"
                )
        rank_ids = sorted(
            {
                int(key.removeprefix("tp_rank_").removesuffix("_read_seconds"))
                for row in rank_rows
                for key in row
                if key.startswith("tp_rank_") and key.endswith("_read_seconds")
            }
        )
        if len(rank_ids) != 8:
            raise RuntimeError(
                f"{side.title()} Engine {engine} has {len(rank_ids)}/8 TP rank timings"
            )
        series = tuple(
            (
                f"tp_rank_{rank}_read_seconds",
                f"TP rank {rank}",
                rank_colors[rank],
            )
            for rank in rank_ids
        ) + (
            (
                "tp_union_read_seconds",
                "TP union window (bandwidth denominator)",
                "#111827",
            ),
        )
        safe_engine = re.sub(r"[^A-Za-z0-9_.-]+", "_", engine)
        plot(
            rank_rows,
            series,
            "Seconds",
            f"{side.title()} Replica {engine}: KV Read (Lustre) Time by TP Rank",
            args.output_dir
            / f"l3-read-time-{side}-{safe_engine}.svg",
        )
    for pattern in ("l3-read-data-engine-*.svg", "l3-read-time-engine-*.svg"):
        for legacy in args.output_dir.glob(pattern):
            legacy.unlink()
    positive_l3 = [
        float(row["prefill_l3_effective_bandwidth_gbps"])
        for row in rows
        if float(row["prefill_l3_read_seconds"]) > 0
    ]
    positive_decode_l3 = [
        float(row["decode_l3_effective_bandwidth_gbps"])
        for row in rows
        if float(row["decode_l3_read_seconds"]) > 0
    ]
    summary = {
        "turns": len(rows),
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "pd_timing_coverage": transfer_coverage,
        "l3_timing_coverage": sum(
            float(row["l3_read_seconds"]) > 0 for row in rows
        ) / len(rows),
        "first_l3_read_turn": next(
            (
                int(row["turn_id"])
                for row in rows
                if float(row["l3_read_seconds"]) > 0
            ),
            None,
        ),
        "final_cache_hit_rate": float(rows[-1]["cache_hit_rate"]),
        "final_logical_kv_gib": float(rows[-1]["total_logical_kv_gib"]),
        "median_pd_effective_bandwidth_gbps": statistics.median(
            float(row["pd_effective_bandwidth_gbps"]) for row in rows
        ),
        "median_prefill_l3_effective_bandwidth_gbps": (
            statistics.median(positive_l3) if positive_l3 else None
        ),
        "median_decode_l3_effective_bandwidth_gbps": (
            statistics.median(positive_decode_l3)
            if positive_decode_l3
            else None
        ),
        "engine_l3_timing": {
            side_engine: {
                "requests": len(rank_rows),
                "median_tp_union_read_seconds": statistics.median(
                    float(row["tp_union_read_seconds"])
                    for row in rank_rows
                ),
                "per_rank_median_read_seconds": {
                    str(rank): statistics.median(
                        float(row[f"tp_rank_{rank}_read_seconds"])
                        for row in rank_rows
                    )
                    for rank in range(8)
                },
            }
            for side_engine, rank_rows in sorted(engine_rank_rows.items())
        },
    }
    summary_path = args.summary_output or args.output_dir / "metrics-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {len(rows)} turns to {args.output_dir}")


if __name__ == "__main__":
    main()
