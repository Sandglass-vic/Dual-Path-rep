#!/usr/bin/env python3
"""Do real Decode-path runs show D->P interference on Decode ITL?

Rebuilds a g6 (Decode) monotonic-clock timeline from the Decode server log and
the benchmark's fullchain rounds:

- D->P window of a request: all 4 ranks finish L3 read + scatter
  (``decode_before_read_end_ns + lustre_read_raw + scatter_raw``; the TP sync
  releases on the slowest rank), plus ``decode_l3_sync_wait_seconds``, then
  ``d_to_p_transfer_raw_seconds``.
- Token arrival times of a request: ``decode_dispatch_ns`` plus the client's
  cumulative ITL (the client runs on g6 too).

Each ITL interval is classified by its overlap with other requests' D->P
windows, and compared with a Prefill-path run of the same setup (no D->P).

With ``--prefill-log`` the windows are also mapped onto the g5 clock (offset
from ``p_to_d_end_ns`` vs the Decode-side arrival of the same request) to check
the window placement against each request's own forward start, and to compare
Prefill forwards that overlap other requests' D->P with those that do not.

    analyze_real_d2p.py <decode-run-result-dir> <decode-server-log>
        [--prefill-log <prefill-server-log>] [--baseline <prefill-path-result-dir>]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def rounds(result_dir: Path) -> list[dict]:
    path = next(result_dir.glob("32K/agents-*/rep-*/data/fullchain/fullchain_rounds.jsonl"))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def decode_records(log: Path) -> dict[str, list[dict]]:
    per_rid: dict[str, list[dict]] = defaultdict(list)
    for line in log.open(errors="ignore"):
        if "DUALPATH_TTFT_V3" not in line:
            continue
        record = json.loads(line[line.index("{"):])
        if record.get("side") == "decode" and "decode_dispatch_ns" in record:
            per_rid[record["rid"]].append(record)
    return per_rid


def prefill_records(log: Path) -> dict[str, dict[int, dict]]:
    per_rid: dict[str, dict[int, dict]] = defaultdict(dict)
    for line in log.open(errors="ignore"):
        if "DUALPATH_TTFT_V3" not in line:
            continue
        record = json.loads(line[line.index("{"):])
        if record.get("side") == "prefill" and record.get("prefill_pipeline_end_ns"):
            per_rid[record["rid"]][record["tp_rank"]] = record
    return per_rid


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def describe(label: str, itls: list[float]) -> None:
    if not itls:
        print(f"  {label:34s} n=0")
        return
    ms = [v * 1e3 for v in itls]
    print(f"  {label:34s} n={len(ms):6d} mean={statistics.mean(ms):6.1f} p50={quantile(ms, .5):6.1f} "
          f"p90={quantile(ms, .9):6.1f} p99={quantile(ms, .99):6.1f} ms  >60ms={sum(v > 60 for v in ms) / len(ms) * 100:5.2f}%")


def check_prefill(rows: dict, records: dict, windows: list, log: Path) -> None:
    prefill = prefill_records(log)
    offsets = []
    for rid, recs in records.items():
        row = rows.get(rid)
        if not row or rid not in prefill:
            continue
        arrival = (min(r["decode_dispatch_ns"] for r in recs) / 1e9
                   - row["decode_host_to_device_raw_seconds"] - row["decode_dispatch_raw_seconds"])
        offsets.append(arrival - max(p["p_to_d_end_ns"] for p in prefill[rid].values()) / 1e9)
    offset = statistics.median(offsets)
    print(f"  g6-g5 clock offset {offset:.3f} s, p10..p90 spread "
          f"{(quantile(offsets, .1) - offset) * 1e3:+.0f}..{(quantile(offsets, .9) - offset) * 1e3:+.0f} ms")

    forwards = {}
    for rid, ranks in prefill.items():
        if rid in rows:
            end = max(p["prefill_pipeline_end_ns"] for p in ranks.values()) / 1e9
            forwards[rid] = (end - rows[rid]["prefill_computation_raw_seconds"], end)
    g5_windows = [(w0 - offset, w1 - offset, wid) for w0, w1, wid in windows]

    lead = [(forwards[wid][0] - w1) * 1e3 for _, w1, wid in g5_windows if wid in forwards]
    print(f"  own forward start - own D->P end: p10={quantile(lead, .1):.0f} p50={quantile(lead, .5):.0f} "
          f"p90={quantile(lead, .9):.0f} ms")
    spans = sorted(forwards.values())
    busy = sum(b - a for a, b in spans) / (spans[-1][1] - spans[0][0])
    print(f"  Prefill forward busy {busy * 100:.0f}% of wall time")

    overlapped, clear = [], []
    for rid, (begin, end) in forwards.items():
        row = rows[rid]
        if row["turn_id"] < 3 or row["append_tokens"] < 200:
            continue
        shared = sum(max(0.0, min(end, w1) - max(begin, w0)) for w0, w1, wid in g5_windows if wid != rid)
        per_token = (end - begin) / row["append_tokens"] * 1e3
        (overlapped if shared > 0.05 * (end - begin) else clear).append(per_token)
    for label, values in (("forward overlapped by D->P (>5%)", overlapped), ("forward not overlapped", clear)):
        if values:
            print(f"  {label:34s} n={len(values):4d} ms/token p50={quantile(values, .5):.3f} "
                  f"mean={statistics.mean(values):.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("decode_log", type=Path)
    parser.add_argument("--prefill-log", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir
    rows = {r["request_id"]: r for r in rounds(run_dir)}
    records = decode_records(args.decode_log)

    windows = []
    for rid, recs in records.items():
        ranks = {r["tp_rank"]: r for r in recs}
        if len(ranks) < 4 or not any(r.get("d_to_p_transfer_raw_seconds", 0) > 0 for r in ranks.values()):
            continue
        ready = max(
            r["decode_before_read_end_ns"] / 1e9 + r.get("decode_lustre_read_raw_seconds", 0.0)
            + r.get("decode_scatter_raw_seconds", 0.0)
            for r in ranks.values()
        )
        ready += rows[rid].get("decode_l3_sync_wait_seconds", 0.0) if rid in rows else 0.0
        duration = max(r.get("d_to_p_transfer_raw_seconds", 0.0) for r in ranks.values())
        windows.append((ready, ready + duration, rid))
    windows.sort()
    print(f"{run_dir.name}: {len(windows)} D->P windows, mean {statistics.mean(w[1] - w[0] for w in windows) * 1e3:.1f} ms")

    overlapped, clear, near = [], [], []
    for rid, row in rows.items():
        recs = records.get(rid)
        if not recs or row["turn_id"] < 3:
            continue
        start = min(r["decode_dispatch_ns"] for r in recs) / 1e9
        t = start
        for itl in row["itl_seconds"][1:]:
            begin, end = t, t + itl
            t = end
            hit = any(w0 < end and w1 > begin and wid != rid for w0, w1, wid in windows)
            close = any(w0 - 0.3 < end and w1 + 0.3 > begin and wid != rid for w0, w1, wid in windows)
            (overlapped if hit else near if close else clear).append(itl)
    describe("ITL overlapping a D->P window", overlapped)
    describe("ITL within 300 ms of one (not on)", near)
    describe("ITL away from any D->P", clear)

    describe("all ITL", overlapped + near + clear)

    if args.prefill_log:
        check_prefill(rows, records, windows, args.prefill_log)
    if args.baseline:
        base = [x for r in rounds(args.baseline) if r["turn_id"] >= 3 for x in r["itl_seconds"][1:]]
        print(f"{args.baseline.name} (Prefill path, no D->P):")
        describe("all ITL", base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
