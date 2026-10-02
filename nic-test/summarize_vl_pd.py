#!/usr/bin/env python3
"""How much JCT can D->P / NIC H2D interference cost in a real PD run?

For every Decode- or Dynamic-path run given by Slurm job id:

- D->P and Decode-side NIC H2D per request (slowest TP rank) and their share of
  the mean TTFT; Prefill-side NIC H2D for Dynamic runs.
- D->P duty cycle: union of D->P windows over the run span.
- Decode ITL overlapping other requests' D->P windows (same windows as
  analyze_real_d2p.py); their excess over the run's D->P-free ITL mean, summed
  and divided by the agent count, is the extra time each agent spends because
  of D->P, reported as a share of JCT.

    summarize_vl_pd.py <job-id>...   (logs under $DUALPATH_LOG_ROOT/<job>/pd-servers)
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from analyze_real_d2p import decode_records, rounds

RESULTS = Path(__file__).resolve().parent.parent / "results" / "latency-benchmark"
LOG_ROOT = Path(os.environ.get("DUALPATH_LOG_ROOT", "/data/home/yongxiao/mydata/test"))


def union(intervals: list[tuple[float, float]]) -> float:
    total, cur = 0.0, None
    for begin, end in sorted(intervals):
        if cur and begin <= cur[1]:
            cur[1] = max(cur[1], end)
            continue
        if cur:
            total += cur[1] - cur[0]
        cur = [begin, end]
    return total + (cur[1] - cur[0] if cur else 0.0)


def prefill_h2d(log: Path) -> list[float]:
    per_rid: dict[str, float] = defaultdict(float)
    for line in log.open(errors="ignore"):
        if "DUALPATH_TTFT_V3" not in line:
            continue
        record = json.loads(line[line.index("{"):])
        if record.get("side") == "prefill":
            per_rid[record["rid"]] = max(per_rid[record["rid"]], record.get("prefill_host_to_device_raw_seconds", 0.0))
    return [v for v in per_rid.values() if v > 0]


def summarize(job: str) -> dict:
    run_dir = next(RESULTS.glob(f"*-j{job}"))
    meta = next(csv.DictReader((run_dir / "csv" / "runs.csv").open()))
    rows = {r["request_id"]: r for r in rounds(run_dir)}
    records = decode_records(LOG_ROOT / job / "pd-servers" / "decode-0.log")
    agents = int(meta["agents"])
    trajectories = [json.loads(l) for l in next(run_dir.glob("32K/agents-*/rep-*/data/trajectories.jsonl")).open()]
    jct = max(t["elapsed_seconds"] for t in trajectories)
    stall = max(t["turn_client_runtime_seconds"] for t in trajectories)

    windows, d2p, h2d, starts = [], [], [], []
    for rid, recs in records.items():
        ranks = {r["tp_rank"]: r for r in recs}
        starts.append(min(r["decode_dispatch_ns"] for r in recs) / 1e9)
        h2d.append(max(r.get("decode_host_to_device_raw_seconds", 0.0) for r in ranks.values()))
        if len(ranks) < 4 or not any(r.get("d_to_p_transfer_raw_seconds", 0) > 0 for r in ranks.values()):
            continue
        ready = max(r["decode_before_read_end_ns"] / 1e9 + r.get("decode_lustre_read_raw_seconds", 0.0)
                    + r.get("decode_scatter_raw_seconds", 0.0) for r in ranks.values())
        ready += rows[rid].get("decode_l3_sync_wait_seconds", 0.0) if rid in rows else 0.0
        duration = max(r["d_to_p_transfer_raw_seconds"] for r in ranks.values())
        d2p.append(duration)
        windows.append((ready, ready + duration, rid))
    span = max(starts) - min(starts)

    hit, clear = [], []
    for rid, row in rows.items():
        recs = records.get(rid)
        if not recs or row["turn_id"] < 3:
            continue
        t = min(r["decode_dispatch_ns"] for r in recs) / 1e9
        for itl in row["itl_seconds"][1:]:
            begin, t = t, t + itl
            (hit if any(w0 < t and w1 > begin and wid != rid for w0, w1, wid in windows) else clear).append(itl)
    clear_mean = statistics.mean(clear)
    extra = sum(itl - clear_mean for itl in hit) / agents
    ttft = float(meta["ttft_mean_seconds"])
    p_h2d = prefill_h2d(LOG_ROOT / job / "pd-servers" / "prefill-0.log") if meta["l3_read_path"] == "dynamic" else []

    return {
        "job": job, "path": meta["l3_read_path"], "agents": agents, "vl": bool(meta["dual_path_vl"]),
        "jct": jct, "ttft": ttft, "stall": stall,
        "d2p_n": len(d2p), "d2p": statistics.mean(d2p),
        "duty": union([(a, b) for a, b, _ in windows]) / span,
        "d_h2d": statistics.mean(h2d), "p_h2d": statistics.mean(p_h2d) if p_h2d else 0.0,
        "hit_share": len(hit) / (len(hit) + len(clear)),
        "hit_excess": statistics.mean(hit) - clear_mean,
        "hit_long": sum(itl > 0.06 for itl in hit) / len(hit),
        "extra_per_agent": extra,
    }


def main() -> int:
    for job in sys.argv[1:]:
        s = summarize(job)
        print(f"j{job} {s['path']:7s} a{s['agents']:<2d} {'VL ' if s['vl'] else 'off'} "
              f"JCT={s['jct']:7.1f}s TTFT={s['ttft']:6.3f}s | D->P n={s['d2p_n']:4d} {s['d2p'] * 1e3:4.0f}ms "
              f"={s['d2p'] / s['ttft'] * 100:4.1f}%TTFT duty={s['duty'] * 100:4.1f}% | "
              f"D-H2D {s['d_h2d'] * 1e3:3.0f}ms={s['d_h2d'] / s['ttft'] * 100:4.1f}%TTFT"
              + (f" P-H2D {s['p_h2d'] * 1e3:3.0f}ms" if s["p_h2d"] else "")
              + f" | ITL on D->P {s['hit_share'] * 100:4.1f}% +{s['hit_excess'] * 1e3:4.1f}ms "
              f">60ms {s['hit_long'] * 100:4.1f}% -> {s['extra_per_agent']:4.1f}s/agent "
              f"={s['extra_per_agent'] / s['jct'] * 100:4.2f}%JCT"
              + (f" | client stall {s['stall']:.0f}s" if s["stall"] > 1 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
