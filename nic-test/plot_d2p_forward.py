#!/usr/bin/env python3
"""Plot d2p_forward_bench.py results: forward time vs D->P rate per GPU.

Writes d2p-forward.svg, summary.json and REPORT.md into the result directory.
Columns: g5 Prefill forward and g6 Decode step. Lines: mean and p99 per-iteration
time. Hollow markers: D->P did not reach 90% of the target rate; the label under
each point is the achieved rate.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROLE_TITLE = {
    "prefill": "g5 Prefill forward (T=1004, ctx 19,630, eager) - receives D->P into its GPUs",
    "decode": "g6 Decode step (B=6, ctx 19,782, CUDA graph) - sends D->P from its host",
}


def main() -> int:
    result_dir = Path(sys.argv[1])
    records = []
    for path in sorted(result_dir.glob("node*.jsonl")):
        records += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    probes = {(r["level"], r["node"]): r for r in records if r["kind"] == "probe"}
    gens = [r for r in records if r["kind"] == "generator"]
    levels = sorted({k[0] for k in probes})
    nodes = sorted({k[1] for k in probes})

    def achieved(level: float) -> float:
        rates = [g["achieved_gbytes_s"] for g in gens if g["level"] == level]
        return statistics.mean(rates) if rates else 0.0

    summary = []
    for node in nodes:
        base = probes.get((0.0, node)) or probes.get((0, node))
        for level in levels:
            p = probes.get((level, node))
            if not p:
                continue
            rate = achieved(level) if level else 0.0
            summary.append({
                "node": node, "role": p["role"], "level_gbps": level, "achieved_gbps": rate,
                "reached_target": level == 0 or rate >= 0.9 * level,
                "iter_ms_mean": p["iter_ms_mean"], "iter_ms_p50": p["iter_ms_p50"], "iter_ms_p99": p["iter_ms_p99"],
                "slowdown_mean": p["iter_ms_mean"] / base["iter_ms_mean"] if base else None,
                "slowdown_p99": p["iter_ms_p99"] / base["iter_ms_p99"] if base else None,
                "compute_only_ms": p["compute_only_ms"], "iters": p["iters"], "overran": p["overran"],
            })
    extra = {
        "request_bytes": gens[0]["request_bytes"] if gens else None,
        "block_bytes": gens[0]["block_bytes"] if gens else None,
        "threads": gens[0]["threads"] if gens else None,
    }
    (result_dir / "summary.json").write_text(json.dumps({"rows": summary, **extra}, indent=2) + "\n")

    fig, axes = plt.subplots(1, len(nodes), figsize=(7.5 * len(nodes), 5.2), squeeze=False)
    for col, node in enumerate(nodes):
        ax = axes[0][col]
        rows = [s for s in summary if s["node"] == node]
        if not rows:
            continue
        for metric, color, label in (("iter_ms_mean", "#ff7f0e", "mean"), ("iter_ms_p99", "#7f3c00", "p99")):
            xs = [s["level_gbps"] for s in rows]
            ys = [s[metric] for s in rows]
            ax.plot(xs, ys, color=color, label=label, linewidth=1.6)
            for s in rows:
                ax.plot(s["level_gbps"], s[metric], "o", color=color, markersize=6,
                        markerfacecolor=color if s["reached_target"] else "white")
        for s in rows:
            ax.annotate(f"x{s['slowdown_mean']:.2f}", (s["level_gbps"], s["iter_ms_mean"]),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#ff7f0e")
        top = max(s["iter_ms_p99"] for s in rows)
        for s in rows:
            if s["level_gbps"]:
                ax.annotate(f"got {s['achieved_gbps']:.1f}", (s["level_gbps"], 0), textcoords="offset points",
                            xytext=(0, 4), ha="center", fontsize=7, color="#555555")
        ax.axhline(rows[0]["compute_only_ms"], color="#888888", linewidth=0.8, linestyle=":",
                   label=f"compute only, no comm ({rows[0]['compute_only_ms']:.1f} ms, eager)")
        ax.set_ylim(0, top * 1.15)
        ax.set_xticks(levels)
        ax.set_xticklabels([f"{v:g}" for v in levels])
        ax.set_xlabel("D->P RDMA offered rate per GPU (GB/s); 'got' = achieved")
        ax.set_ylabel("time per iteration (ms)")
        ax.set_title(ROLE_TITLE[rows[0]["role"]], fontsize=9.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Qwen2.5-32B TP4 forward under D->P RDMA (g6 host staging -> g5 GPU, real-shaped requests, no VL QoS)\n"
        "hollow marker: D->P did not reach 90% of the offered rate (runs flat out)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(result_dir / "d2p-forward.svg")

    lines = [f"# D->P 对 forward 的影响：`{result_dir.name}`", "", "![d2p-forward](d2p-forward.svg)", "",
             f"每个请求 {extra['request_bytes'] / 1e9:.2f} GB（128 块 × {extra['block_bytes'] / 2**20:.1f} MiB），"
             f"每块 GPU {extra['threads']} 个线程。" if gens else "", ""]
    for node in nodes:
        rows = [s for s in summary if s["node"] == node]
        if not rows:
            continue
        lines += [f"## node {node}（{rows[0]['role']}），纯计算 {rows[0]['compute_only_ms']:.1f} ms", "",
                  "| 目标 GB/s/GPU | 实达 GB/s/GPU | mean ms | p50 ms | p99 ms | mean 倍数 | p99 倍数 | 样本 | 样本截断 |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for s in rows:
            lines.append(
                f"| {s['level_gbps']:g} | {s['achieved_gbps']:.1f} | {s['iter_ms_mean']:.2f} | {s['iter_ms_p50']:.2f} | "
                f"{s['iter_ms_p99']:.2f} | {s['slowdown_mean']:.2f} | {s['slowdown_p99']:.2f} | {s['iters']} | "
                f"{'是' if s['overran'] else ''} |"
            )
        lines.append("")
    (result_dir / "REPORT.md").write_text("\n".join(lines))
    print((result_dir / "REPORT.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
