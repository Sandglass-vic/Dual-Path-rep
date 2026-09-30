#!/usr/bin/env python3
"""Plot allreduce_interference_bench.py results: slowdown vs CX8 load level.

Writes interference.svg, summary.json and REPORT.md into the result directory.
Rows: mean and p99 per-iteration time relative to the scheme's own 0% case.
Columns: g5 Prefill-pattern probe and g6 Decode-pattern probe. Hollow markers:
the scheme did not reach 90% of the target load at that level.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GPU_SHARE_PEAK = 50.0  # GB/s per GPU: CX8 100 GB/s shared by 2 GPUs
LABELS = {
    "p2d": "P->D RDMA (g5 GPU -> g6 host)",
    "d2p": "D->P RDMA (g6 host -> g5 GPU)",
    "hicache_h2d": "hicache H2D (copy engine, both nodes)",
    "nic_h2d": "NIC loopback H2D (on_cuda async, both nodes)",
}
COLORS = {"p2d": "#1f77b4", "d2p": "#ff7f0e", "hicache_h2d": "#2ca02c", "nic_h2d": "#d62728"}
TRAFFIC_NODE = {"p2d": 0, "d2p": 1}


def load(result_dir: Path) -> list[dict]:
    records = []
    for path in sorted(result_dir.glob("node*.jsonl")):
        records += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return records


def achieved_share(records, scheme: str, level: int, node: int) -> float | None:
    """Mean achieved load per GPU as a fraction of its CX8 share, on the traffic node."""
    source = TRAFFIC_NODE.get(scheme, node)
    rates = [
        r["achieved_gbytes_s"] for r in records
        if r["kind"] == "generator" and r["scheme"] == scheme and r["level"] == level and r["node"] == source
    ]
    if not rates:
        return None
    return sum(rates) / len(rates) / GPU_SHARE_PEAK


def main() -> int:
    result_dir = Path(sys.argv[1])
    records = load(result_dir)
    probes = {(r["scheme"], r["level"], r["node"]): r for r in records if r["kind"] == "probe"}
    hca = {(r["scheme"], r["level"], r["node"]): r["hca"] for r in records if r["kind"] == "hca"}
    schemes = [s for s in LABELS if any(k[0] == s for k in probes)]
    levels = sorted({k[1] for k in probes})
    nodes = sorted({k[2] for k in probes})

    summary = []
    for scheme in schemes:
        for node in nodes:
            base = probes.get((scheme, 0, node))
            for level in levels:
                p = probes.get((scheme, level, node))
                if not p:
                    continue
                share = achieved_share(records, scheme, level, node) if level else 0.0
                summary.append({
                    "scheme": scheme, "level": level, "node": node, "role": p["role"],
                    "iter_ms_mean": p["iter_ms_mean"], "iter_ms_p99": p["iter_ms_p99"],
                    "slowdown_mean": p["iter_ms_mean"] / base["iter_ms_mean"] if base else None,
                    "slowdown_p99": p["iter_ms_p99"] / base["iter_ms_p99"] if base else None,
                    "achieved_share": share,
                    "reached_target": share is not None and (level == 0 or share >= 0.9 * level / 100),
                    "overran": p["overran"], "graph": p["graph"],
                    "ar_bytes": p["ar_bytes"], "ar_count": p["ar_count"], "iters": p["iters"],
                    "hca": hca.get((scheme, level, node)),
                })
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    fig, axes = plt.subplots(2, len(nodes), figsize=(7.5 * len(nodes), 9), squeeze=False, sharex=True)
    for col, node in enumerate(nodes):
        rows = [s for s in summary if s["node"] == node]
        if not rows:
            continue
        ar_kib = rows[0]["ar_bytes"] / 1024
        pattern = (
            f"g5 Prefill pattern: {rows[0]['ar_count']} x all-reduce {ar_kib / 1024:.1f} MiB + 1 all-gather (eager)"
            if rows[0]["role"] == "prefill"
            else f"g6 Decode pattern: {rows[0]['ar_count']} x all-reduce {ar_kib:.0f} KiB + 1 all-gather"
            + (" (CUDA graph)" if rows[0]["graph"] else " (eager)")
        )
        for row, metric in enumerate(("slowdown_mean", "slowdown_p99")):
            ax = axes[row][col]
            for scheme in schemes:
                pts = sorted((s for s in rows if s["scheme"] == scheme and s[metric]), key=lambda s: s["level"])
                if not pts:
                    continue
                xs = [s["level"] for s in pts]
                ys = [s[metric] for s in pts]
                ax.plot(xs, ys, color=COLORS[scheme], label=LABELS[scheme], linewidth=1.6)
                for s in pts:
                    ax.plot(s["level"], s[metric], "o", color=COLORS[scheme], markersize=6,
                            markerfacecolor=COLORS[scheme] if s["reached_target"] else "white")
            ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--")
            ax.set_xticks(levels)
            ax.set_xticklabels([f"{v}%" for v in levels])
            ax.grid(alpha=0.3)
            kind = "mean" if metric == "slowdown_mean" else "p99"
            ax.set_ylabel(f"TP4 comm time per iteration, {kind} (x of 0% load)")
            if row == 0:
                ax.set_title(pattern, fontsize=10)
            if row == 1:
                ax.set_xlabel("Offered traffic per GPU, % of its CX8 share (100% = 50 GB/s = 800 Gb/s / 2 GPUs)")
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "TP4 all-reduce slowdown under KV traffic (g5/g6, A100 PCIe + CX8, no VL QoS)\n"
        "hollow marker: scheme did not reach 90% of the target load (runs flat out)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(result_dir / "interference.svg")

    lines = [f"# TP4 all-reduce 干扰：`{result_dir.name}`", "", "![interference](interference.svg)", ""]
    for node in nodes:
        rows = [s for s in summary if s["node"] == node]
        if not rows:
            continue
        lines += [f"## node {node}（{rows[0]['role']} 模式）", "",
                  "| 方案 | 目标 | 单卡实达（占 50 GB/s） | 每次迭代 mean ms | p99 ms | mean 倍数 | p99 倍数 | 样本截断 |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        for s in rows:
            share = "—" if s["achieved_share"] is None else f"{s['achieved_share'] * 100:.1f}%"
            lines.append(
                f"| {s['scheme']} | {s['level']}% | {share} | {s['iter_ms_mean']:.3f} | {s['iter_ms_p99']:.3f} | "
                f"{s['slowdown_mean']:.2f} | {s['slowdown_p99']:.2f} | {'是' if s['overran'] else ''} |"
            )
        lines.append("")
    (result_dir / "REPORT.md").write_text("\n".join(lines))
    print((result_dir / "REPORT.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
