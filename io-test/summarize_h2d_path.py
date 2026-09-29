#!/usr/bin/env python3
"""Summarize a run_h2d_path_bench.sh result directory into summary.json and REPORT.md."""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GATE_LAYER_RATIO = 1.2
GATE_BW_RATIO = 0.8


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def perftest_section(cases: list[dict]) -> tuple[str, list[dict]]:
    bw = [c for c in cases if c["kind"] == "bw"]
    lat = [c for c in cases if c["kind"] == "lat"]
    rows = []
    for c in bw:
        v = c.get("verdict") or {}
        rows.append([
            c["direction"], "+".join(map(str, c["gpus"])), c["hca"], c["size"], c["qps"], c["sl"],
            c.get("bw_gbps"), c.get("bw_gbytes_s"), v.get("hca_requests"),
            v.get("reached_hca_port"), v.get("gpu_pcie_ratio"),
        ])
    text = "### A1 perftest 同网卡回环带宽\n\n" + table(
        ["方向", "GPU", "HCA", "消息 B", "QP", "SL", "Gb/s", "GB/s", "HCA 请求数", "出端口", "GPU PCIe / 数据量"],
        rows,
    )
    if lat:
        text += "\n\n### A1 perftest 往返延迟（host ↔ GPU）\n\n" + table(
            ["GPU", "HCA", "消息 B", "t_typical µs", "t_avg µs", "p99 µs"],
            [[c["gpus"][0], c["hca"], c["size"], c.get("t_typical_us"), c.get("t_avg_us"), c.get("p99_us")] for c in lat],
        )
    return text, bw + lat


def aggregate_pybench(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = (r["label"], r["layout"], r["pattern"], r["direction"], r["mode"])
        groups[key].append(r)
    rows = []
    for (label, layout, pattern, direction, mode), items in groups.items():
        v = items[0]["verdict"]
        rows.append({
            "label": label,
            "layout": layout,
            "pattern": pattern,
            "direction": direction,
            "mode": mode,
            "gpus": len(items),
            "gbytes_s_per_gpu": statistics.mean(i["gbytes_s"] for i in items),
            "aggregate_gbytes_s": items[0]["aggregate_gbytes_s"],
            "layer_ms_p50": statistics.mean(i["layer_ms_p50"] for i in items) if items[0]["layer_ms_p50"] else None,
            "layer_ms_p99": max(i["layer_ms_p99"] for i in items) if items[0]["layer_ms_p99"] else None,
            "total_ms_median": statistics.mean(i["total_s_median"] for i in items) * 1e3,
            "submit_ms_median": statistics.mean(i["submit_ms_median"] for i in items),
            "cpu_s_per_rep": statistics.mean(i["cpu_s_per_rep"] for i in items),
            "requests_per_layer": items[0]["requests_per_layer"],
            "verified": all(i["verified"] for i in items),
            "crossed_hca": v.get("crossed_hca"),
            "reached_hca_port": v.get("reached_hca_port"),
            "hca_requests": v.get("hca_requests"),
            "gpu_pcie_ratio": v.get("gpu_pcie_ratio"),
            "mc_ib_sl": items[0].get("mc_ib_sl"),
        })
    rows.sort(key=lambda r: (r["label"], r["layout"], r["pattern"], r["direction"], r["mode"]))
    return rows


def gate(rows: list[dict]) -> list[dict]:
    single = [r for r in rows if r["label"] == "gpus-0"]
    index = {(r["layout"], r["pattern"], r["direction"], r["mode"]): r for r in single}
    out = []
    for (layout, pattern, direction, mode), nic in index.items():
        if mode != "nic":
            continue
        base = index.get((layout, pattern, direction, "hicache"))
        if not base:
            continue
        if direction == "h2d" and nic["layer_ms_p50"] and base["layer_ms_p50"]:
            layer_ratio = nic["layer_ms_p50"] / base["layer_ms_p50"]
        else:
            layer_ratio = nic["total_ms_median"] / base["total_ms_median"]
        bw_ratio = nic["gbytes_s_per_gpu"] / base["gbytes_s_per_gpu"]
        out.append({
            "layout": layout,
            "pattern": pattern,
            "direction": direction,
            "nic_over_hicache_time": layer_ratio,
            "nic_over_hicache_bw": bw_ratio,
            "nic_crossed_hca": nic["crossed_hca"],
            "hicache_crossed_hca": base["crossed_hca"],
            "pass": (layer_ratio <= GATE_LAYER_RATIO or bw_ratio >= GATE_BW_RATIO)
            and bool(nic["crossed_hca"])
            and nic["verified"],
        })
    return out


def interference(result_dir: Path) -> list[dict]:
    rows = []
    for name in ("idle", "hicache", "nic"):
        path = result_dir / f"interference-{name}.log"
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            match = re.search(r"allreduce\s+(\d+) B\s+([\d.]+) us\s+busbw\s+([\d.]+)", line)
            if match:
                rows.append({
                    "load": name,
                    "bytes": int(match.group(1)),
                    "latency_us": float(match.group(2)),
                    "busbw_gbytes_s": float(match.group(3)),
                })
    return rows


def main() -> int:
    result_dir = Path(sys.argv[1])
    perftest = [
        case
        for path in sorted(result_dir.glob("perftest*/cases.jsonl"))
        for case in load_jsonl(path)
        if case.get("bw_gbps") or case.get("t_avg_us")
    ]
    pybench = aggregate_pybench(load_jsonl(result_dir / "pybench.jsonl"))
    gates = gate(pybench)
    contention = interference(result_dir)

    sections = [f"# H2D/D2H 路径 mini bench：`{result_dir.name}`\n"]
    perftest_text, perftest_rows = perftest_section(perftest) if perftest else ("", [])
    if perftest_text:
        sections.append(perftest_text)
    if pybench:
        sections.append("### A2 Qwen2.5-32B TP4 形状（每 rank 每次 512 页 × 64 层 = 2 GiB）\n\n" + table(
            ["并发", "host 布局", "页分布", "方向", "模式", "GB/s/GPU", "总 GB/s", "每层 p50 ms",
             "每层 p99 ms", "一次总 ms", "提交 ms", "CPU s/次", "请求/层", "校验", "经 HCA", "GPU PCIe / 数据量"],
            [[r["label"], r["layout"], r["pattern"], r["direction"], r["mode"] + (f" sl{r['mc_ib_sl']}" if r["mc_ib_sl"] else ""),
              r["gbytes_s_per_gpu"], r["aggregate_gbytes_s"], r["layer_ms_p50"], r["layer_ms_p99"],
              r["total_ms_median"], r["submit_ms_median"], r["cpu_s_per_rep"], r["requests_per_layer"],
              r["verified"], r["crossed_hca"], r["gpu_pcie_ratio"]] for r in pybench],
        ))
    if gates:
        sections.append(
            f"### 决策门槛（单 GPU；nic 时间 ≤ {GATE_LAYER_RATIO}× hicache，或带宽 ≥ {GATE_BW_RATIO}× hicache，且经过 HCA、校验通过）\n\n"
            + table(
                ["host 布局", "页分布", "方向", "nic/hicache 时间", "nic/hicache 带宽", "nic 经 HCA", "hicache 经 HCA", "通过"],
                [[g["layout"], g["pattern"], g["direction"], g["nic_over_hicache_time"], g["nic_over_hicache_bw"],
                  g["nic_crossed_hca"], g["hicache_crossed_hca"], g["pass"]] for g in gates],
            )
        )
    if contention:
        sections.append("### A4 H2D 负载下的 TP4 all-reduce\n\n" + table(
            ["负载", "消息 B", "延迟 µs", "busbw GB/s"],
            [[c["load"], c["bytes"], c["latency_us"], c["busbw_gbytes_s"]] for c in contention],
        ))

    summary = {"perftest": perftest_rows, "pybench": pybench, "gate": gates, "interference": contention}
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (result_dir / "REPORT.md").write_text("\n\n".join(sections) + "\n")
    print((result_dir / "REPORT.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
