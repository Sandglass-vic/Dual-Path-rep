#!/usr/bin/env python3
"""Figures for docs/vl_report.md.

- vl-report-nictest: TP4 probe slowdown vs D->P / NIC H2D load, Job 649 (no VL)
  vs Job 650 (probe VL0, traffic VL1).
- vl-report-mechanism: real 1P1D runs, Decode ITL overlapping D->P (excess and
  >60 ms share) and the resulting share of JCT, no VL vs VL.
- vl-report-jct: JCT change with VL per path, against the Prefill-path spread.

    plot_vl_report.py [--out docs/figures]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from summarize_vl_pd import RESULTS, summarize

HERE = Path(__file__).resolve().parent
PAIRS = [  # (label, off job, on job)
    ("decode a8\n32K", "651", "652"),
    ("dynamic a8\n32K", "655", "656"),
    ("decode a16\n32K", "658", "659"),
    ("dynamic a16\n32K", "662", "663"),
    ("decode a8\ntrace255", "666", "669"),
    ("dynamic a8\ntrace255", "665", "668"),
    ("decode a16\ntrace255", "672", "675"),
    ("dynamic a16\ntrace255*", "671", "674"),
]
PREFILL_PAIRS = [("a8 32K", "653", "654"), ("a8 32K", "653", "657"), ("a16 32K", "660", "661"),
                 ("a8 trace255", "664", "667"), ("a16 trace255", "670", "673")]
OFF, ON = "#d9534f", "#337ab7"


def jct(job: str) -> float:
    path = next(next(RESULTS.glob(f"*-j{job}")).glob("32K/agents-*/rep-*/data/trajectories.jsonl"))
    return max(json.loads(line)["elapsed_seconds"] for line in path.open())


def bars(ax, off, on, labels, ylabel, title, fmt):
    x = range(len(labels))
    for shift, values, color, name in ((-0.2, off, OFF, "no VL"), (0.2, on, ON, "VL")):
        rects = ax.bar([i + shift for i in x], values, 0.4, color=color, label=name)
        ax.bar_label(rects, [fmt.format(v) for v in values], fontsize=7, padding=1)
    ax.set_xticks(list(x), labels, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(fontsize=8)


def nictest(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, scheme, title in ((axes[0], "d2p", "D->P load"), (axes[1], "nic_h2d", "NIC loopback H2D load")):
        for job, color, name in (("649", OFF, "no VL (649)"), ("650", ON, "probe VL0 / traffic VL1 (650)")):
            rows = json.loads((HERE / "results" / f"{job}-allreduce-interference" / "summary.json").read_text())
            for role, style in (("decode", "-"), ("prefill", "--")):
                levels = defaultdict(list)
                for r in rows:
                    if r["scheme"] == scheme and r["role"] == role:
                        levels[r["level"]].append(r["slowdown_mean"])
                xs = sorted(levels)
                ax.plot(xs, [sum(levels[x]) / len(levels[x]) for x in xs], style, marker="o", ms=3,
                        color=color, label=f"{role} probe, {name}")
        ax.set_yscale("log")
        ax.set_xlabel("traffic level (% of max rate)")
        ax.set_ylabel("TP4 probe slowdown (x, log)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.suptitle("nic-test: sustained traffic vs TP4 all-reduce probe (FORCE_NIC=0)", fontsize=11)
    fig.tight_layout()
    save(fig, out / "vl-report-nictest")


def mechanism(out: Path) -> None:
    labels = [p[0] for p in PAIRS]
    off = [summarize(p[1]) for p in PAIRS]
    on = [summarize(p[2]) for p in PAIRS]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5))
    bars(axes[0], [s["hit_excess"] * 1e3 for s in off], [s["hit_excess"] * 1e3 for s in on], labels,
         "ms", "Decode step overlapping another request's D->P: extra time vs non-overlapping steps", "{:.1f}")
    bars(axes[1], [s["hit_long"] * 100 for s in off], [s["hit_long"] * 100 for s in on], labels,
         "%", "Share of overlapping steps longer than 60 ms", "{:.1f}")
    bars(axes[2], [s["extra_per_agent"] / s["jct"] * 100 for s in off],
         [s["extra_per_agent"] / s["jct"] * 100 for s in on], labels,
         "% of JCT", "D->P interference per agent as share of JCT", "{:.2f}")
    fig.text(0.01, 0.005, "* Job 671 has a 300 s client stall; its JCT is not comparable (the per-step numbers are).",
             fontsize=7)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    save(fig, out / "vl-report-mechanism")
    return off, on


def jct_change(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    groups = [("Prefill path (no D->P)", PREFILL_PAIRS, "#999999"),
              ("Decode path", [p for p in PAIRS if p[0].startswith("decode")], OFF),
              ("Dynamic path", [p for p in PAIRS if p[0].startswith("dynamic") and p[1] != "671"], ON)]
    x, ticks, names = 0, [], []
    spread = [(jct(on) / jct(off) - 1) * 100 for _, off, on in PREFILL_PAIRS]
    ax.axhspan(min(spread), max(spread), color="#999999", alpha=0.15, label="Prefill-path run-to-run spread")
    for title, pairs, color in groups:
        for i, (label, off, on) in enumerate(pairs):
            delta = (jct(on) / jct(off) - 1) * 100
            rect = ax.bar(x, delta, 0.7, color=color, label=title if i == 0 else None)
            ax.bar_label(rect, [f"{delta:+.1f}%"], fontsize=7, padding=1)
            ticks.append(x)
            agents, dataset = label.rstrip("*").split()[-2:]
            names.append(f"{agents}\n{dataset}\n{off}->{on}")
            x += 1
        x += 0.6
    ax.set_xticks(ticks, names, fontsize=6.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("JCT change with VL (%)")
    ax.set_title("JCT: VL vs no VL (single runs; Dynamic a16 671->674 excluded, 300 s stall)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    save(fig, out / "vl-report-jct")


def save(fig, stem: Path) -> None:
    for suffix in (".svg", ".png"):
        fig.savefig(stem.with_suffix(suffix), dpi=150)
    plt.close(fig)
    print(f"wrote {stem}.svg/.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE.parent / "docs" / "figures")
    out = parser.parse_args().out
    nictest(out)
    mechanism(out)
    jct_change(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
