#!/usr/bin/env python3
"""Run one command and record which hardware paths its bytes crossed.

    capture_path_counters.py --out case.json [--hca mlx5_0 ...] [--gpus 0,1]
        [--expected-bytes N] -- COMMAND ...

Evidence collected around the command:

- HCA port counters (``port_{xmit,rcv}_data`` in 4-byte units) and every
  ``hw_counters`` entry. A same-HCA loopback may never reach the wire, so the
  responder-side ``rx_{write,read}_requests`` are the proof that the HCA
  executed the transfer.
- Per-GPU PCIe rx/tx from ``nvidia-smi dmon -s t`` (1 s samples, MB/s). The
  integral over the run approximates the bytes that crossed each GPU's link.
- Optionally any extra sampler (``--sampler-cmd``, e.g. AMDuProfPcm as root),
  started before and terminated after the command; its output goes to
  ``<out>.sampler.log``.

The command inherits stdout/stderr, so its own result lines are unaffected.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HCA_ROOT = Path("/sys/class/infiniband")
PORT_COUNTERS = (
    "port_xmit_data",
    "port_rcv_data",
    "port_xmit_packets",
    "port_rcv_packets",
)
DMON_SETTLE_SECONDS = 1.5
MIN_HCA_REQUESTS = 100
PCIE_FRACTION = 0.5


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def hca_snapshot(devices: list[str]) -> dict[str, dict[str, int]]:
    snapshot: dict[str, dict[str, int]] = {}
    for device in devices:
        port = HCA_ROOT / device / "ports" / "1"
        values: dict[str, int] = {}
        for name in PORT_COUNTERS:
            value = read_int(port / "counters" / name)
            if value is not None:
                values[name] = value
        hw = port / "hw_counters"
        if hw.is_dir():
            for entry in sorted(hw.iterdir()):
                value = read_int(entry)
                if value is not None:
                    values[f"hw.{entry.name}"] = value
        snapshot[device] = values
    return snapshot


def hca_delta(before: dict, after: dict) -> dict[str, dict[str, int]]:
    delta: dict[str, dict[str, int]] = {}
    for device, values in after.items():
        base = before.get(device, {})
        changed = {
            name: value - base[name]
            for name, value in values.items()
            if name in base and value != base[name]
        }
        for name in ("port_xmit_data", "port_rcv_data"):
            if name in changed:
                changed[f"{name}_bytes"] = changed[name] * 4
        delta[device] = changed
    return delta


class DmonSampler:
    """Collect `nvidia-smi dmon -s t` rows in a background thread."""

    def __init__(self) -> None:
        self.rows: list[tuple[float, int, float, float]] = []
        self.error = ""
        try:
            self.process = subprocess.Popen(
                ["nvidia-smi", "dmon", "-s", "t", "-d", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            self.process = None
            self.error = str(exc)
            return
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            tokens = line.split()
            if not tokens or tokens[0].startswith("#"):
                continue
            try:
                gpu = int(tokens[-3])
                rx = float(tokens[-2]) if tokens[-2] != "-" else 0.0
                tx = float(tokens[-1]) if tokens[-1] != "-" else 0.0
            except (ValueError, IndexError):
                continue
            self.rows.append((time.time(), gpu, rx, tx))

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.thread.join(timeout=5)

    def summary(self, start: float, end: float) -> dict[str, dict[str, float]]:
        per_gpu: dict[str, dict[str, float]] = {}
        for stamp, gpu, rx, tx in self.rows:
            if not start <= stamp <= end + 1.0:
                continue
            entry = per_gpu.setdefault(
                str(gpu),
                {"samples": 0, "rx_mb": 0.0, "tx_mb": 0.0, "rx_peak_mbps": 0.0, "tx_peak_mbps": 0.0},
            )
            entry["samples"] += 1
            entry["rx_mb"] += rx
            entry["tx_mb"] += tx
            entry["rx_peak_mbps"] = max(entry["rx_peak_mbps"], rx)
            entry["tx_peak_mbps"] = max(entry["tx_peak_mbps"], tx)
        for entry in per_gpu.values():
            entry["rx_bytes_est"] = entry["rx_mb"] * 1e6
            entry["tx_bytes_est"] = entry["tx_mb"] * 1e6
        return per_gpu


def verdict(
    delta: dict[str, dict[str, int]],
    pcie: dict[str, dict[str, float]],
    gpus: list[str],
    expected_bytes: int | None,
) -> dict[str, object]:
    requests = sum(
        values.get("hw.rx_write_requests", 0) + values.get("hw.rx_read_requests", 0)
        for values in delta.values()
    )
    wire_bytes = sum(
        values.get("port_xmit_data_bytes", 0) + values.get("port_rcv_data_bytes", 0)
        for values in delta.values()
    )
    gpu_bytes = sum(
        pcie.get(gpu, {}).get("rx_bytes_est", 0.0) + pcie.get(gpu, {}).get("tx_bytes_est", 0.0)
        for gpu in gpus
    )
    result: dict[str, object] = {
        "hca_requests": requests,
        "hca_wire_bytes": wire_bytes,
        "gpu_pcie_bytes_est": gpu_bytes,
        "crossed_hca": requests >= MIN_HCA_REQUESTS,
        "reached_hca_port": wire_bytes > 0,
    }
    if expected_bytes:
        result["expected_bytes"] = expected_bytes
        result["gpu_pcie_ratio"] = gpu_bytes / expected_bytes
        result["hca_wire_ratio"] = wire_bytes / expected_bytes
        result["crossed_gpu_pcie"] = gpu_bytes >= PCIE_FRACTION * expected_bytes
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--hca", action="append", default=[])
    parser.add_argument("--gpus", default="", help="comma-separated GPU indices the case uses")
    parser.add_argument("--expected-bytes", type=int, default=0)
    parser.add_argument("--label", default="")
    parser.add_argument("--sampler-cmd", default=os.environ.get("PATH_SAMPLER_CMD", ""))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing command after --")
    devices = args.hca or ["mlx5_0", "mlx5_4"]
    gpus = [gpu for gpu in args.gpus.split(",") if gpu]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sampler_proc = None
    if args.sampler_cmd:
        sampler_log = open(f"{args.out}.sampler.log", "w")
        sampler_proc = subprocess.Popen(
            shlex.split(args.sampler_cmd),
            stdout=sampler_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    dmon = DmonSampler()
    time.sleep(DMON_SETTLE_SECONDS)

    before = hca_snapshot(devices)
    start = time.time()
    returncode = subprocess.call(command)
    end = time.time()
    after = hca_snapshot(devices)

    time.sleep(DMON_SETTLE_SECONDS)
    dmon.stop()
    if sampler_proc is not None:
        os.killpg(sampler_proc.pid, signal.SIGINT)
        try:
            sampler_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(sampler_proc.pid, signal.SIGKILL)

    delta = hca_delta(before, after)
    pcie = dmon.summary(start, end)
    record = {
        "label": args.label,
        "command": command,
        "returncode": returncode,
        "wall_seconds": end - start,
        "hca_delta": delta,
        "gpu_pcie": pcie,
        "dmon_error": dmon.error,
        "verdict": verdict(delta, pcie, gpus, args.expected_bytes or None),
    }
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print("PATH_COUNTERS " + json.dumps({"label": args.label, **record["verdict"]}, sort_keys=True), flush=True)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
