#!/usr/bin/env python3
"""Capture mount, filesystem, network, RDMA, and NFS counters for an I/O phase."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path


PREFIX = "IOSTATE "


def mount_for(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    best: dict[str, object] | None = None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        mountpoint = Path(fields[4].replace("\\040", " "))
        try:
            resolved.relative_to(mountpoint)
        except ValueError:
            continue
        if best is None or len(str(mountpoint)) > len(str(best["mountpoint"])):
            right_fields = right.split()
            best = {
                "mountpoint": str(mountpoint),
                "root": fields[3],
                "mount_options": fields[5].split(","),
                "optional_fields": fields[6:],
                "filesystem": right_fields[0],
                "source": right_fields[1],
                "super_options": right_fields[2].split(","),
            }
    if best is None:
        raise RuntimeError(f"no mount entry found for {resolved}")
    return best


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


# Effective payload fraction of the PCIe line rate: 8b/10b up to Gen2,
# 128b/130b for Gen3 through Gen5, and the Gen6 FLIT overhead.
PCIE_ENCODING = {2.5: 0.8, 5.0: 0.8, 8.0: 128 / 130, 16.0: 128 / 130, 32.0: 128 / 130}


def pcie_link(device: Path) -> dict[str, object] | None:
    """Theoretical one-direction PCIe payload bandwidth of a device link."""
    speed_text = read_text(device / "current_link_speed")
    width = read_int(device / "current_link_width")
    if not speed_text or not width:
        return None
    try:
        gigatransfers = float(speed_text.split()[0])
    except ValueError:
        return None
    efficiency = PCIE_ENCODING.get(gigatransfers, 128 / 130)
    bytes_per_second = gigatransfers * 1e9 * width * efficiency / 8
    return {
        "current_link_speed": speed_text,
        "max_link_speed": read_text(device / "max_link_speed"),
        "current_link_width": width,
        "max_link_width": read_int(device / "max_link_width"),
        "theoretical_gib_s": bytes_per_second / 2**30,
    }


def infiniband_port(device: str) -> dict[str, object] | None:
    root = Path("/sys/class/infiniband") / device
    port = root / "ports/1"
    rate_text = read_text(port / "rate")
    if rate_text is None:
        return None
    try:
        gigabits = float(rate_text.split()[0])
    except ValueError:
        gigabits = 0.0
    pci_device = (root / "device").resolve()
    return {
        "rate": rate_text,
        "rate_gbit_s": gigabits,
        "theoretical_gib_s": gigabits * 1e9 / 8 / 2**30,
        "state": read_text(port / "state"),
        "link_layer": read_text(port / "link_layer"),
        "board_id": read_text(root / "board_id"),
        "pci_address": pci_device.name,
        "pcie": pcie_link(pci_device),
        "net_devices": sorted(
            entry.name for entry in (root / "device/net").iterdir()
        )
        if (root / "device/net").is_dir()
        else [],
    }


def block_device_for(source: str) -> dict[str, object] | None:
    """Model and PCIe link of the NVMe namespace backing a local filesystem."""
    name = Path(source).name
    holder = Path("/sys/class/block") / name
    if not holder.is_dir():
        return None
    resolved = holder.resolve()
    controller = resolved.parent
    pci_device = (controller / "device").resolve() if (controller / "device").exists() else None
    return {
        "device": name,
        "model": read_text(controller / "model"),
        "firmware": read_text(controller / "firmware_rev"),
        "logical_block_bytes": read_int(holder / "queue/logical_block_size"),
        "rotational": read_int(holder / "queue/rotational"),
        "pcie": pcie_link(pci_device) if pci_device else None,
    }


def interface_link(name: str) -> dict[str, object]:
    root = Path("/sys/class/net") / name
    return {
        "mtu": read_int(root / "mtu"),
        # IPoIB datagram mode caps the MTU at 2044 bytes, which dominates the
        # achievable TCP throughput long before the IB link rate does.
        "ipoib_mode": read_text(root / "mode"),
        "operstate": read_text(root / "operstate"),
    }


def theoretical_ceilings(
    interface: str, devices: list[str], mount: dict[str, object]
) -> dict[str, object]:
    ports = {device: infiniband_port(device) for device in devices}
    ports = {device: payload for device, payload in ports.items() if payload}
    storage_port = next(
        (
            payload
            for payload in ports.values()
            if interface in (payload.get("net_devices") or [])
        ),
        None,
    )
    block = None
    if str(mount.get("filesystem")) not in {"nfs", "nfs4"}:
        block = block_device_for(str(mount.get("source", "")))
    limits: dict[str, float] = {}
    if storage_port:
        limits["ib_link_gib_s"] = float(storage_port["theoretical_gib_s"])
        if storage_port.get("pcie"):
            limits["hca_pcie_gib_s"] = float(storage_port["pcie"]["theoretical_gib_s"])
    if block and block.get("pcie"):
        limits["nvme_pcie_gib_s"] = float(block["pcie"]["theoretical_gib_s"])
    return {
        "interface": interface_link(interface),
        "storage_interface_port": storage_port,
        "infiniband_ports": ports,
        "block_device": block,
        "limits_gib_s": limits,
        "binding_limit_gib_s": min(limits.values()) if limits else None,
    }


def network_counters(interface: str) -> dict[str, int] | None:
    for line in Path("/proc/net/dev").read_text().splitlines():
        name, separator, values = line.partition(":")
        if separator and name.strip() == interface:
            fields = [int(value) for value in values.split()]
            return {
                "rx_bytes": fields[0],
                "rx_packets": fields[1],
                "rx_errors": fields[2],
                "rx_dropped": fields[3],
                "tx_bytes": fields[8],
                "tx_packets": fields[9],
                "tx_errors": fields[10],
                "tx_dropped": fields[11],
            }
    return None


def hca_counters(devices: list[str]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for device in devices:
        counters: dict[str, int] = {}
        root = Path("/sys/class/infiniband") / device / "ports/1/counters"
        for name in ("port_rcv_data", "port_xmit_data", "port_rcv_packets", "port_xmit_packets"):
            value = read_int(root / name)
            if value is not None:
                counters[name] = value
        if counters:
            result[device] = counters
    return result


def rpc_counters() -> dict[str, dict[str, list[int | float]]]:
    result: dict[str, dict[str, list[int | float]]] = {}
    for name, path in (
        ("nfs_client", Path("/proc/net/rpc/nfs")),
        ("nfs_server", Path("/proc/net/rpc/nfsd")),
    ):
        try:
            counters: dict[str, list[int | float]] = {}
            for line in path.read_text().splitlines():
                fields = line.split()
                if fields:
                    values: list[int | float] = []
                    for value in fields[1:]:
                        try:
                            values.append(int(value))
                        except ValueError:
                            values.append(float(value))
                    counters[fields[0]] = values
            result[name] = counters
        except (FileNotFoundError, PermissionError):
            continue
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--interface", default="ibs3")
    parser.add_argument("--hca", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    args.path.mkdir(parents=True, exist_ok=True)
    stat = args.path.stat()
    fs = args.path.stat().st_dev
    vfs = __import__("os").statvfs(args.path)
    mount = mount_for(args.path)
    devices = args.hca or ["mlx5_0", "mlx5_4"]
    record = {
        "schema_version": 1,
        "label": args.label,
        "timestamp_ns": time.time_ns(),
        "host": socket.gethostname().split(".")[0],
        "path": str(args.path.resolve()),
        "path_device": fs,
        "path_uid": stat.st_uid,
        "path_gid": stat.st_gid,
        "mount": mount,
        "theoretical": theoretical_ceilings(args.interface, devices, mount),
        "filesystem": {
            "block_size": vfs.f_frsize,
            "blocks": vfs.f_blocks,
            "blocks_available": vfs.f_bavail,
            "bytes_available": vfs.f_bavail * vfs.f_frsize,
        },
        "interface": {
            "name": args.interface,
            "counters": network_counters(args.interface),
        },
        "hca": hca_counters(devices),
        "rpc": rpc_counters(),
    }
    rendered = json.dumps(record, sort_keys=True)
    print(PREFIX + rendered, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
