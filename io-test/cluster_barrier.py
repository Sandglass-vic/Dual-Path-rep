#!/usr/bin/env python3
"""Synchronize independent Slurm tasks to a shared start instant.

g5 and g6 do not run a common time source: their CLOCK_REALTIME values differ by
tens of seconds. A barrier that hands out an absolute epoch timestamp therefore
releases the nodes at completely different instants, and any aggregate bandwidth
computed from `min(wall_start)` / `max(wall_end)` across nodes is meaningless.

`synchronize()` measures the leader-minus-local clock offset with an NTP-style
round trip before releasing, so every task can translate its own timestamps into
the leader's timebase.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import struct
import time

OFFSET_SAMPLES = 25
CONNECT_TIMEOUT_SECONDS = 300
DEFAULT_LEAD_NS = 1_000_000_000
PROBE = b"P"
RELEASE = b"R"


def _recv_exactly(connection: socket.socket, count: int) -> bytes:
    payload = b""
    while len(payload) < count:
        chunk = connection.recv(count - len(payload))
        if not chunk:
            raise RuntimeError("barrier peer closed the connection early")
        payload += chunk
    return payload


def _serve_offset_probes(connection: socket.socket) -> None:
    """Answer probes until the follower asks to be released."""
    connection.settimeout(60)
    while True:
        tag = _recv_exactly(connection, 1)
        if tag == RELEASE:
            return
        if tag != PROBE:
            raise RuntimeError(f"unexpected barrier tag: {tag!r}")
        _recv_exactly(connection, 8)
        connection.sendall(struct.pack("!q", time.time_ns()))


def _measure_offset(connection: socket.socket) -> tuple[int, int]:
    """Return (leader_minus_local_ns, best_round_trip_ns) for this connection."""
    samples: list[tuple[int, int]] = []
    for _ in range(OFFSET_SAMPLES):
        local_before = time.time_ns()
        connection.sendall(PROBE + struct.pack("!q", local_before))
        leader_ns = struct.unpack("!q", _recv_exactly(connection, 8))[0]
        local_after = time.time_ns()
        round_trip = local_after - local_before
        midpoint = (local_before + local_after) // 2
        samples.append((round_trip, leader_ns - midpoint))
    best_round_trip = min(sample[0] for sample in samples)
    # Keep the quietest quartile: a delayed reply biases the offset by half the
    # extra latency, so the minimum-RTT samples are the accurate ones.
    threshold = best_round_trip * 2
    quiet = [offset for round_trip, offset in samples if round_trip <= threshold]
    return int(statistics.median(quiet)), int(best_round_trip)


def synchronize(
    nodes: int,
    node_index: int,
    host: str,
    port: int,
    lead_ns: int = DEFAULT_LEAD_NS,
) -> dict[str, int]:
    """Block until every task has arrived, then return the shared release point.

    The returned mapping always describes the caller's own clock:

    - `release_epoch_ns`: local CLOCK_REALTIME value to start working at;
    - `leader_minus_local_ns`: add this to a local timestamp to obtain the
      leader's timebase;
    - `offset_round_trip_ns`: best probe round trip, i.e. the offset error bound.
    """
    if nodes == 1:
        return {
            "release_epoch_ns": time.time_ns() + lead_ns,
            "leader_minus_local_ns": 0,
            "offset_round_trip_ns": 0,
        }
    if node_index == 0:
        connections: list[socket.socket] = []
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(("0.0.0.0", port))
                server.listen(nodes - 1)
                server.settimeout(CONNECT_TIMEOUT_SECONDS)
                for _ in range(nodes - 1):
                    connection, _ = server.accept()
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    if _recv_exactly(connection, 5) != b"READY":
                        raise RuntimeError("invalid barrier handshake")
                    connections.append(connection)
                for connection in connections:
                    _serve_offset_probes(connection)
                release_epoch_ns = time.time_ns() + lead_ns
                for connection in connections:
                    connection.sendall(struct.pack("!q", release_epoch_ns))
        finally:
            for connection in connections:
                connection.close()
        return {
            "release_epoch_ns": release_epoch_ns,
            "leader_minus_local_ns": 0,
            "offset_round_trip_ns": 0,
        }

    if not host:
        raise RuntimeError("a sync host is required for a multi-node barrier")
    deadline = time.monotonic() + CONNECT_TIMEOUT_SECONDS
    while True:
        try:
            with socket.create_connection((host, port), timeout=5) as connection:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                connection.sendall(b"READY")
                offset_ns, round_trip_ns = _measure_offset(connection)
                connection.sendall(RELEASE)
                connection.settimeout(CONNECT_TIMEOUT_SECONDS)
                leader_release_ns = struct.unpack(
                    "!q", _recv_exactly(connection, 8)
                )[0]
            return {
                "release_epoch_ns": leader_release_ns - offset_ns,
                "leader_minus_local_ns": offset_ns,
                "offset_round_trip_ns": round_trip_ns,
            }
        except OSError:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"timed out connecting to {host}:{port}")
            time.sleep(0.2)


def wait_until(release_epoch_ns: int) -> None:
    while True:
        remaining = release_epoch_ns - time.time_ns()
        if remaining <= 0:
            return
        time.sleep(min(remaining / 1e9, 0.005))


def coordinate(nodes: int, node_index: int, host: str, port: int) -> dict[str, int]:
    """Synchronize and block until the shared release instant."""
    handshake = synchronize(nodes, node_index, host, port)
    wait_until(handshake["release_epoch_ns"])
    return handshake


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.nodes <= 0 or not 1 <= args.port <= 65535:
        parser.error("nodes and port must be valid positive values")

    node_index = int(os.environ.get("SLURM_PROCID", "0"))
    handshake = coordinate(args.nodes, node_index, args.host, args.port)
    print("BARRIER_RELEASE " + json.dumps(handshake, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
