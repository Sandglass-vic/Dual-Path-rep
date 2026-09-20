#!/usr/bin/env bash
# Shared DGX H100 compute-fabric mapping for Mooncake PD and HiCache traffic.

DUALPATH_DGX_H100_IB_DEVICE_MAP='{"0":"mlx5_0","1":"mlx5_3","2":"mlx5_4","3":"mlx5_5","4":"mlx5_6","5":"mlx5_9","6":"mlx5_10","7":"mlx5_11"}'
IB_DEVICE=${IB_DEVICE:-$DUALPATH_DGX_H100_IB_DEVICE_MAP}
MOONCAKE_DEVICE=${MOONCAKE_DEVICE:-$IB_DEVICE}
export IB_DEVICE MOONCAKE_DEVICE DUALPATH_DGX_H100_IB_DEVICE_MAP

dualpath_validate_ib_devices() {
  python3 - "$IB_DEVICE" "${TP_SIZE:-8}" <<'PY'
import json
import sys
from pathlib import Path

config = sys.argv[1].strip()
tp_size = int(sys.argv[2])
try:
    parsed = json.loads(config)
except json.JSONDecodeError:
    devices = [item.strip() for item in config.split(",") if item.strip()]
else:
    if not isinstance(parsed, dict):
        raise SystemExit("IB_DEVICE JSON must be a GPU-to-HCA object")
    expected = {str(index) for index in range(tp_size)}
    if set(parsed) != expected:
        raise SystemExit(
            f"IB_DEVICE must map GPU ranks 0..{tp_size - 1}; found {sorted(parsed)}"
        )
    devices = [parsed[str(index)].strip() for index in range(tp_size)]
    # DGX H100 gives every rank its own HCA. Fabrics with fewer HCAs than
    # ranks work, but the shared ports cap per-rank bandwidth.
    if len(set(devices)) != tp_size:
        print(
            f"NOTE: {len(set(devices))} unique HCAs for {tp_size} ranks; "
            "ranks share RDMA ports"
        )

for device in devices:
    state_path = Path(
        f"/sys/class/infiniband/{device}/ports/1/state"
    )
    if not state_path.is_file():
        raise SystemExit(f"Missing RDMA device: {device}")
    state = state_path.read_text().strip()
    if "ACTIVE" not in state:
        raise SystemExit(f"RDMA device is not ACTIVE: {device} ({state})")

print(f"Validated RDMA devices: {','.join(devices)}")
PY
}
