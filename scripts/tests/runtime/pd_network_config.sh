#!/usr/bin/env bash
# Shared pza100 A100 compute-fabric mapping for Mooncake PD traffic.

DUALPATH_PZA100_IB_DEVICE_MAP='{"0":"mlx5_0","1":"mlx5_0","2":"mlx5_4","3":"mlx5_4"}'
IB_DEVICE=${IB_DEVICE:-$DUALPATH_PZA100_IB_DEVICE_MAP}
MOONCAKE_DEVICE=${MOONCAKE_DEVICE:-$IB_DEVICE}
export IB_DEVICE MOONCAKE_DEVICE DUALPATH_PZA100_IB_DEVICE_MAP

# Per-path IB service level (SL), 0-3. With opensm-qos-dualpath.conf applied
# SL0-3 map to VL0-3; the current SM maps SL1-3 all to VL1.
# Leaving all six unset keeps every QP on SL0.
# Mooncake reads MC_IB_SL once per process, so the NIC H2D engine always shares
# the SL of the PD engine in the same process (P2D on Prefill, D2P on Decode).
DUALPATH_VL_NAMES=(
  DUAL_PATH_VL_P_NCCL DUAL_PATH_VL_D_NCCL DUAL_PATH_VL_P2D DUAL_PATH_VL_D2P
  DUAL_PATH_VL_P_NIC_H2D DUAL_PATH_VL_D_NIC_H2D
)

dualpath_resolve_ib_vl() {
  local name enabled=0
  if [[ -n ${DUAL_PATH_VL_NCCL:-} ]]; then
    echo "DUAL_PATH_VL_NCCL was removed; set DUAL_PATH_VL_P_NCCL and DUAL_PATH_VL_D_NCCL" >&2
    return 2
  fi
  for name in "${DUALPATH_VL_NAMES[@]}"; do
    [[ -n ${!name:-} ]] && enabled=1
  done
  export DUAL_PATH_VL_ENABLED=$enabled
  (( enabled )) || return 0
  for name in DUAL_PATH_VL_P_NCCL DUAL_PATH_VL_D_NCCL DUAL_PATH_VL_P2D DUAL_PATH_VL_D2P; do
    printf -v "$name" '%s' "${!name:-0}"
  done
  DUAL_PATH_VL_P_NIC_H2D=${DUAL_PATH_VL_P_NIC_H2D:-$DUAL_PATH_VL_P2D}
  DUAL_PATH_VL_D_NIC_H2D=${DUAL_PATH_VL_D_NIC_H2D:-$DUAL_PATH_VL_D2P}
  for name in "${DUALPATH_VL_NAMES[@]}"; do
    if [[ ! ${!name} =~ ^[0-3]$ ]]; then
      echo "$name must be 0-3 (operational VLs), got '${!name}'" >&2
      return 2
    fi
    export "$name"
  done
  if [[ $DUAL_PATH_VL_P_NIC_H2D != "$DUAL_PATH_VL_P2D" ]]; then
    echo "DUAL_PATH_VL_P_NIC_H2D must equal DUAL_PATH_VL_P2D" \
         "(loopback QP SL is not separable yet)" >&2
    return 2
  fi
  if [[ $DUAL_PATH_VL_D_NIC_H2D != "$DUAL_PATH_VL_D2P" ]]; then
    echo "DUAL_PATH_VL_D_NIC_H2D must equal DUAL_PATH_VL_D2P" \
         "(loopback QP SL is not separable yet)" >&2
    return 2
  fi
}

dualpath_apply_ib_vl() {
  local role=$1
  dualpath_resolve_ib_vl || return 2
  if [[ $DUAL_PATH_VL_ENABLED != 1 ]]; then
    echo "IB VL: disabled (all QPs on SL0)"
    return 0
  fi
  case $role in
    prefill) export NCCL_IB_SL=$DUAL_PATH_VL_P_NCCL MC_IB_SL=$DUAL_PATH_VL_P2D ;;
    decode) export NCCL_IB_SL=$DUAL_PATH_VL_D_NCCL MC_IB_SL=$DUAL_PATH_VL_D2P ;;
    *) echo "dualpath_apply_ib_vl: unknown role $role" >&2; return 2 ;;
  esac
  echo "IB VL: role=$role NCCL_IB_SL=$NCCL_IB_SL MC_IB_SL=$MC_IB_SL" \
       "(PD send and NIC H2D share MC_IB_SL)"
}

dualpath_validate_ib_devices() {
  python3 - "$IB_DEVICE" "${TP_SIZE:-4}" <<'PY'
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
