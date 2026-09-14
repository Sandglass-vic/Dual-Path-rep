#!/usr/bin/env bash

# Apply one coherent 1P2D ablation profile. Call this after setting
# ABLATION_PROFILE and before launching Prefill/Decode processes.
apply_ablation_profile() {
  local profile=${1:-full}

  PD_TRANSFER_BACKEND=mooncake
  CHUNKED_PREFILL_SIZE=8192
  DISABLE_OVERLAP_SCHEDULE=0
  MEASURED_PREFILL_L3=1
  MEASURED_DECODE_OFFLOAD=1

  case "$profile" in
    full)
      ;;
    no-prefill-l3)
      MEASURED_PREFILL_L3=0
      ;;
    no-decode-offload)
      MEASURED_DECODE_OFFLOAD=0
      ;;
    no-l3)
      MEASURED_PREFILL_L3=0
      MEASURED_DECODE_OFFLOAD=0
      ;;
    bulk-prefill)
      CHUNKED_PREFILL_SIZE=-1
      ;;
    no-overlap)
      DISABLE_OVERLAP_SCHEDULE=1
      ;;
    *)
      echo "Unknown ABLATION_PROFILE=$profile" >&2
      echo "Valid profiles: full, no-prefill-l3, no-decode-offload, no-l3, bulk-prefill, no-overlap" >&2
      return 2
      ;;
  esac

  export ABLATION_PROFILE=$profile
  export PD_TRANSFER_BACKEND CHUNKED_PREFILL_SIZE
  export DISABLE_OVERLAP_SCHEDULE
  export MEASURED_PREFILL_L3 MEASURED_DECODE_OFFLOAD
}

print_ablation_profile() {
  printf '%s\n' \
    "ABLATION_PROFILE=$ABLATION_PROFILE" \
    "PD_TRANSFER_BACKEND=$PD_TRANSFER_BACKEND" \
    "CHUNKED_PREFILL_SIZE=$CHUNKED_PREFILL_SIZE" \
    "DISABLE_OVERLAP_SCHEDULE=$DISABLE_OVERLAP_SCHEDULE" \
    "MEASURED_PREFILL_L3=$MEASURED_PREFILL_L3" \
    "MEASURED_DECODE_OFFLOAD=$MEASURED_DECODE_OFFLOAD"
}
