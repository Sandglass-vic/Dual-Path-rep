#!/usr/bin/env bash
# Derive an image with the DUAL_PATH_KVTRANS=layer|full switch.
#
# Re-applies patch_sglang_benchmark_timing.py (for the per-endpoint control
# socket lock) and then patch_sglang_kvtrans.py on top of the NIC H2D image.
# With DUAL_PATH_KVTRANS unset the image behaves like the base: Prefill path
# layer-wise, Decode path full block.
#
#   scripts/infra/build_kvtrans_image.sh
#   BASE_IMAGE=... OUTPUT_IMAGE=... scripts/infra/build_kvtrans_image.sh
set -euo pipefail

CODE_HOME=${CODE_HOME:-/data/home/yongxiao/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/data/home/yongxiao/mydata}
BASE_IMAGE=${BASE_IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-l3bps-20260929-nich2d-async.sqsh}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-${BASE_IMAGE%.sqsh}-kvtrans.sqsh}
NODE=${NODE:-g5}

[[ -f $BASE_IMAGE ]] || { echo "Missing base image: $BASE_IMAGE" >&2; exit 2; }
[[ ! -e $OUTPUT_IMAGE ]] || { echo "Refusing to overwrite $OUTPUT_IMAGE" >&2; exit 2; }

srun -N1 -w "$NODE" -p run --cpus-per-task=32 --mem=64G -t 01:00:00 -J kvtrans-image \
  --container-image="$BASE_IMAGE" \
  --container-writable --container-remap-root \
  --container-mounts="$CODE_HOME:/workspace/dual-path:ro" \
  --no-container-mount-home \
  --container-save="$OUTPUT_IMAGE" \
  bash -c '
    set -euo pipefail
    python3 /workspace/dual-path/scripts/infra/patch_sglang_benchmark_timing.py
    python3 /workspace/dual-path/scripts/infra/patch_sglang_kvtrans.py
    python3 - <<PY
from pathlib import Path
import sglang.srt.utils.dualpath_kvtrans as kvtrans
root = Path(kvtrans.__file__).resolve().parents[1]
conn = (root / "disaggregation/mooncake/conn.py").read_text()
assert "_dualpath_sockets" in conn and "_dualpath_send_layerwise" in conn
assert "dualpath_full_preload" in (root / "mem_cache/hiradix_cache.py").read_text()
assert "dualpath_remote_flags" in (root / "managers/tp_worker.py").read_text()
assert (kvtrans.PREFILL_MODE, kvtrans.DECODE_MODE) == ("layer", "full")
print("kvtrans patch verified")
PY
  '
sha256sum "$OUTPUT_IMAGE"
echo "KVTRANS_IMAGE=$OUTPUT_IMAGE"
