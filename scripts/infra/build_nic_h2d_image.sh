#!/usr/bin/env bash
# Derive an image with the DUAL_PATH_CUDAMEMCPY=hicache|nic switch.
#
# Applies scripts/infra/patch_sglang_nic_h2d.py on top of an existing patched
# image and saves the result next to it. The switch defaults to hicache, so the
# new image behaves exactly like the base unless DUAL_PATH_CUDAMEMCPY=nic.
#
#   scripts/infra/build_nic_h2d_image.sh
#   BASE_IMAGE=... OUTPUT_IMAGE=... scripts/infra/build_nic_h2d_image.sh
set -euo pipefail

CODE_HOME=${CODE_HOME:-/data/home/yongxiao/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/data/home/yongxiao/mydata}
BASE_IMAGE=${BASE_IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-l3bps-20260929.sqsh}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-${BASE_IMAGE%.sqsh}-nich2d.sqsh}
NODE=${NODE:-g5}

[[ -f $BASE_IMAGE ]] || { echo "Missing base image: $BASE_IMAGE" >&2; exit 2; }
[[ ! -e $OUTPUT_IMAGE ]] || { echo "Refusing to overwrite $OUTPUT_IMAGE" >&2; exit 2; }

srun -N1 -w "$NODE" -p run --cpus-per-task=32 --mem=64G -t 01:00:00 -J nic-h2d-image \
  --container-image="$BASE_IMAGE" \
  --container-writable --container-remap-root \
  --container-mounts="$CODE_HOME:/workspace/dual-path:ro" \
  --no-container-mount-home \
  --container-save="$OUTPUT_IMAGE" \
  bash -c '
    set -euo pipefail
    python3 /workspace/dual-path/scripts/infra/patch_sglang_nic_h2d.py
    DUAL_PATH_CUDAMEMCPY=hicache python3 - <<PY
from pathlib import Path
from sglang.srt.mem_cache import dualpath_nic_copy, memory_pool_host
source = Path(memory_pool_host.__file__).read_text()
assert source.count("_dualpath_nic_copy.ENABLED") == 3, "switch sites missing"
assert not dualpath_nic_copy.ENABLED
print("nic h2d patch verified")
PY
  '
sha256sum "$OUTPUT_IMAGE"
echo "NIC_H2D_IMAGE=$OUTPUT_IMAGE"
