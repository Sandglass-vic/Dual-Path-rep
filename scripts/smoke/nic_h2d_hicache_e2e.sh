#!/usr/bin/env bash
# End-to-end HiCache check of DUAL_PATH_CUDAMEMCPY on one GPU.
#
# Starts SGLang with a small device KV pool, write-through HiCache and the
# direct io backend, then:
#   1. sends prompt A (greedy),
#   2. sends distinct prompts that evict A from the device pool (A stays in
#      the host pool through write-through),
#   3. sends A again, which must be served by a host -> device load.
# Runs once per mode and prints the outputs, cached_tokens and the HCA
# rx_write_requests delta. nic must match hicache token for token.
#
#   srun -N1 -w g5 -p run --gres=gpu:1 -c 32 --mem=64G -t 30 \
#     scripts/smoke/nic_h2d_hicache_e2e.sh
set -euo pipefail

CODE_HOME=${CODE_HOME:-/data/home/yongxiao/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/data/home/yongxiao/mydata}
IMAGE=${IMAGE:-$RUNTIME_HOME/images/sglang:26.01-py3-mooncake0.3.13.post1-dualpath-patched-l3bps-20260929-nich2d.sqsh}
MODEL=${MODEL:-/data/models/Qwen/Qwen3-0.6B}
MODES=${MODES:-"hicache nic"}
LAYOUTS=${LAYOUTS:-"page_first_direct layer_first"}
HCA=${HCA:-mlx5_0}
PORT=${PORT:-31080}
RESULT_DIR=${RESULT_DIR:-$RUNTIME_HOME/test/nic-h2d/e2e-${SLURM_JOB_ID:-manual}}
mkdir -p "$RESULT_DIR"

for layout in $LAYOUTS; do
  for mode in $MODES; do
    log=$RESULT_DIR/server-$layout-$mode.log
    echo "== layout=$layout mode=$mode"
    srun --overlap --ntasks=1 \
      --container-image="$IMAGE" --container-mounts=/data:/data \
      --container-workdir="$CODE_HOME" --no-container-mount-home \
      env DUAL_PATH_CUDAMEMCPY="$mode" DUAL_PATH_NIC_DEVICE="$HCA" \
      DUALPATH_NO_CROSS_TURN_KV_REUSE=0 \
      XDG_CACHE_HOME=/tmp/nic-h2d-e2e-cache \
      bash -c '
        set -euo pipefail
        python3 -m sglang.launch_server --model-path "$1" --port "$2" --tp 1 \
          --page-size 64 --max-total-tokens 16384 --mem-fraction-static 0.5 \
          --enable-hierarchical-cache --hicache-ratio 4 \
          --hicache-io-backend direct --hicache-mem-layout "$3" \
          --hicache-write-policy write_through \
          --disable-cuda-graph --log-level info >"$4" 2>&1 &
        server=$!
        trap "kill $server 2>/dev/null; wait $server 2>/dev/null || true" EXIT
        python3 "$5" --port "$2" --hca "$6" --label "$3-$7" --server-pid "$server"
      ' _ "$MODEL" "$PORT" "$layout" "$log" "$CODE_HOME/scripts/smoke/nic_h2d_hicache_client.py" "$HCA" "$mode" \
      | tee -a "$RESULT_DIR/results.jsonl"
    grep -E "DUAL_PATH_CUDAMEMCPY=nic|DUAL_PATH_NIC_REGISTER" "$log" | head -5 || true
  done
done
python3 - "$RESULT_DIR/results.jsonl" <<'PY'
import json
import sys
from collections import defaultdict

by_layout = defaultdict(dict)
for line in open(sys.argv[1]):
    if line.startswith("E2E_RESULT "):
        record = json.loads(line.split(" ", 1)[1])
        layout, mode = record["label"].rsplit("-", 1)
        by_layout[layout][mode] = record
ok = True
for layout, modes in by_layout.items():
    for mode, r in modes.items():
        print(f"{layout:18s} {mode:8s} reload_cached={r['reload_cached_tokens']}/{r['prompt_tokens']} "
              f"same_as_first={r['reload_matches_first']} hca_rx_write_requests={r['hca_rx_write_requests']}")
        ok &= r["reload_matches_first"] and r["reload_cached_tokens"] > 0
    if {"hicache", "nic"} <= modes.keys():
        same = modes["hicache"]["first_text"] == modes["nic"]["first_text"]
        print(f"{layout:18s} nic output == hicache output: {same}")
        ok &= same
print("E2E " + ("PASS" if ok else "FAIL"))
PY
