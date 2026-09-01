#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR is required}"
: "${LOG_DIR:?LOG_DIR is required}"
: "${PREFILL_HOST:?PREFILL_HOST is required}"

IB_DEVICE=${IB_DEVICE:-mlx5_0}
PREFILL_PORT=${PREFILL_PORT:-30000}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
if [[ -n ${PREFILL_ID:-} ]]; then
  PREFILL_LOG="$LOG_DIR/prefill-$PREFILL_ID.log"
else
  PREFILL_LOG="$LOG_DIR/prefill.log"
fi

mkdir -p "$LOG_DIR"

echo "Starting TP8 prefill on $PREFILL_HOST"
exec python -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --served-model-name qwen2.5-32b \
  --host 0.0.0.0 \
  --port "$PREFILL_PORT" \
  --tp-size 8 \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --context-length 4096 \
  --mem-fraction-static 0.75 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
  --disaggregation-ib-device "$IB_DEVICE" \
  >"$PREFILL_LOG" 2>&1
