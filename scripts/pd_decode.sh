#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?MODEL_DIR is required}"
: "${LOG_DIR:?LOG_DIR is required}"
: "${DECODE_HOST:?DECODE_HOST is required}"

IB_DEVICE=${IB_DEVICE:-mlx5_0}
DECODE_PORT=${DECODE_PORT:-30000}
if [[ -n ${DECODE_ID:-} ]]; then
  DECODE_LOG="$LOG_DIR/decode-$DECODE_ID.log"
else
  DECODE_LOG="$LOG_DIR/decode.log"
fi

mkdir -p "$LOG_DIR"

echo "Starting TP8 decode on $DECODE_HOST"
exec python -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --served-model-name qwen2.5-32b \
  --host 0.0.0.0 \
  --port "$DECODE_PORT" \
  --tp-size 8 \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --context-length 4096 \
  --mem-fraction-static 0.75 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "$IB_DEVICE" \
  >"$DECODE_LOG" 2>&1
