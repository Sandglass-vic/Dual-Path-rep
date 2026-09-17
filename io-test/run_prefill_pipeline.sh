#!/usr/bin/env bash
# Reproduce Prefill HiCache O_DIRECT submission, rate limiting, and layer_first scatter.
set -euo pipefail

CODE_HOME=${CODE_HOME:-/workspace/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/mnt/lustre/users/sezhou/dual-path}
RUN_ID=${RUN_ID:-prefill-pipeline-$(date +%Y%m%d-%H%M%S)}
DATA_DIR=${DATA_DIR:-$RUNTIME_HOME/io-test/$RUN_ID/kv-pages}
RESULT_DIR=${RESULT_DIR:-$CODE_HOME/io-test/results/$RUN_ID}
TP_RANKS=${TP_RANKS:-8}
AGENTS=${AGENTS:-8}
PAGES_PER_AGENT=${PAGES_PER_AGENT:-512}
PAGE_BYTES=${PAGE_BYTES:-2097152}
PREFETCH_CAPACITY_TOKENS=${PREFETCH_CAPACITY_TOKENS:-181043}
BACKLOG_TOKENS=${BACKLOG_TOKENS:-98304}
HOST_POOL_TOKENS=${HOST_POOL_TOKENS:-0}
FULL_POOL_COMPARE=${FULL_POOL_COMPARE:-0}
H2D_COPY_ROUNDS=${H2D_COPY_ROUNDS:-0}
SINGLE_CASE=${SINGLE_CASE:-0}
KEEP_DATA=${KEEP_DATA:-0}
PREPARE_DATA=${PREPARE_DATA:-1}

mkdir -p "$RESULT_DIR"

if [[ $PREPARE_DATA == 1 ]]; then
  python3 "$CODE_HOME/io-test/kv_read_bench.py" prepare \
    --data-dir "$DATA_DIR" \
    --tp-ranks "$TP_RANKS" \
    --pages-per-rank "$((AGENTS * PAGES_PER_AGENT))" \
    --page-bytes "$PAGE_BYTES"
fi

run_case() {
  local name=$1
  local capacity=$2
  local occupied=$3
  local host_start=${4:-0}
  python3 "$CODE_HOME/io-test/prefill_pipeline_bench.py" \
    --data-dir "$DATA_DIR" \
    --tp-ranks "$TP_RANKS" \
    --agents "$AGENTS" \
    --pages-per-agent "$PAGES_PER_AGENT" \
    --page-bytes "$PAGE_BYTES" \
    --host-pool-tokens "$HOST_POOL_TOKENS" \
    --host-start-token "$host_start" \
    --h2d-copy-rounds "$H2D_COPY_ROUNDS" \
    --prefetch-capacity-tokens "$capacity" \
    --initial-prefetch-occupied-tokens "$occupied" \
    | tee "$RESULT_DIR/$name.log"
}

if [[ $SINGLE_CASE == 1 ]]; then
  run_case single 0 0 0
elif [[ $FULL_POOL_COMPARE == 1 ]]; then
  request_tokens=$((AGENTS * PAGES_PER_AGENT * 64))
  high_start=$((HOST_POOL_TOKENS - request_tokens))
  run_case full-pool-low 0 0 0
  run_case full-pool-high 0 0 "$high_start"
else
  run_case unlimited 0 0
  run_case real-capacity-empty "$PREFETCH_CAPACITY_TOKENS" 0
  run_case real-capacity-backlog "$PREFETCH_CAPACITY_TOKENS" "$BACKLOG_TOKENS"
fi

if [[ $KEEP_DATA == 0 ]]; then
  python3 "$CODE_HOME/io-test/kv_read_bench.py" cleanup --data-dir "$DATA_DIR"
fi

echo "PREFILL_PIPELINE_BENCH_COMPLETE result_dir=$RESULT_DIR"
