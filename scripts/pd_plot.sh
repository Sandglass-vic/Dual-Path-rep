#!/usr/bin/env bash
# Post-process one completed manual benchmark run on a node with Lustre access.
set -euo pipefail

CODE_HOME=${CODE_HOME:-/labhome/sezhou/dual-path}
RUNTIME_HOME=${RUNTIME_HOME:-/mnt/lustre/users/sezhou/dual-path}
TASK_NAME=${TASK_NAME:-${1:-}}
: "${TASK_NAME:?Usage: TASK_NAME=<run> scripts/pd_plot.sh, or pass the run name as argument}"
[[ $TASK_NAME =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || {
  echo "Invalid TASK_NAME: $TASK_NAME" >&2
  exit 2
}

if [[ -n ${ALLOC_JOB_ID:-} ]]; then
  JOB_RUNTIME_DIR=${JOB_RUNTIME_DIR:-$RUNTIME_HOME/test/$ALLOC_JOB_ID}
fi
if [[ -z ${RESULT_DIR:-} && -n ${JOB_RUNTIME_DIR:-} ]]; then
  RESULT_DIR=$JOB_RUNTIME_DIR/$TASK_NAME/logs
fi
if [[ -z ${LOG_DIR:-} && -n ${JOB_RUNTIME_DIR:-} ]]; then
  LOG_DIR=$JOB_RUNTIME_DIR
fi
RESULT_DIR=${RESULT_DIR:-}
LOG_DIR=${LOG_DIR:-}
[[ -n $RESULT_DIR && -n $LOG_DIR ]] || {
  echo "Set ALLOC_JOB_ID, or explicitly set RESULT_DIR and LOG_DIR." >&2
  exit 2
}
REPO_RESULT_DIR=${REPO_RESULT_DIR:-$CODE_HOME/results/latency-benchmark/$TASK_NAME}
[[ -d $RESULT_DIR ]] || { echo "Result directory does not exist: $RESULT_DIR" >&2; exit 2; }
[[ -d $LOG_DIR ]] || { echo "Log directory does not exist: $LOG_DIR" >&2; exit 2; }

python3 "$CODE_HOME/scripts/benchmark/collect_benchmark_metrics.py" \
  --result-dir "$RESULT_DIR" --log-dir "$LOG_DIR" \
  --repo-result-dir "$REPO_RESULT_DIR" --code-home "$CODE_HOME"
echo "BENCHMARK_PLOT_COMPLETE result_dir=$REPO_RESULT_DIR"
