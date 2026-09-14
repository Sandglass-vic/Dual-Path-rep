#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
common=(
  TASK_NAME=test-manual-launch
  PREFILL_HOSTS="p0 p1"
  DECODE_HOSTS="d0 d1"
  ALLOC_JOB_ID=12345
  DUALPATH_SKIP_ALLOCATION_CHECK=1
  PD_DRY_RUN=1
)

output=$(env "${common[@]}" DUALPATH_HOSTNAME=p1 "$ROOT/scripts/pd_benchmark.sh" server)
grep -q 'role=prefill id=1' <<<"$output"
grep -q -- '--gres=gpu:8' <<<"$output"
grep -q -- '--quit-on-interrupt' <<<"$output"

output=$(env "${common[@]}" DUALPATH_HOSTNAME=d0 "$ROOT/scripts/pd_benchmark.sh" server)
grep -q 'role=decode id=0' <<<"$output"

output=$(env "${common[@]}" DUALPATH_HOSTNAME=p1 "$ROOT/scripts/pd_benchmark.sh" prefill)
grep -q 'role=prefill id=1' <<<"$output"

output=$(env "${common[@]}" DUALPATH_HOSTNAME=d0 "$ROOT/scripts/pd_benchmark.sh" decode)
grep -q 'role=decode id=0' <<<"$output"

output=$(env "${common[@]}" DUALPATH_HOSTNAME=c0 "$ROOT/scripts/pd_benchmark.sh" client)
grep -q 'role=client id=0' <<<"$output"
grep -q -- '--gres=none' <<<"$output"

if env "${common[@]}" DUALPATH_HOSTNAME=p0 "$ROOT/scripts/pd_benchmark.sh" client >/dev/null 2>&1; then
  echo "client role unexpectedly accepted an engine node" >&2
  exit 1
fi

if env "${common[@]}" DUALPATH_HOSTNAME=p0 "$ROOT/scripts/pd_benchmark.sh" decode >/dev/null 2>&1; then
  echo "decode role unexpectedly accepted a prefill node" >&2
  exit 1
fi

if env "${common[@]}" PREFILL_HOSTS="p0 d0" DUALPATH_HOSTNAME=p0 \
  "$ROOT/scripts/pd_benchmark.sh" server >/dev/null 2>&1; then
  echo "duplicate engine host unexpectedly accepted" >&2
  exit 1
fi

output=$(env PD_DRY_RUN=1 TASK_NAME=test-alloc DATASET=32K AGENTS=2 \
  "$ROOT/scripts/pd_alloc.sh" 1 2)
grep -q 'Requesting 1P2D + 1 client: 4 nodes' <<<"$output"
grep -q -- '--nodes=4' <<<"$output"
grep -q -- 'pd_alloc.sh' <<<"$output"

allocation_root=$(mktemp -d)
trap 'rm -rf "$allocation_root"' EXIT
output=$(env SLURM_JOB_ID=24680 SLURM_JOB_NODELIST=test-nodes \
  DUALPATH_ALLOCATED_HOSTS='p0 d0 d1 c0' PREFILL_NODES=1 DECODE_NODES=2 \
  TASK_NAME=test-map DATASET=32K AGENTS=2 REPETITION=1 \
  DUALPATH_L3_READ_PATH=decode ABLATION_PROFILE=full IMAGE=/tmp/test.sqsh \
  ENGINE_CPUS=112 CLIENT_CPUS=8 RUNTIME_HOME="$allocation_root" \
  PD_ALLOCATION_NO_SHELL=1 PD_KEEP_ALLOCATION_ENV=1 \
  "$ROOT/scripts/pd_alloc.sh" __allocation_shell)
grep -q 'p0 .*prefill-0' <<<"$output"
grep -q 'd0 .*decode-0' <<<"$output"
grep -q 'd1 .*decode-1' <<<"$output"
grep -q 'c0 .*client' <<<"$output"
grep -q '^  ssh p0$' <<<"$output"
grep -q '^  cd .*pd_benchmark.sh prefill$' <<<"$output"
grep -q '^  ssh d0$' <<<"$output"
grep -q '^  cd .*pd_benchmark.sh decode$' <<<"$output"
grep -q '^  ssh c0$' <<<"$output"
grep -q '^  cd .*DATASET=32K AGENTS=1 REPETITION=1 DUALPATH_L3_READ_PATH=decode scripts/pd_benchmark.sh client$' <<<"$output"
grep -q 'export PREFILL_HOSTS=p0' "$allocation_root/config/allocations/24680.env"
grep -q 'export DECODE_HOSTS=d0\\ d1' "$allocation_root/config/allocations/24680.env"

env SLURM_JOB_ID=24681 SLURM_JOB_NODELIST=test-nodes \
  DUALPATH_ALLOCATED_HOSTS='p0 d0 c0' PREFILL_NODES=1 DECODE_NODES=1 \
  TASK_NAME=test-cleanup DATASET=32K AGENTS=1 REPETITION=1 \
  DUALPATH_L3_READ_PATH=prefill ABLATION_PROFILE=full IMAGE=/tmp/test.sqsh \
  ENGINE_CPUS=112 CLIENT_CPUS=8 RUNTIME_HOME="$allocation_root" \
  PD_ALLOCATION_NO_SHELL=1 \
  "$ROOT/scripts/pd_alloc.sh" __allocation_shell >/dev/null
if [[ -e $allocation_root/config/allocations/24681.env ]]; then
  echo "allocation environment was not removed on exit" >&2
  exit 1
fi

fake_bin=$(mktemp -d)
fake_srun_log=$allocation_root/fake-srun.log
cat >"$fake_bin/srun" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >>"$FAKE_SRUN_LOG"
printf '\n' >>"$FAKE_SRUN_LOG"
while (( $# > 0 )) && [[ $1 == --* ]]; do
  shift
done
exec "$@"
EOF
chmod +x "$fake_bin/srun"
env PATH="$fake_bin:$PATH" FAKE_SRUN_LOG="$fake_srun_log" \
  SLURM_JOB_ID=24682 SLURM_JOB_NODELIST=test-nodes \
  DUALPATH_ALLOCATED_HOSTS='p0 d0 c0' PREFILL_NODES=1 DECODE_NODES=1 \
  TASK_NAME=test-remote-config DATASET=32K AGENTS=1 REPETITION=1 \
  DUALPATH_L3_READ_PATH=prefill ABLATION_PROFILE=full IMAGE=/tmp/test.sqsh \
  ENGINE_CPUS=112 CLIENT_CPUS=8 CODE_HOME="$ROOT" RUNTIME_HOME="$allocation_root" \
  PD_FORCE_CONFIG_SRUN=1 PD_ALLOCATION_NO_SHELL=1 \
  "$ROOT/scripts/pd_alloc.sh" __allocation_shell >/dev/null
grep -q -- '--nodelist=c0' "$fake_srun_log"
grep -q -- '__write_allocation_config' "$fake_srun_log"
if [[ -e $allocation_root/config/allocations/24682.env ]]; then
  echo "remote allocation environment was not removed on exit" >&2
  exit 1
fi

config=$(mktemp)
printf '%s\n' \
  'export ALLOC_JOB_ID=12345' \
  'export TASK_NAME=test-auto-config' \
  'export PREFILL_HOSTS=p0' \
  'export DECODE_HOSTS=d0\ d1' \
  'export CLIENT_HOST=c0' \
  'export ROUTER_HOST=c0' \
  'export DATASET=32K' \
  'export AGENTS=1' \
  'export REPETITION=1' \
  'export DUALPATH_L3_READ_PATH=prefill' \
  'export IMAGE=/tmp/test.sqsh' \
  'export ENGINE_CPUS=112' \
  'export CLIENT_CPUS=8' >"$config"
output=$(env ALLOC_JOB_ID=12345 PD_ALLOCATION_ENV="$config" \
  DUALPATH_SKIP_ALLOCATION_CHECK=1 PD_DRY_RUN=1 DUALPATH_HOSTNAME=d1 \
  "$ROOT/scripts/pd_benchmark.sh")
grep -q 'role=decode id=1' <<<"$output"

echo "PASS: manual salloc role resolution"
