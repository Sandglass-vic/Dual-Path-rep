#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
common=(
  TASK_NAME=test-manual-launch
  PREFILL_HOSTS="p0 p1"
  DECODE_HOSTS="d0 d1"
  CLIENT_HOST=c0
  ROUTER_HOST=c0
  ALLOC_JOB_ID=12345
  DUALPATH_L3_READ_PATH=prefill
  DUALPATH_SKIP_ALLOCATION_CHECK=1
  PD_DRY_RUN=1
)

output=$(env "${common[@]}" DUALPATH_HOSTNAME=p1 "$ROOT/scripts/pd_benchmark.sh" server)
grep -q 'role=prefill id=1' <<<"$output"
grep -q -- '--gres=gpu:4' <<<"$output"
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
grep -q -- '--overlap' <<<"$output"

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
grep -q 'Requesting 1P2D, client=decode: 3 nodes, TP=4, GPUs/engine=4' <<<"$output"
grep -q -- '--nodes=3' <<<"$output"
grep -q -- '--gres=gpu:4' <<<"$output"
grep -q -- 'pd_alloc.sh' <<<"$output"
if env PD_DRY_RUN=1 ENGINE_GPUS=2 TP_SIZE=4 \
  "$ROOT/scripts/pd_alloc.sh" 1 1 >/dev/null 2>&1; then
  echo "TP_SIZE larger than ENGINE_GPUS unexpectedly accepted" >&2
  exit 1
fi

allocation_root=$(mktemp -d)
trap 'rm -rf "$allocation_root"' EXIT
output=$(env SLURM_JOB_ID=24680 SLURM_JOB_NODELIST=test-nodes \
  DUALPATH_ALLOCATED_HOSTS='p0 d0 d1' PREFILL_NODES=1 DECODE_NODES=2 \
  CLIENT_PLACEMENT=decode ENGINE_GPUS=4 TP_SIZE=4 \
  TASK_NAME=test-map DATASET=32K AGENTS=2 REPETITION=1 \
  DUALPATH_L3_READ_PATH=decode DUAL_PATH_O_DIRECT_READ=1 \
  ABLATION_PROFILE=no-l3 IMAGE=/tmp/test.sqsh \
  ENGINE_CPUS=112 CLIENT_CPUS=8 RUNTIME_HOME="$allocation_root" \
  PD_ALLOCATION_NO_SHELL=1 PD_KEEP_ALLOCATION_ENV=1 \
  "$ROOT/scripts/pd_alloc.sh" __allocation_shell)
grep -q 'p0 .*prefill-0' <<<"$output"
grep -q 'd0 .*decode-0' <<<"$output"
grep -q 'd1 .*decode-1' <<<"$output"
grep -q 'd1 .*client (colocated with decode-1)' <<<"$output"
grep -q '^  ssh p0$' <<<"$output"
grep -q '^  cd .*DUALPATH_L3_READ_PATH=<prefill|decode> scripts/pd_benchmark.sh prefill$' <<<"$output"
grep -q '^  ssh d0$' <<<"$output"
grep -q '^  cd .*DUALPATH_L3_READ_PATH=<prefill|decode> scripts/pd_benchmark.sh decode$' <<<"$output"
grep -q '^  ssh d1$' <<<"$output"
grep -q '^  cd .*DATASET=32K AGENTS=1 REPETITION=1 DUALPATH_L3_READ_PATH=<prefill|decode> scripts/pd_benchmark.sh client$' <<<"$output"
grep -q 'export PREFILL_HOSTS=p0' "$allocation_root/config/allocations/24680.env"
grep -q 'export DECODE_HOSTS=d0\\ d1' "$allocation_root/config/allocations/24680.env"
grep -q 'export CLIENT_HOST=d1' "$allocation_root/config/allocations/24680.env"
grep -q 'export CLIENT_PLACEMENT=decode' "$allocation_root/config/allocations/24680.env"
grep -q 'export TP_SIZE=4' "$allocation_root/config/allocations/24680.env"
if grep -qE 'DUALPATH_L3_READ_PATH|DUAL_PATH_O_DIRECT_READ|ABLATION_PROFILE' \
  "$allocation_root/config/allocations/24680.env"; then
  echo "runtime settings unexpectedly persisted in allocation environment" >&2
  exit 1
fi

env SLURM_JOB_ID=24681 SLURM_JOB_NODELIST=test-nodes \
  DUALPATH_ALLOCATED_HOSTS='p0 d0' PREFILL_NODES=1 DECODE_NODES=1 \
  CLIENT_PLACEMENT=decode ENGINE_GPUS=4 TP_SIZE=4 \
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
  DUALPATH_ALLOCATED_HOSTS='p0 d0' PREFILL_NODES=1 DECODE_NODES=1 \
  CLIENT_PLACEMENT=decode ENGINE_GPUS=4 TP_SIZE=4 \
  TASK_NAME=test-remote-config DATASET=32K AGENTS=1 REPETITION=1 \
  DUALPATH_L3_READ_PATH=prefill ABLATION_PROFILE=full IMAGE=/tmp/test.sqsh \
  ENGINE_CPUS=112 CLIENT_CPUS=8 CODE_HOME="$ROOT" RUNTIME_HOME="$allocation_root" \
  PD_FORCE_CONFIG_SRUN=1 PD_ALLOCATION_NO_SHELL=1 \
  "$ROOT/scripts/pd_alloc.sh" __allocation_shell >/dev/null
grep -q -- '--nodelist=d0' "$fake_srun_log"
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
  DUALPATH_L3_READ_PATH=prefill \
  "$ROOT/scripts/pd_benchmark.sh")
grep -q 'role=decode id=1' <<<"$output"

colocated_config=$(mktemp)
printf '%s\n' \
  'export ALLOC_JOB_ID=12346' \
  'export PREFILL_HOSTS=p0' \
  'export DECODE_HOSTS=d0' \
  'export CLIENT_HOST=d0' \
  'export ROUTER_HOST=d0' \
  'export DUALPATH_L3_READ_PATH=prefill' \
  'export IMAGE=/tmp/test.sqsh' \
  'export ENGINE_CPUS=32' \
  'export CLIENT_CPUS=8' \
  'export ENGINE_GPUS=4' \
  'export TP_SIZE=4' >"$colocated_config"
output=$(env ALLOC_JOB_ID=12346 PD_ALLOCATION_ENV="$colocated_config" \
  DUALPATH_SKIP_ALLOCATION_CHECK=1 PD_DRY_RUN=1 DUALPATH_HOSTNAME=d0 \
  TP_SIZE=2 DUALPATH_L3_READ_PATH=prefill DUAL_PATH_O_DIRECT_READ=0 \
  ABLATION_PROFILE=no-l3 \
  "$ROOT/scripts/pd_benchmark.sh" decode)
grep -q 'role=decode id=0' <<<"$output"
grep -q -- '--exclusive' <<<"$output"
grep -q 'tp=4 l3_read=prefill o_direct=0' <<<"$output"
output=$(env ALLOC_JOB_ID=12346 PD_ALLOCATION_ENV="$colocated_config" \
  DUALPATH_SKIP_ALLOCATION_CHECK=1 PD_DRY_RUN=1 DUALPATH_HOSTNAME=d0 \
  DUALPATH_L3_READ_PATH=prefill \
  "$ROOT/scripts/pd_benchmark.sh" client)
grep -q 'role=client id=0' <<<"$output"
grep -q -- '--overlap' <<<"$output"
grep -q -- '--gres=none' <<<"$output"

echo "PASS: manual salloc role resolution"
