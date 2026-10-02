#!/usr/bin/env bash
# Read-only InfiniBand QoS / VL inventory for g5/g6.
#
# Needs root (SMP/SA MADs go through /dev/infiniband/umad*, mode 0600).
# Sends only Get() MADs; never modifies SM, switch, or HCA state.
#
#   sudo bash scripts/infra/ib_qos_probe.sh [OUT_DIR]
#
# CA_PORTS overrides the probed devices. The CX8 compute HCAs run in
# multi-plane mode (NUM_OF_PLANES_P1=4): the aggregated mlx5_0/mlx5_4 only
# answer GSI (SA, PMA), SMPs need an SMI sub-device with one port per plane:
#
#   rdma dev add smi0 type SMI parent mlx5_0
#   rdma dev add smi1 type SMI parent mlx5_4
#   CA_PORTS="smi0 smi1 mlx5_1 mlx5_5" bash scripts/infra/ib_qos_probe.sh
set -uo pipefail

CODE_HOME=${CODE_HOME:-/data/home/yongxiao/dual-path}
OUT=${1:-$CODE_HOME/results/ib-qos-probe/$(hostname -s)-$(date +%Y%m%d-%H%M%S)}
CA_PORTS=${CA_PORTS:-"mlx5_0 mlx5_4 mlx5_1 mlx5_5"}

if [[ $EUID -ne 0 ]]; then
  echo "must run as root (umad devices are root-only)" >&2
  exit 2
fi

mkdir -p "$OUT"

run() {
  local name=$1
  shift
  {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
    "$@" 2>&1
    printf '[exit %d]\n' "$?"
  } >"$OUT/$name.txt"
}

run host hostname -f
run ibstat ibstat
run ibdev2netdev ibdev2netdev -v
run rdma-dev rdma dev show
run nfs-mounts findmnt -t nfs,nfs4
[[ -r /etc/exports ]] && run exports cat /etc/exports

probed=()
for ca in $CA_PORTS; do
  for sys in /sys/class/infiniband/"$ca"/ports/*; do
    [[ -d $sys ]] || continue
    [[ $(<"$sys/link_layer") == InfiniBand ]] || continue
    port=${sys##*/}
    lid=$(( $(<"$sys/lid") ))
    smlid=$(( $(<"$sys/sm_lid") ))
    q=(-C "$ca" -P "$port")
    tag=$ca-p$port
    probed+=("$tag")

    run "$tag-sminfo" sminfo "${q[@]}"
    run "$tag-sm-nodedesc" smpquery "${q[@]}" nodedesc "$smlid"
    run "$tag-sm-nodeinfo" smpquery "${q[@]}" nodeinfo "$smlid"

    run "$tag-local-portinfo" smpquery "${q[@]}" portinfo "$lid" "$port"
    run "$tag-local-sl2vl" smpquery "${q[@]}" sl2vl "$lid" "$port"
    run "$tag-local-vlarb" smpquery "${q[@]}" vlarb "$lid" "$port"

    # Directed route 0,<port> = the switch port facing this HCA port. Its
    # egress arbitration toward the HCA is what protects inbound traffic.
    dr=0,$port
    run "$tag-peer-nodeinfo" smpquery "${q[@]}" -D nodeinfo "$dr"
    run "$tag-peer-nodedesc" smpquery "${q[@]}" -D nodedesc "$dr"
    peer_port=$(awk -F. '/^LocalPort/ {gsub(/[^0-9]/, "", $NF); print $NF}' \
      "$OUT/$tag-peer-nodeinfo.txt")
    if [[ -n $peer_port ]]; then
      run "$tag-peer-portinfo" smpquery "${q[@]}" -D portinfo "$dr" "$peer_port"
      run "$tag-peer-sl2vl" smpquery "${q[@]}" -D sl2vl "$dr" "$peer_port"
      run "$tag-peer-vlarb" smpquery "${q[@]}" -D vlarb "$dr" "$peer_port"
    fi

    run "$tag-perf-xmtsl" perfquery "${q[@]}" -X "$lid" "$port"
    run "$tag-perf-rcvsl" perfquery "${q[@]}" -S "$lid" "$port"
    run "$tag-perf-vlxmitwait" perfquery "${q[@]}" --vlxmitcounters "$lid" "$port"

    run "$tag-sa-noderecords" saquery "${q[@]}" NR
    run "$tag-sa-mcmembers" saquery "${q[@]}" -g
    run "$tag-ibnetdiscover" ibnetdiscover "${q[@]}"
    run "$tag-iblinkinfo" iblinkinfo "${q[@]}" -l
  done
done

for ca in $CA_PORTS; do
  [[ -e /sys/class/infiniband/$ca/device ]] || continue
  # mlxconfig segfaults on SMI sub-devices; query the parent HCA instead.
  [[ $ca == smi* ]] || run "$ca-mlxconfig" mlxconfig -d "$ca" -e q
done

if [[ -n ${SUDO_UID:-} ]]; then
  chown -R "$SUDO_UID:${SUDO_GID:-$SUDO_UID}" "$OUT"
fi

{
  echo "# IB QoS probe $(hostname -s) $(date -Is)"
  for tag in "${probed[@]}"; do
    echo
    echo "## $tag"
    grep -hE '^(SMLid|VLCap|OperVLs|VLHighLimit|VLArbHighCap|VLArbLowCap|MtuCap|NeighborMTU)' \
      "$OUT/$tag-local-portinfo.txt"
    grep -h '^Node Description' "$OUT/$tag-sm-nodedesc.txt" | sed 's/^/SM /'
    grep -h '^Node Description' "$OUT/$tag-peer-nodedesc.txt" | sed 's/^/peer /'
  done
} >"$OUT/SUMMARY.txt" 2>/dev/null

echo "wrote $OUT"
cat "$OUT/SUMMARY.txt"
