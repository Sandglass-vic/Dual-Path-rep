#!/usr/bin/env bash
# Manage the NFSv3/RDMA view of g5's /data export on an NFS client node (g6).
# The regular /data stays on NFSv4.1/TCP; NFSv4.1 would trunk a second mount
# onto that TCP session, so the RDMA view uses NFSv3, which has no client-id
# trunking and gets its own transport.
set -euo pipefail

SERVER=192.168.11.5
EXPORT=/data
MOUNT_POINT=/mnt/nfsrdma
RDMA_PORT=20049
OPTIONS=vers=3,proto=rdma,port=$RDMA_PORT,rsize=1048576,wsize=1048576,hard

usage() {
  cat <<EOF
Usage: scripts/pza100-nfsrdma.sh mount|unmount|status

Run on the NFS client node (g6); mount/unmount use sudo.
  mount    mount $SERVER:$EXPORT at $MOUNT_POINT over NFS/RDMA ($OPTIONS)
  unmount  unmount $MOUNT_POINT; refuses while processes still use it
  status   exit 0 when $MOUNT_POINT is an active RDMA mount
EOF
}

die() { echo "pza100-nfsrdma: $*" >&2; exit 2; }

mount_line() {
  awk -v mp="$MOUNT_POINT" '$2 == mp' /proc/mounts
}

is_rdma_mounted() {
  mount_line | grep -q 'proto=rdma'
}

show_status() {
  local line
  line=$(mount_line)
  if [[ -z $line ]]; then
    echo "NFSRDMA_NOT_MOUNTED mount_point=$MOUNT_POINT"
    return 1
  fi
  echo "$line"
  grep -A12 "mounted on $MOUNT_POINT " /proc/self/mountstats | grep -m1 'xprt:' || true
  if ! is_rdma_mounted; then
    echo "NFSRDMA_WRONG_TRANSPORT mount_point=$MOUNT_POINT" >&2
    return 1
  fi
  echo "NFSRDMA_ACTIVE mount_point=$MOUNT_POINT"
}

do_mount() {
  [[ $(stat -f -c %T "$EXPORT") != xfs ]] || \
    die "$EXPORT is local on $(hostname -s); run this on the NFS client, not the server"
  if is_rdma_mounted; then
    show_status
    return
  fi
  [[ -z $(mount_line) ]] || die "$MOUNT_POINT is mounted without proto=rdma; unmount it first"
  grep -qw '^rpcrdma' /proc/modules || sudo modprobe rpcrdma
  sudo mkdir -p "$MOUNT_POINT"
  sudo mount -t nfs -o "$OPTIONS" "$SERVER:$EXPORT" "$MOUNT_POINT"
  show_status
}

do_unmount() {
  if [[ -z $(mount_line) ]]; then
    echo "NFSRDMA_NOT_MOUNTED mount_point=$MOUNT_POINT"
    return
  fi
  local users
  users=$(sudo fuser -m "$MOUNT_POINT" 2>/dev/null | tr -s ' ' || true)
  if [[ -n ${users// /} ]]; then
    echo "pza100-nfsrdma: $MOUNT_POINT is still in use by PIDs:$users" >&2
    sudo fuser -vm "$MOUNT_POINT" >&2 || true
    exit 1
  fi
  sudo umount "$MOUNT_POINT"
  echo "NFSRDMA_UNMOUNTED mount_point=$MOUNT_POINT"
}

case ${1:-} in
  mount) do_mount ;;
  unmount|umount) do_unmount ;;
  status) show_status ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
