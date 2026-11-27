#!/usr/bin/env bash
# Snapshots of a long run as a network family of their own (METHOD_NOTES 2.62):
#   pid_snap_links.sh <source family> <snapshot, e.g. 20> <new family>
# ckpt_-1_<new>s<k>e.pt -> ckpt_-1_<source>s<k>k<snapshot>e.pt for every seed k that has the snapshot; the raw (non-EMA)
# snapshot files are removed (never used). Links only: login node.
set -euo pipefail
S=/work/u2507115/pid_v2/seqnet
src=$1; snap=$2; dst=$3
n=0
for f in "$S"/ckpt_-1_${src}s*k${snap}e.pt; do
  [ -e "$f" ] || continue
  k=$(basename "$f" | sed -E "s/^ckpt_-1_${src}s([0-9]+)k${snap}e\.pt$/\1/")
  ln -sf "$(basename "$f")" "$S/ckpt_-1_${dst}s${k}e.pt"
  rm -f "$S/ckpt_-1_${src}s${k}k${snap}.pt"
  n=$((n + 1))
done
echo "$dst: $n network(s) linked to the ${snap}k snapshots of $src"
