#!/usr/bin/env bash
# Submit one job as soon as the shared queue has room (fewer than <limit> jobs of this user), then print its id.
#   submit_when_room.sh <limit> <sbatch arguments...>          (run from the AI_CUP_2026-1 checkout; login node: submit/query only)
set -uo pipefail
limit=$1; shift
part=8gpus  # the limits are per partition (8gpus: 13 submitted; 16gpus: 9, jobs of at least 8 GPUs)
prev=""
for x in "$@"; do [ "$prev" = "-p" ] && part=$x; prev=$x; done
out=""
while [ -z "$out" ]; do
  n=$(squeue -h -u u2507115 -p "$part" 2>/dev/null | wc -l)
  if [ "$n" -lt "$limit" ]; then out=$(sbatch --parsable "$@" 2>/dev/null); fi
  [ -z "$out" ] && sleep 30
done
echo "$out"
