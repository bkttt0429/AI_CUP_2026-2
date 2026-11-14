#!/usr/bin/env bash
# METHOD_NOTES 2.115, probe P7, step 1: the second feature set plus eight value scalars (scripts/pid_val_extract.py) for the
# probe-B games -> /work/u2507115/pid_v2/probep/ply_jv (float16, 76 dims).
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 B=/work/u2507115/pid_v2/probeb D=/work/u2507115/pid_v2/probep rc
  bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_val_extract.py --keys $B/keys.txt --proj $D/proj3.npz --out $D/ply_jv --workers 10
  rc=$?
  echo "parts: $(ls $D/ply_jv | grep -c 'npz$'); $(du -sh $D/ply_jv | cut -f1)"
  return $rc
}
main "$@"; exit $?
