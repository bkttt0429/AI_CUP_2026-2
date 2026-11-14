#!/usr/bin/env bash
# METHOD_NOTES 2.107, probe P6, step 1: the supervised projection fitted JOINTLY over [pooled, played-point, policy-expected
# played-point] means (rows of scripts/pid_pol_extract.py; players of folds 3 and 4 only, never fold 0), then the per-ply
# features of the probe-B games under it, with and without four policy scalars (two stores from one pass):
#   /work/u2507115/pid_v2/probep/ply_j   64 dims        /work/u2507115/pid_v2/probep/ply_js   64 dims + 4 scalars
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 B=/work/u2507115/pid_v2/probeb D=/work/u2507115/pid_v2/probep rc
  if [ ! -s $D/proj3.npz ]; then
    python -I $R2/scripts/pid_sup_fit.py --rows $D/rows --blocks g x e --fit-folds 3 4 --k 64 --out $D/proj3.npz || { echo "projection fit FAILED"; return 1; }
  fi
  bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_sup_ply_extract.py --keys $B/keys.txt --proj $D/proj3.npz --scalars --out $D/ply_js --out2 $D/ply_j --workers 10
  rc=$?
  echo "parts: $(ls $D/ply_j | grep -c 'npz$') / $(ls $D/ply_js | grep -c 'npz$'); $(du -sh $D | cut -f1)"
  return $rc
}
main "$@"; exit $?
