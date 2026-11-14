#!/usr/bin/env bash
# METHOD_NOTES 2.102, probe P, step 1: policy-relative rows and per-ply features (scripts/pid_pol_extract.py) of the probe-B
# games (the simulated pools of folds 1 and 2 and up to 32 games of every player of folds 3 and 4; never fold 0).
#   pid_exp_probep_x.sh [first shard, default 0] [shards in total, default 8]
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 B=/work/u2507115/pid_v2/probeb D=/work/u2507115/pid_v2/probep
  mkdir -p $D
  bash $R2/scripts/nano4/shards_offset.sh 8 "${1:-0}" "${2:-8}" python -I $R2/scripts/pid_pol_extract.py --keys $B/keys.txt --proj $B/proj.npz --out $D --workers 10
  local rc=$?
  echo "parts: rows $(ls $D/rows | grep -c npz), per-ply $(ls $D/ply_r | grep -c npz) / $(ls $D/ply_fc | grep -c npz) / $(ls $D/ply_s | grep -c npz); $(du -sh $D | cut -f1)"
  return $rc
}
main "$@"; exit $?
