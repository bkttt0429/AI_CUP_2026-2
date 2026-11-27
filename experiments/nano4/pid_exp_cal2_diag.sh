#!/usr/bin/env bash
# METHOD_NOTES 2.238: the assignment of the multi-game questions and the prior of the 1-game questions were set on 11-15
# with the scores of that day; this re-runs the three tools of 11-15 under a current document (seen hold-out seeds 0-3):
#   1. scripts/pid_v2_assign_struct.py   pseudo-pool emulation of the assignment methods by contamination + the label-free
#                                        bound on repeated owners in the REAL pools (T_m)
#   2. scripts/pid_v2_assign_tune_old.py the same methods on the old-pool simulation (has 5-game questions)
#   3. scripts/pid_probe_overlap.py      the temperature T of the fused scores (1-game questions) and the share estimate
#   pid_exp_cal2_diag.sh <document> [<pseudo-pool vector folder> [<old-pool vector folder>]]
# Results: results/final/v2_assign_struct_<doc>.json, v2_assign_tune_old_<doc>.json, v2_probe_overlap_<doc>.json.
# Nothing here chooses or writes a file. From the AI_CUP_2026-1 checkout. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 d=$1 ft=${2:-/work/u2507115/pid_v2/vtmp8/ft} old=${3:-/work/u2507115/pid_v2/vtmp8/old} rc=0 free
  [ -s $R2/results/pid_fusion_$d.json ] || { echo "no document $d"; return 1; }
  free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')
  [ "$free" -ge 4800 ] || { echo "/work has only $free MB free: not started"; return 1; }
  echo "== 1. assignment emulation + real-pool bound, document $d ($(date +%T))"
  python -I $R2/scripts/pid_v2_assign_struct.py --doc "$d" --ft-dir "$ft" --seeds 0 1 2 3 --out $R2/results/final/v2_assign_struct_$d.json 2>&1 | grep -vE '^\s*$' | tail -n 60 | cut -c1-260 || rc=1
  echo "== 2. old-pool simulation ($(date +%T))"
  python -I $R2/scripts/pid_v2_assign_tune_old.py --doc "$d" --ft-dir "$old" --lams 1 2 3 4 --taus 0.5 1 --out $R2/results/final/v2_assign_tune_old_$d.json 2>&1 | grep -vE '^\s*$' | tail -n 40 | cut -c1-260 || rc=1
  echo "== 3. temperature and share of the 1-game questions ($(date +%T))"
  python -I $R2/scripts/pid_probe_overlap.py --doc "$d" --ft-dir "$ft" --seeds 0 1 2 3 --out $R2/results/final/v2_probe_overlap_$d.json 2>&1 | grep -vE '^\s*$' | tail -n 40 | cut -c1-260 || rc=1
  echo "done at $(date +%T), rc $rc"
  return $rc
}
main "$@"; exit $?
