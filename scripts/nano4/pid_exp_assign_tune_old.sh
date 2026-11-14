#!/usr/bin/env bash
# METHOD_NOTES 2.122: the assignment check on the old-pool simulation (5-game questions included; scripts/pid_v2_assign_tune_old.py).
#   pid_exp_assign_tune_old.sh [document name]
# CPU work; from the AI_CUP_2026-1 checkout. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 doc=${1:-v2o13c_9}
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 python -I $R2/scripts/pid_v2_assign_tune_old.py --doc $doc --out $R2/results/final/v2_assign_tune_old_$doc.json
}
main "$@"; exit $?
