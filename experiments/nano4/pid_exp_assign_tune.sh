#!/usr/bin/env bash
# METHOD_NOTES 2.122: the strength of the one-to-one assignment of the multi-game questions, re-measured with the current
# scores on the official pseudo-pools (scripts/pid_v2_assign_tune.py). CPU work; one job.
#   pid_exp_assign_tune.sh [document name]
# From the AI_CUP_2026-1 checkout. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 doc=${1:-v2o13c_9}
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 python -I $R2/scripts/pid_v2_assign_tune.py --fusion $R2/results/pid_fusion_$doc.json --seeds 0 1 2 3 --out $R2/results/final/v2_assign_tune_$doc.json
}
main "$@"; exit $?
