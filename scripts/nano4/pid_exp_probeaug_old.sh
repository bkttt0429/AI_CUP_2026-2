#!/usr/bin/env bash
# METHOD_NOTES 2.123, probe T2 part O: the lending of the multi-game questions' games on the old-pool simulation (real
# question sizes, true answers; scripts/pid_probe_aug_old.py).
#   pid_exp_probeaug_old.sh [fusion document, default v2o13c_9] [further options of the script]
# From the AI_CUP_2026-1 checkout, CPU only (nine worker processes). Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 doc=${1:-v2o13c_9}
  shift || true
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -I $R2/scripts/pid_probe_aug_old.py --doc "$doc" --out $R2/results/final/v2_probe_aug_old_$doc.json "$@"
}
main "$@"; exit $?
