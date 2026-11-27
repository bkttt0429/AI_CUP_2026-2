#!/usr/bin/env bash
# METHOD_NOTES 2.123, probe T2: lend the games of the confidently assigned multi-game questions to their Top-1 candidates
# and re-score the 1-game questions (scripts/pid_probe_aug.py: official pseudo-pools with true answers, emulation on the real
# question games, and what it would change in the real pools).
#   pid_exp_probeaug.sh [fusion document, default v2o13c_9] [further options of the script]
# From the AI_CUP_2026-1 checkout, CPU only (seven worker processes). Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 doc=${1:-v2o13c_9}
  shift || true
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -I $R2/scripts/pid_probe_aug.py --doc "$doc" --out $R2/results/final/v2_probe_aug_$doc.json "$@"
}
main "$@"; exit $?
