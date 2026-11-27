#!/usr/bin/env bash
# METHOD_NOTES 2.123, probe S: the share of the 1-game questions whose owner also owns a multi-game question (estimated
# without answers on the real pools, checked on planted structures in the pseudo-pools) and what a prior built on it is worth
# (scripts/pid_probe_overlap.py).
#   pid_exp_probeoverlap.sh [fusion document, default v2o13c_9] [further options of the script]
# From the AI_CUP_2026-1 checkout, CPU only. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 doc=${1:-v2o13c_9}
  shift || true
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -I $R2/scripts/pid_probe_overlap.py --doc "$doc" --out $R2/results/final/v2_probe_overlap_$doc.json "$@"
}
main "$@"; exit $?
