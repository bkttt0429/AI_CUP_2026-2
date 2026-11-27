#!/usr/bin/env bash
# METHOD_NOTES 2.139: what the remaining error is made of (scripts/pid_probe_bottleneck.py) - measurements on the official
# pseudo-pools of one fusion document; nothing is chosen.
#   pid_exp_bottleneck.sh [fusion document, default v2o19d_j] [further options of the script]
# From the AI_CUP_2026-1 checkout, CPU only (four worker processes, one per hold-out seed). Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 doc=${1:-v2o19d_j}
  shift || true
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -I $R2/scripts/pid_probe_bottleneck.py --doc "$doc" --out $R2/results/final/v2_probe_bottleneck_$doc.json "$@"
}
main "$@"; exit $?
