#!/usr/bin/env bash
# METHOD_NOTES 2.139: follow-ups to the bottleneck measurements (scripts/pid_probe_bottleneck2.py): second-stage headroom,
# the colour gap by source, look-alike pairs, score normalisations, nearest-game evidence. Exploratory; nothing is chosen.
#   pid_exp_bottleneck2.sh [fusion document, default v2o19d_j] [further options of the script]
# From the AI_CUP_2026-1 checkout, CPU only (four worker processes). Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 doc=${1:-v2o19d_j}
  shift || true
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_probe_bottleneck2.py --doc "$doc" --out $R2/results/final/v2_probe_bottleneck2_$doc.json "$@"
}
main "$@"; exit $?
