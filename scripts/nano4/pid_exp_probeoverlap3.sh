#!/usr/bin/env bash
# METHOD_NOTES 2.123, probe S3: the owners of the 1-game questions are not the owners of the 3-game questions.
#   pid_exp_probeoverlap3.sh real|old [fusion document, default v2o13c_9] [further options of the script]
#     real  scripts/pid_probe_overlap3.py      estimates on the real pools (no answers) + planted structure in the pseudo-pools
#     old   scripts/pid_probe_overlap3_old.py  planted structure in the old-pool simulation (real question sizes, true answers)
# From the AI_CUP_2026-1 checkout, CPU only. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 mode=$1 doc=${2:-v2o13c_9} script out
  shift; shift || true
  case $mode in
    real) script=pid_probe_overlap3.py; out=v2_probe_overlap3_$doc.json ;;
    old)  script=pid_probe_overlap3_old.py; out=v2_probe_overlap3_old_$doc.json ;;
    *) echo "mode real or old"; return 2 ;;
  esac
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -I $R2/scripts/$script --doc "$doc" --out $R2/results/final/$out "$@"
}
main "$@"; exit $?
