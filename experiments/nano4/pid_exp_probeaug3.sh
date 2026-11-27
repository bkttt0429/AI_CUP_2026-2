#!/usr/bin/env bash
# METHOD_NOTES 2.127, probe T3: lending the games of the 2- and 5-game questions to their assigned candidates, with every
# candidate's column put back where it was (de-biased), on top of the prior of the 1-game questions.
#   pid_exp_probeaug3.sh [fusion document, default v2o16c_b]
# Both instruments in one job, side by side:
#   scripts/pid_probe_aug3_old.py  the old-pool simulation (real question sizes, true answers)        -> results/final/v2_probe_aug3_old_<doc>.json
#   scripts/pid_probe_aug3.py      the official pseudo-pools (true answers) and the real pools (counts) -> results/final/v2_probe_aug3_<doc>.json
# From the AI_CUP_2026-1 checkout, CPU only (24 cores). Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 doc=${1:-v2o16c_b} J=${SLURM_JOB_ID:-local} rc=0
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  python -I $R2/scripts/pid_probe_aug3_old.py --doc "$doc" --procs 12 --workers 6 --out $R2/results/final/v2_probe_aug3_old_$doc.json > logs/nano4/probeaug3-$J-old.log 2>&1 &
  local p_old=$!
  python -I $R2/scripts/pid_probe_aug3.py --doc "$doc" --procs 4 --workers 6 --out $R2/results/final/v2_probe_aug3_$doc.json > logs/nano4/probeaug3-$J-real.log 2>&1 &
  local p_real=$!
  wait $p_old || { echo "the old-pool part failed"; rc=1; }
  echo "== old-pool simulation"; cat logs/nano4/probeaug3-$J-old.log
  wait $p_real || { echo "the pseudo-pool part failed"; rc=1; }
  echo "== official pseudo-pools and real pools"; cat logs/nano4/probeaug3-$J-real.log
  return $rc
}
main "$@"; exit $?
