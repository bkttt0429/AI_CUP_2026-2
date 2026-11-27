#!/usr/bin/env bash
# METHOD_NOTES 2.166: are official candidates people we already know (training players, or candidates of the original
# pool of the same side)? scripts/pid_probe_link.py - a start-up check on a few parts of the games, then the measurement.
# Measurement only. From the AI_CUP_2026-1 checkout, one GPU. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 J=${SLURM_JOB_ID:-local}
  python -I -m py_compile $R2/scripts/pid_probe_link.py || { echo "the script does not compile"; return 1; }
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_probe_link.py --smoke --nets o22Ss0 --alias o22Fs0 --out /work/u2507115/pid_v2/pair_eval/link_smoke.json > logs/nano4/link-$J-smoke.log 2>&1 \
    || { echo "start-up check FAILED:"; tail -25 logs/nano4/link-$J-smoke.log; return 1; }
  echo "start-up check passed: $(grep -c ': n ' logs/nano4/link-$J-smoke.log) groups printed"
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_probe_link.py --out $R2/results/final/v2_probe_link.json
}
main "$@"; exit $?
