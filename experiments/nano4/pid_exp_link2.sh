#!/usr/bin/env bash
# METHOD_NOTES 2.166 (second run): the linkage probe with a valid positive control (two held-out games per training
# player) and the training prototypes saved for the pair-level analysis (scripts/pid_probe_link2.py).
# From the AI_CUP_2026-1 checkout, one GPU, 48 cores for the loader. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 J=${SLURM_JOB_ID:-local} W=/work/u2507115/pid_v2/pair_eval
  python -I -m py_compile $R2/scripts/pid_probe_link.py $R2/scripts/pid_probe_link2.py || { echo "a script does not compile"; return 1; }
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_probe_link.py --smoke --nets o22Ss0 --alias o22Fs0 --protos $W/link_protos_smoke.npz --out $W/link_smoke.json > logs/nano4/link2-$J-smoke.log 2>&1 \
    || { echo "start-up check FAILED:"; tail -25 logs/nano4/link2-$J-smoke.log; return 1; }
  python -I $R2/scripts/pid_probe_link2.py --protos $W/link_protos_smoke.npz --out $W/link2_smoke.json >> logs/nano4/link2-$J-smoke.log 2>&1 || { echo "start-up check of the pair analysis FAILED:"; tail -25 logs/nano4/link2-$J-smoke.log; return 1; }
  echo "start-up checks passed"
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_probe_link.py --protos $W/link_protos.npz --out $R2/results/final/v2_probe_link.json || return 1
  echo; echo "===== pair-level analysis"
  python -I $R2/scripts/pid_probe_link2.py --protos $W/link_protos.npz --out $R2/results/final/v2_probe_link2.json
}
main "$@"; exit $?
