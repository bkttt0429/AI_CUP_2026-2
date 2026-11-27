#!/usr/bin/env bash
# METHOD_NOTES 2.122, last step: the chosen assignment (soft lam 2 for the Top-1, Sinkhorn tau 0.5 for the order) on the
# old-pool simulation (5-game questions included), then its submission file.
# CPU work; from the AI_CUP_2026-1 checkout. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 doc=${1:-v2o13c_9}
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 python -I $R2/scripts/pid_v2_assign_tune_old.py --doc $doc --lams 1 2 3 4 --taus 0.5 1 --contam 0 0.01 0.02 0.03 0.05 --draws 200 \
    --out $R2/results/final/v2_assign_tune_old2_$doc.json || echo "old-pool check FAILED"
  bash $R2/scripts/nano4/pid_v2_make_file2.sh $doc pid_${doc}_a2k.csv 2 0.5 > logs/nano4/assignfinal-${SLURM_JOB_ID:-local}-a2k.log 2>&1 || { echo "build a2k FAILED"; tail -n 5 logs/nano4/assignfinal-${SLURM_JOB_ID:-local}-a2k.log; return 1; }
  echo "== file pid_${doc}_a2k.csv (lam 2, tau 0.5): $(grep -E 'questions with a different Top-1' logs/nano4/assignfinal-${SLURM_JOB_ID:-local}-a2k.log | head -1); $(grep -E 'questions with any different answer' logs/nano4/assignfinal-${SLURM_JOB_ID:-local}-a2k.log | head -1) | $(grep -E '^md5' logs/nano4/assignfinal-${SLURM_JOB_ID:-local}-a2k.log | tail -1)"
}
main "$@"; exit $?
