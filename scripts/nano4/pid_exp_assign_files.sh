#!/usr/bin/env bash
# METHOD_NOTES 2.122: files of the final composition under other assignments of the multi-game questions, after a regression
# check (the soft assignment with lam = 1 must reproduce the final file byte for byte). Which file, if any, is offered is
# decided by the rule in the notes, not here.
#   pid_exp_assign_files.sh [document name]
# CPU work; from the AI_CUP_2026-1 checkout. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 doc=${1:-v2o13c_9} M
  M=$R2/scripts/nano4/pid_v2_make_file2.sh
  bash $M $doc pid_${doc}_chk.csv 1 > logs/nano4/assignfiles-${SLURM_JOB_ID:-local}-chk.log 2>&1 || { echo "regression build FAILED"; tail -n 5 logs/nano4/assignfiles-${SLURM_JOB_ID:-local}-chk.log; return 1; }
  if cmp -s $R2/upload/pid_${doc}_chk.csv $R2/upload/pid_${doc}s.csv; then echo "regression check OK: lam = 1 reproduces pid_${doc}s.csv"; rm -f $R2/upload/pid_${doc}_chk.csv; else echo "regression check FAILED: the lam = 1 file differs from pid_${doc}s.csv"; return 1; fi
  local spec name lam tau
  for spec in "a2 2" "a3 3" "a4 4" "h hard" "k none 0.5" "k1 none 1" "a3k 3 0.5" "a4k 4 0.5"; do
    read -r name lam tau <<< "$spec"
    bash $M $doc pid_${doc}_${name}.csv $lam ${tau:-} > logs/nano4/assignfiles-${SLURM_JOB_ID:-local}-$name.log 2>&1 || { echo "build $name FAILED"; continue; }
    echo "== file pid_${doc}_${name}.csv (lam $lam${tau:+, tau $tau}): $(grep -E 'questions with a different Top-1' logs/nano4/assignfiles-${SLURM_JOB_ID:-local}-$name.log | head -1) | $(grep -E '^md5' logs/nano4/assignfiles-${SLURM_JOB_ID:-local}-$name.log | tail -1)"
  done
}
main "$@"; exit $?
