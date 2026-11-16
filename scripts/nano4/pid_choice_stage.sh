#!/usr/bin/env bash
set -euo pipefail
study=/home/u2507115/.cache/pidchoice_wt
run_dir=$1
stage=$2
shift 2
study_python=/home/u2507115/BK_AICUP_2026-1/.venv/bin/python
export AICUP1_ROOT=/home/u2507115/BK_AICUP_2026-1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
budget() {
  "$study_python" -I "$study/scripts/pid_choice_runtime.py" --run-dir "$run_dir" --job-file "$study/.superpowers/sdd/2026-11-16-choice-signal/jobs.txt"
}
budget
case "$stage" in
  reference)
    pool=$1
    shift
    for seed in "$@"; do
      "$study_python" -I "$study/scripts/pid_choice_audit.py" --run-dir "$run_dir" --stage reference --pool "$pool" --seed "$seed"
      budget
    done
    ;;
  evaluate|baseline-check|baseline-extra)
    "$study_python" -I "$study/scripts/pid_choice_eval.py" --run-dir "$run_dir" --stage "$stage" "$@"
    ;;
  *) echo "unsupported study stage: $stage" >&2; exit 2 ;;
esac
budget
