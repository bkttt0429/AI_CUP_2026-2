#!/usr/bin/env bash
set -euo pipefail
study=/home/u2507115/.cache/pidchoice_wt
run_dir=/work/u2507115/pid_v2/choice_probe_20261116/2993db0b3ed8
study_python=/home/u2507115/BK_AICUP_2026-1/.venv/bin/python
export AICUP1_ROOT=/home/u2507115/BK_AICUP_2026-1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
case "$1" in
  pilot)
    "$study_python" -I -m pytest -q "$study/tests"
    "$study_python" -I "$study/scripts/pid_reference_policy.py" --run-dir "$run_dir" --stage prepare
    "$study_python" -I "$study/scripts/pid_reference_policy.py" --run-dir "$run_dir" --stage pilot
    ;;
  run)
    "$study_python" -I "$study/scripts/pid_reference_policy.py" --run-dir "$run_dir" --stage run
    ;;
  *) exit 2 ;;
esac
