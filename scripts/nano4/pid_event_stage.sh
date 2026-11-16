#!/usr/bin/env bash
set -euo pipefail
study=/home/u2507115/.cache/pidchoice_wt
run_dir=$1
stage=$2
study_python=/home/u2507115/BK_AICUP_2026-1/.venv/bin/python
export AICUP1_ROOT=/home/u2507115/BK_AICUP_2026-1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
case "$stage" in
  pilot)
    "$study_python" -I -m pytest -q "$study/tests"
    "$study_python" -I "$study/scripts/pid_event_probe.py" --run-dir "$run_dir" --stage protocol
    "$study_python" -I "$study/scripts/pid_event_probe.py" --run-dir "$run_dir" --stage pilot
    ;;
  calibration)
    "$study_python" -I "$study/scripts/pid_event_probe.py" --run-dir "$run_dir" --stage extract
    "$study_python" -I "$study/scripts/pid_event_probe.py" --run-dir "$run_dir" --stage study
    ;;
  *) exit 2 ;;
esac
