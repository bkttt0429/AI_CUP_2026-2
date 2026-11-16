#!/usr/bin/env bash
set -euo pipefail
study=/home/u2507115/.cache/pidchoice_wt
run_dir=$1
study_python=/home/u2507115/BK_AICUP_2026-1/.venv/bin/python
"$study_python" -I "$study/scripts/pid_choice_data_audit.py" --run-dir "$run_dir"
"$study_python" -I "$study/scripts/pid_choice_audit.py" --run-dir "$run_dir" --stage calibration
"$study_python" -I "$study/scripts/pid_choice_eval.py" --run-dir "$run_dir" --stage baseline-check
