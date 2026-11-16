#!/usr/bin/env bash
set -euo pipefail
study=/home/u2507115/.cache/pidchoice_wt
run_dir=$1
python -I "$study/scripts/pid_choice_data_audit.py" --run-dir "$run_dir"
python -I "$study/scripts/pid_choice_audit.py" --run-dir "$run_dir" --stage calibration
python -I "$study/scripts/pid_choice_eval.py" --run-dir "$run_dir" --stage baseline-check
