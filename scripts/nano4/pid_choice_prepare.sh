#!/usr/bin/env bash
set -euo pipefail
study_root=/home/u2507115/.cache/pidchoice_wt
ledger_dir="$study_root/.superpowers/sdd/2026-11-16-choice-signal"
export AICUP1_ROOT=/home/u2507115/BK_AICUP_2026-1
python -I -m pytest -q "$study_root/tests/test_choice_pipeline.py" "$study_root/tests/test_choice_signal.py"
for spec in /work/u2507115:6144 /home/u2507115:3072; do
  mount=${spec%:*}
  threshold=${spec##*:}
  available=$(df -BM --output=avail "$mount" | awk 'NR==2 {gsub(/M/, ""); print $1}')
  [[ "$available" =~ ^[0-9]+$ ]] && (( available >= threshold )) || { echo "insufficient available space $mount: $available MiB"; exit 2; }
done
python -I "$study_root/scripts/pid_choice_manifest.py" --run-root /work/u2507115/pid_v2/choice_probe_20261116 > "$ledger_dir/run-path.txt"
run_dir=$(cat "$ledger_dir/run-path.txt")
printf '%s\n' "RUN_DIR=$run_dir"
python -I "$study_root/scripts/pid_choice_extract.py" --run-dir "$run_dir" --stage pilot --workers 10
