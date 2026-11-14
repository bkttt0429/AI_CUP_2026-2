#!/usr/bin/env bash
# One submission file from a fusion document of the v2 pipeline (METHOD_NOTES 2.37): soft assignment over the questions of
# 2 / 3 / 5 games, old pools from the fixed old submission.   pid_v2_make_file.sh <document name> <upload file name>
# Writes results/submission_<document name>s/submission_player.csv, copies it to upload/<upload file name> and prints how it
# differs from the live final file (pid_v2x_ns.csv, Public 0.9035072, since 2026-11-14 19:00) and from the final before it.
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
doc=$1
name=$2
out=$R2/results/submission_${doc}s/submission_player.csv
python -I $R2/scripts/predict_pid_v2.py --tests-dir $R2/data/tests_1111 --fusion $R2/results/pid_fusion_$doc.json --trunk-store $R2/data/trunk/tests_1111_v2 \
  --old-submission $R2/results/submission_old_embt_assign/submission_player.csv --assign big --big-lam 1 --out $out
cp $out $R2/upload/$name
python -I $R2/scripts/pid_diff_files.py $R2/upload/pid_v2x_ns.csv $R2/upload/$name
python -I $R2/scripts/pid_diff_files.py $R2/upload/pid_v2sl_alls.csv $R2/upload/$name
