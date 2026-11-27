#!/usr/bin/env bash
# As pid_v2_make_file2.sh, with the lending of METHOD_NOTES 2.123 as a further argument:
#   pid_v2_make_file3.sh <document name> <upload file name> <lam> <tau> <lend margin | none> [file to compare with]
#     lam, tau     the assignment of the multi-game questions (2 0.5 = the "a2k" files)
#     lend margin  multi-game questions whose Top-1 leads by at least this much lend their games to it; the 1-game questions
#                  are re-scored. "none" = no lending (must reproduce the pid_v2_make_file2.sh file byte for byte)
# Prints how the file differs from the final file (pid_v2o13c_9s.csv) and from the optional comparison file.
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
doc=$1; name=$2; lam=$3; tau=$4; lend=$5; cmp=${6:-}
out=$R2/results/submission_${name%.csv}/submission_player.csv
opts=(--assign big --big-lam "$lam" --big-tau "$tau")
[ "$lend" = none ] || opts+=(--lend-margin "$lend")
python -I $R2/scripts/predict_pid_v2.py --tests-dir $R2/data/tests_1111 --fusion $R2/results/pid_fusion_$doc.json --trunk-store $R2/data/trunk/tests_1111_v2 \
  --old-submission $R2/results/submission_old_embt_assign/submission_player.csv "${opts[@]}" --out $out
cp $out $R2/upload/$name
python -I $R2/scripts/pid_diff_files.py $R2/upload/pid_v2o13c_9s.csv $R2/upload/$name
[ -z "$cmp" ] || python -I $R2/scripts/pid_diff_files.py $R2/upload/$cmp $R2/upload/$name
echo "md5 $(md5sum $R2/upload/$name | cut -c1-32) $name"
