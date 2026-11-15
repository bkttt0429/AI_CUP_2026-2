#!/usr/bin/env bash
# As pid_v2_make_file2.sh, with the prior of the 1-game questions of METHOD_NOTES 2.123 (probe S3) as a further argument:
#   pid_v2_make_file4.sh <document name> <upload file name> <lam> <tau> <w3 | none> [T, default 1.6] [file to compare with]
#     lam, tau  the assignment of the multi-game questions (2 0.5 = the "a2k" files)
#     w3        probability that the owner of a 1-game question is one of the candidates assigned to the 3-game questions
#               (independent draws: 1/6; estimated 0.014 [0.001, 0.062]; 0.05 = the conservative setting). "none" = no prior
#               (must reproduce the pid_v2_make_file2.sh file byte for byte)
# Prints how the file differs from the final file (pid_v2o13c_9s.csv) and from the optional comparison file.
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
doc=$1; name=$2; lam=$3; tau=$4; w3=$5; temp=${6:-1.6}; cmp=${7:-}
out=$R2/results/submission_${name%.csv}/submission_player.csv
opts=(--assign big --big-lam "$lam" --big-tau "$tau")
[ "$w3" = none ] || opts+=(--one-prior3 "$w3" --prior-T "$temp")
python -I $R2/scripts/predict_pid_v2.py --tests-dir $R2/data/tests_1111 --fusion $R2/results/pid_fusion_$doc.json --trunk-store $R2/data/trunk/tests_1111_v2 \
  --old-submission $R2/results/submission_old_embt_assign/submission_player.csv "${opts[@]}" --out $out
cp $out $R2/upload/$name
python -I $R2/scripts/pid_diff_files.py $R2/upload/pid_v2o13c_9s.csv $R2/upload/$name
[ -z "$cmp" ] || python -I $R2/scripts/pid_diff_files.py $R2/upload/$cmp $R2/upload/$name
echo "md5 $(md5sum $R2/upload/$name | cut -c1-32) $name"
