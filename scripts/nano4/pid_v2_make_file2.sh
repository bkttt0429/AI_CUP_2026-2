#!/usr/bin/env bash
# As pid_v2_make_file.sh, with the assignment of the multi-game questions as arguments (METHOD_NOTES 2.122):
#   pid_v2_make_file2.sh <document name> <upload file name> <lam | hard | none> [tau]
#     lam      soft assignment with that penalty (the submissions so far: 1)       hard   the hard one-to-one assignment
#     tau      Sinkhorn prices with that temperature; with a lam the Top-1 is the soft assignment's, with "none" Sinkhorn alone
# Prints how the file differs from the final file (pid_v2o13c_9s.csv, since 2026-11-15 04:32) and from the one before it.
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
doc=$1
name=$2
lam=$3
tau=${4:-}
out=$R2/results/submission_${name%.csv}/submission_player.csv
opts=(--assign big)
case $lam in hard|none) ;; *) opts+=(--big-lam "$lam") ;; esac
[ -n "$tau" ] && opts+=(--big-tau "$tau")
[ "$lam" = none ] && [ -z "$tau" ] && { echo "none needs a tau"; exit 2; }
[ "$lam" = hard ] && [ -n "$tau" ] && { echo "hard takes no tau (use a large lam with a tau)"; exit 2; }
python -I $R2/scripts/predict_pid_v2.py --tests-dir $R2/data/tests_1111 --fusion $R2/results/pid_fusion_$doc.json --trunk-store $R2/data/trunk/tests_1111_v2 \
  --old-submission $R2/results/submission_old_embt_assign/submission_player.csv "${opts[@]}" --out $out
cp $out $R2/upload/$name
python -I $R2/scripts/pid_diff_files.py $R2/upload/pid_v2o13c_9s.csv $R2/upload/$name
python -I $R2/scripts/pid_diff_files.py $R2/upload/pid_v2c2_bs.csv $R2/upload/$name
echo "md5 $(md5sum $R2/upload/$name | cut -c1-32) $name"
