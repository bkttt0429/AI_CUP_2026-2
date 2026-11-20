#!/usr/bin/env bash
# METHOD_NOTES 2.194: the submission file of a composition document that exists already (same steps and settings as the
# verdict chain pid_v2_newnets3f.sh: a2k assignment, 1-game prior 0.05 / 1.6, old pools from pid_v2o16c_b_a2k_p3.csv).
# The files are built in upload/ by the make_file scripts and moved AT ONCE to results/final/not_for_upload/ - a file
# goes back to upload/ by hand only after its candidate has passed the rule.
#   pid_exp_compfile.sh <document> [<file to compare with, in upload/> ...]
# From the AI_CUP_2026-1 checkout. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 n=$1 f rc=0
  shift
  local N=$R2/results/final/not_for_upload
  mkdir -p $N
  for f in pid_${n}s.csv pid_${n}_a2k.csv pid_${n}_a2k_p3.csv; do [ -e $R2/upload/$f ] && { echo "upload/$f exists already"; return 1; }; done
  bash $R2/scripts/nano4/pid_v2_make_file2.sh "$n" "pid_${n}s.csv" 1 && bash $R2/scripts/nano4/pid_v2_make_file2.sh "$n" "pid_${n}_a2k.csv" 2 0.5 \
    && bash $R2/scripts/nano4/pid_v2_make_file4.sh "$n" "pid_${n}_a2k_p3.csv" 2 0.5 0.05 1.6 pid_v2o16c_b_a2k_p3.csv || rc=1
  for f in pid_${n}s.csv pid_${n}_a2k.csv pid_${n}_a2k_p3.csv; do [ -e $R2/upload/$f ] && mv -f $R2/upload/$f $N/; done
  [ "$rc" = 0 ] || { echo "the file was not built"; return 1; }
  echo "== $N/pid_${n}_a2k_p3.csv: md5 $(md5sum $N/pid_${n}_a2k_p3.csv | cut -c1-32), $(wc -l < $N/pid_${n}_a2k_p3.csv) lines"
  for f in "$@"; do
    echo "== against upload/$f"
    python -I $R2/scripts/pid_diff_files.py $R2/upload/$f $N/pid_${n}_a2k_p3.csv || rc=1
  done
  return $rc
}
main "$@"; exit $?
