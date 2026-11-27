#!/usr/bin/env bash
# METHOD_NOTES 2.201 / 2.204: the submission files of the fold-0 round's candidates, built into results/final/not_for_upload/
# (a file goes to upload/ by hand only after its candidate has passed the rule). Needs the real-pool vectors of every family
# (data/seq_ft; o44F-o47F: scripts/nano4/pid_exp_bigft.sh <family> real). The documents are rebuilt with
# data/day_scripts/build_v2h.py (real-pool vectors required), replacing the reading-only ones of pid_build_doc_names.py.
#   pid_exp_f0_files.sh <name>=<composition> ...
# Every file is compared with the live final upload/pid_v2fin_x_a2k_p3.csv. From the AI_CUP_2026-1 checkout.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 c n rc=0
  for c in "$@"; do
    n=${c%%=*}
    python -I $R2/data/day_scripts/build_v2h.py "$n" "${c#*=}" || { echo "document $n was not built"; rc=1; continue; }
    bash $R2/scripts/nano4/pid_exp_compfile.sh "$n" pid_v2fin_x_a2k_p3.csv || rc=1
  done
  ls -la $R2/results/final/not_for_upload/ | grep -E "pid_($(printf '%s|' "$@" | sed 's/=[^|]*//g; s/|$//'))_a2k_p3\.csv"
  echo "done at $(date +%T), rc $rc"
  return $rc
}
main "$@"; exit $?
