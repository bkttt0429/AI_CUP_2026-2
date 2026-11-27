#!/usr/bin/env bash
# Compositions of families that have ALREADY been through a chain, compared with the live composition (v2sl_all): the steps of
# pid_v2_newnets2.sh without new fine-tunes (they are skipped when the vectors exist).
#   pid_v2_combo.sh <result tag> <any processed family> <candidate>=<composition> ...
# In a composition the word LIVE stands for the live file's composition.
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 tag=$1 fam=$2 live="o3SL*3+o3S*3+o4S*3+o3BS*3+py*3+o1+o2+px2+pb+px+m101x" c cands=()
  shift 2
  for c in "$@"; do cands+=("${c//LIVE/$live}"); done
  OUT=$tag bash $R2/scripts/nano4/pid_v2_newnets2.sh "$fam" v2sl_all "${cands[@]}"
}
main "$@"; exit $?
