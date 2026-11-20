#!/usr/bin/env bash
# pid_chain_job4.sh with the faster chain pid_v2_newnets3f.sh (METHOD_NOTES 2.190). The original's header:
# As pid_chain_job3.sh, with the final file of 2026-11-15 04:32 as the reference (v2o13c_9 = pid_v2o13c_9s.csv, Public
# 0.9059854; METHOD_NOTES 2.121) and pid_v2_newnets3.sh as the chain (four hold-out seeds, files with both assignments).
#   LINKS="<family>:<seed>=<checkpoint file> ..."  optional, made first (only for files that exist)
#   REF=<fusion document>                           optional: the document the candidates are compared with (default v2o13c_9;
#                                                   v2o16c_b once pid_v2o16c_b_a2k.csv is the final file). LIVE stays the composition below.
#   pid_chain_job4.sh combo <tag> <family> <name>=<composition> ...     LIVE in a composition = the final's composition
#        (o13F*9+o12F*9+BASE); BASE = the composition without its supervised-input families of the first feature set
# The first family is fine-tuned when its vectors are missing; every other family of a composition must be complete.
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 S=/work/u2507115/pid_v2/seqnet l fam seed src mode=$1 tag=$2 first=$3 c cands=()
  local base="o7F*9+o9D*3+o3SL*3+o3S*3+o4S*3+o3BS*3+py*3+o1+o2+px2+pb+px+m101x"
  local live="o13F*9+o12F*9+$base"
  shift 3
  [ "$mode" = combo ] || { echo "only the combo mode exists here"; return 2; }
  for l in ${LINKS:-}; do
    fam=${l%%:*}; seed=${l#*:}; src=${seed#*=}; seed=${seed%%=*}
    if [ -s "$S/$src" ]; then ln -sf "$src" "$S/ckpt_-1_${fam}s${seed}e.pt"; echo "link ckpt_-1_${fam}s${seed}e.pt -> $src"; else echo "missing $src (no link)"; fi
  done
  for c in "$@"; do c=${c//LIVE/$live}; cands+=("${c//BASE/$base}"); done
  OUT=$tag bash $R2/scripts/nano4/pid_v2_newnets3f.sh "$first" "${REF:-v2o13c_9}" "${cands[@]}"
}
main "$@"; exit $?
