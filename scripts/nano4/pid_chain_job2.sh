#!/usr/bin/env bash
# As pid_chain_job.sh, with the LIVE file of 2026-11-14 19:00 as the reference (v2x_n = pid_v2x_ns.csv, Public 0.9035072;
# METHOD_NOTES 2.111). One Slurm job, to be submitted with -d afterany:<training jobs> or --begin.
#   LINKS="<family>:<seed>=<checkpoint file> ..."  optional, made first (only for files that exist)
#   RM="<file> ..."                                 optional, unused non-EMA twins in the checkpoint directory
#   pid_chain_job2.sh new <tag> <family> [processed families ...]
#        candidates <tag>_all = live + every given family x 3, <tag>_h = x 6, <tag>_p = <tag>_all without the first family
#   pid_chain_job2.sh combo <tag> <family> <name>=<composition> ...     LIVE in a composition = the live composition;
#        BASE = the live composition without its o12K term (to put another supervised-input family in its place)
# The first family is fine-tuned on the pools when its vectors are missing; everything else must have been processed.
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 S=/work/u2507115/pid_v2/seqnet l fam seed src mode=$1 tag=$2 first=$3 c cands=()
  local base="o7F*9+o9D*3+o3SL*3+o3S*3+o4S*3+o3BS*3+py*3+o1+o2+px2+pb+px+m101x"
  local live="o12K*6+$base"
  shift 3
  for l in ${RM:-}; do case $l in */*|..*) echo "skipped $l";; *) rm -f "$S/$l" && echo "removed $l";; esac; done
  for l in ${LINKS:-}; do
    fam=${l%%:*}; seed=${l#*:}; src=${seed#*=}; seed=${seed%%=*}
    if [ -s "$S/$src" ]; then ln -sf "$src" "$S/ckpt_-1_${fam}s${seed}e.pt"; echo "link ckpt_-1_${fam}s${seed}e.pt -> $src"; else echo "missing $src (no link)"; fi
  done
  mult() { local k=$1 s="" g; shift; for g in "$@"; do s="$s+$g*$k"; done; echo "${s#+}"; }
  if [ "$mode" = combo ]; then
    for c in "$@"; do c=${c//LIVE/$live}; cands+=("${c//BASE/$base}"); done
  else
    cands=("${tag}_all=$(mult 3 "$first" "$@")+$live" "${tag}_h=$(mult 6 "$first" "$@")+$live")
    if [ $# -gt 0 ]; then cands+=("${tag}_p=$(mult 3 "$@")+$live"); fi
  fi
  OUT=$tag bash $R2/scripts/nano4/pid_v2_newnets2.sh "$first" v2x_n "${cands[@]}"
}
main "$@"; exit $?
