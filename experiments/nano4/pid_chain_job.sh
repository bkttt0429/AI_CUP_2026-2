#!/usr/bin/env bash
# A chain as ONE Slurm job that needs nothing outside Slurm (METHOD_NOTES 2.111): submitted with -d afterany:<training jobs>
# or --begin, it makes the links itself and then runs the chain against the live composition. (The watchers of 2.106 died
# with the conversation's process group; Slurm dependencies do not.)
#   LINKS="<family>:<seed>=<checkpoint file> ..."  optional, made first (only for files that exist)
#   RM="<file> ..."                                 optional, unused non-EMA twins in the checkpoint directory
#   pid_chain_job.sh <tag> <family> [processed families ...]            -> scripts/nano4/pid_v2_newnets_recipe2.sh
#   pid_chain_job.sh combo <tag> <family> <name>=<composition> ...      -> scripts/nano4/pid_v2_combo.sh
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 S=/work/u2507115/pid_v2/seqnet l fam seed src
  for l in ${RM:-}; do case $l in */*|..*) echo "skipped $l";; *) rm -f "$S/$l" && echo "removed $l";; esac; done
  for l in ${LINKS:-}; do
    fam=${l%%:*}; seed=${l#*:}; src=${seed#*=}; seed=${seed%%=*}
    if [ -s "$S/$src" ]; then ln -sf "$src" "$S/ckpt_-1_${fam}s${seed}e.pt"; echo "link ckpt_-1_${fam}s${seed}e.pt -> $src"; else echo "missing $src (no link)"; fi
  done
  if [ "$1" = combo ]; then shift; bash $R2/scripts/nano4/pid_v2_combo.sh "$@"; else TAG=$1 bash $R2/scripts/nano4/pid_v2_newnets_recipe2.sh "${@:2}"; fi
}
main "$@"; exit $?
