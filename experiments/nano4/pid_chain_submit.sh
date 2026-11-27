#!/usr/bin/env bash
# Submit one chain of new official networks against the live composition (scripts/nano4/pid_v2_newnets_recipe2.sh).
#   pid_chain_submit.sh <tag> <family> [families already processed ...]
# Optional links first:  LINKS="<new family>:<seed>=<checkpoint file name> ..."  (in /work/u2507115/pid_v2/seqnet)
# Optional:  CANCEL="<job id> ..." (a training job whose remaining networks are given up, to free its slot)
#            RM="<file name> ..."  (files of that directory to remove first: the unused non-EMA twins)
# One job with 8 GPUs on whichever of the two partitions has room first. From the AI_CUP_2026-1 checkout.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 S=/work/u2507115/pid_v2/seqnet tag=$1 l fam seed src
  shift
  for l in ${CANCEL:-}; do scancel "$l" && echo "cancelled job $l"; done
  for l in ${RM:-}; do case $l in */*|..*) echo "skipped $l";; *) rm -f "$S/$l" && echo "removed $l";; esac; done
  for l in ${LINKS:-}; do
    fam=${l%%:*}; seed=${l#*:}; src=${seed#*=}; seed=${seed%%=*}
    if [ -s "$S/$src" ]; then ln -sf "$src" "$S/ckpt_-1_${fam}s${seed}e.pt"; echo "link ckpt_-1_${fam}s${seed}e.pt -> $src"; else echo "missing $src (no link)"; fi
  done
  sbatch --parsable --export=ALL,TAG="$tag" --exclude='25a-hgpn[001,003-010,039,062,069,155,177]' -p 8gpus,16gpus --gres=gpu:8 -c 96 -t 04:00:00 \
    -J "p_chain_$tag" scripts/nano4/run.sbatch bash $R2/scripts/nano4/pid_v2_newnets_recipe2.sh "$@"
}
main "$@"; exit $?
