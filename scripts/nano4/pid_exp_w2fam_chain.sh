#!/usr/bin/env bash
# METHOD_NOTES 2.197: the chains of the second-window families (o40-o43; twins without the second window: o38F, o32F, o33F, o35F).
#   pid_exp_w2fam_chain.sh <o40|o41|o42|o43>   the family's vectors (real pools, pseudo-pools of four seeds, old-pool
#                                              simulation; pid_chain_job4.sh, vectors only) and then the family alone against
#                                              its twin alone (pid_exp_compcmp.sh: pseudo-pools of four seeds, old-pool simulation;
#                                              SKIP_TWIN=1 leaves that comparison out)
#   pid_exp_w2fam_chain.sh all                 candidates against v2fin_x (REF_DOC overrides), the faster verdict chain:
#                                                v2fin3_a = v2fin_x + o42F, o41F, o43F, o40F x18   (PRIMARY: the new families are ADDED,
#                                                           notes 2.196)
#                                                v2fin3_x = v2fin_x with o33F, o32F, o35F, o34F REPLACED by them (control)
#                                              and then both on the fresh hold-out seeds 4 and 5 (pid_exp_fresh45m.sh).
# Every file a chain writes into upload/ is moved to results/final/not_for_upload/ (a passing candidate is put back by hand).
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 S=/work/u2507115/pid_v2/seqnet fam=$1 J=${SLURM_JOB_ID:-local} twin links="" i n free f rc=0
  local OLD="o18F*9+o17F*9+o16F*9+o13F*9+o12F*9+o7F*9"
  free=$(df -BM --output=avail /home/u2507115 | tail -1 | tr -dc '0-9')
  [ "$free" -ge "${MIN_HOME_MB:-2000}" ] || { echo "/home has only $free MB free: not started (a family's vectors need 450 MB, the floor is 1.5 GB)"; return 1; }
  mkdir -p $R2/results/final/not_for_upload
  if [ "$fam" = all ]; then
    for f in o40F o41F o42F o43F; do
      n=$(ls $S | grep -c "^ckpt_-1_${f}s[0-9]e\.pt$")
      [ "$n" -ge 7 ] || { echo "family $f has only $n networks linked"; return 1; }
    done
    local ref=${REF_DOC:-v2fin_x}
    local add="o33F*18+o32F*18+o35F*18+o34F*18+o42F*18+o41F*18+o43F*18+o40F*18+$OLD"
    local rep="o42F*18+o41F*18+o43F*18+o40F*18+$OLD"
    REF=$ref bash $R2/scripts/nano4/pid_chain_job4f.sh combo v2fin3 o40F "v2fin3_a=$add" "v2fin3_x=$rep" || rc=1
    for f in $R2/upload/pid_v2fin3_*.csv; do [ -e "$f" ] && mv -f "$f" $R2/results/final/not_for_upload/; done
    echo "===== fresh hold-out seeds 4 and 5 ($(date +%T))"
    SLURM_JOB_ID=${J}9 bash $R2/scripts/nano4/pid_exp_fresh45m.sh "$ref" "o32F o33F o34F o35F o40F o41F o42F o43F" v2fin3_a v2fin3_x || rc=1
    return $rc
  fi
  case "$fam" in
    o40) twin=o38F; for i in 0 1 2 3 4 5; do links+=" o40F:$i=ckpt_-1_o40As${i}e.pt"; done; links+=" o40F:6=ckpt_-1_o40Cs0e.pt o40F:7=ckpt_-1_o40Cs1e.pt" ;;
    o41) twin=o32F; for i in 0 1 2 3; do links+=" o41F:$i=ckpt_-1_o41Ss${i}e.pt o41F:$((i + 4))=ckpt_-1_o41SMs${i}e.pt"; done ;;
    o42) twin=o33F; for i in 0 1 2 3 4 5; do links+=" o42F:$i=ckpt_-1_o42As${i}e.pt"; done; links+=" o42F:6=ckpt_-1_o42Cs0e.pt o42F:7=ckpt_-1_o42Cs1e.pt" ;;
    o43) twin=o35F; for i in 0 1 2 3 4 5; do links+=" o43F:$i=ckpt_-1_o43As${i}e.pt"; done; links+=" o43F:6=ckpt_-1_o43Cs0e.pt o43F:7=ckpt_-1_o43Cs1e.pt" ;;
    *) echo "usage: pid_exp_w2fam_chain.sh <o40|o41|o42|o43|all>"; return 2 ;;
  esac
  n=0; for i in $links; do [ -s "$S/${i#*=}" ] && n=$((n + 1)); done
  [ "$n" -ge 7 ] || { echo "only $n of the eight checkpoints of $fam exist: not started"; return 1; }
  local tag=v2${fam}p
  echo "family $fam ($n checkpoints), twin $twin"
  LINKS="$links" REF=v2fin_x bash $R2/scripts/nano4/pid_chain_job4.sh combo $tag ${fam}F || return 1
  # SKIP_TWIN=1: the twin's vectors are not on disk (o38F: deleted 2026-11-17 23:02 for the space; o40F against o38F was
  # read before that with pid_exp_shmfam.sh, notes 2.199)
  [ "${SKIP_TWIN:-0}" = 1 ] && { echo "the comparison with the twin $twin is skipped (SKIP_TWIN=1)"; return 0; }
  python -I $R2/data/day_scripts/build_v2h.py ${tag}_ref1 "${twin}*18" || return 1
  echo "===== the family alone against its twin alone ($(date +%T))"
  SLURM_JOB_ID=${J}8 bash $R2/scripts/nano4/pid_exp_compcmp.sh ${tag}_ref1 "${tag}_only=${fam}F*18"
}
main "$@"; exit $?
