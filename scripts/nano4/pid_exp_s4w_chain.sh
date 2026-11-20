#!/usr/bin/env bash
# METHOD_NOTES 2.186 / 2.187: the chains of the scale4 round (families o36-o39).
#   FAST=1 pid_exp_s4w_chain.sh <o36|o37|o38|o39>   the family's vectors only (real pools, pseudo-pools of four seeds, old-pool
#                                                   simulation; about half an hour) - what the 'all' job needs
#   pid_exp_s4w_chain.sh <o36|o37|o38|o39>          the full family chain: v2<fam>m_x (the final's composition with the original
#                                                   family replaced), _only, _ref1 = its twin of the previous round (o32-o35)
#   pid_exp_s4w_chain.sh all                        v2fin2_x = the four original families replaced by o36-o39 (PRIMARY)
#                                                   v2fin2_a = original x9 + new x9 each (recorded only)
# 'all' compares with v2fin_x (REF_DOC overrides), the family chains with v2bgm_d2. Every file a chain writes into upload/ is moved to
# results/final/not_for_upload/
# (a passing candidate is rebuilt with pid_v2_make_file4.sh). From the AI_CUP_2026-1 checkout, 8 GPUs.
# Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 S=/work/u2507115/pid_v2/seqnet fam=$1 J=${SLURM_JOB_ID:-local} orig twin links="" i n free f
  local FINAL="o23F*18+o22F*18+o21F*18+o19F*18+o18F*9+o17F*9+o16F*9+o13F*9+o12F*9+o7F*9"
  free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')
  [ "$free" -ge 3000 ] || { echo "/work has only $free MB free: not started"; return 1; }
  mkdir -p $R2/results/final/not_for_upload
  if [ "$fam" = all ]; then
    for f in o36F o37F o38F o39F; do
      n=$(ls $S | grep -c "^ckpt_-1_${f}s[0-9]e\.pt$")
      [ "$n" -ge 7 ] || { echo "family $f has only $n networks linked"; return 1; }
    done
    local all="o37F*18+o36F*18+o39F*18+o38F*18+o18F*9+o17F*9+o16F*9+o13F*9+o12F*9+o7F*9"
    local add="o23F*9+o37F*9+o22F*9+o36F*9+o21F*9+o39F*9+o19F*9+o38F*9+o18F*9+o17F*9+o16F*9+o13F*9+o12F*9+o7F*9"
    # METHOD_NOTES 2.190: the faster chain (pid_chain_job4f.sh: split-parallel pseudo-pools, old-pool simulation and files at the
    # same time; checked against the original chain on v2fin, 2026-11-17 16:42: every number and all six files identical), and the
    # reference is v2fin_x - the candidate of the merged retrain that passed the rule at 16:24 (notes 2.186: "compared with it
    # once it has passed"). The comparison with the live final v2bgm_d2 is run next to it by hand (pid_v2_offsim_par.py).
    REF=${REF_DOC:-v2fin_x} bash $R2/scripts/nano4/pid_chain_job4f.sh combo v2fin2 o36F "v2fin2_x=$all" "v2fin2_a=$add"
    local rc=$?
    for f in $R2/upload/pid_v2fin2_*.csv; do [ -e "$f" ] && mv -f "$f" $R2/results/final/not_for_upload/; done
    return $rc
  fi
  case "$fam" in
    o36) orig=o22F; twin=o32F; for i in 0 1 2 3; do links+=" o36F:$i=ckpt_-1_o36Ss${i}e.pt o36F:$((i + 4))=ckpt_-1_o36SMs${i}e.pt"; done ;;
    o37) orig=o23F; twin=o33F ;;
    o38) orig=o19F; twin=o34F ;;
    o39) orig=o21F; twin=o35F ;;
    *) echo "usage: pid_exp_s4w_chain.sh <o36|o37|o38|o39|all>"; return 2 ;;
  esac
  if [ "$fam" != o36 ]; then
    for i in 0 1 2 3 4 5; do links+=" ${fam}F:$i=ckpt_-1_${fam}As${i}e.pt"; done
    links+=" ${fam}F:6=ckpt_-1_${fam}Cs0e.pt ${fam}F:7=ckpt_-1_${fam}Cs1e.pt"
  fi
  n=0; for i in $links; do [ -s "$S/${i#*=}" ] && n=$((n + 1)); done
  [ "$n" -ge 7 ] || { echo "only $n of the eight checkpoints of $fam exist: not started"; return 1; }
  local x=${FINAL/${orig}\*18/${fam}F*18} tag=v2${fam}m
  [ "$x" != "$FINAL" ] || { echo "the original family $orig is not in the final's composition"; return 1; }
  echo "family $fam ($n checkpoints) replaces $orig: $x"
  if [ "${FAST:-0}" = 1 ]; then  # METHOD_NOTES 2.187: only what the merged candidate needs - the family's vectors (real pools, pseudo-pools of four seeds, old-pool simulation); no per-family documents
    LINKS="$links" REF=v2bgm_d2 bash $R2/scripts/nano4/pid_chain_job4.sh combo $tag ${fam}F
    return $?
  fi
  LINKS="$links" REF=v2bgm_d2 bash $R2/scripts/nano4/pid_chain_job4.sh combo $tag ${fam}F "${tag}_x=$x" "${tag}_only=${fam}F*18" "${tag}_ref1=${twin}*18"
  local rc=$?
  for f in $R2/upload/pid_${tag}_*.csv; do [ -e "$f" ] && mv -f "$f" $R2/results/final/not_for_upload/; done
  for f in ${tag}_ref1 ${tag}_only; do [ -s $R2/results/pid_fusion_$f.json ] || { echo "no document $f: the one-family comparison is skipped"; return 1; }; done
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -I $R2/scripts/pid_v2_offsim.py --fusion $R2/results/pid_fusion_${tag}_ref1.json --alt $R2/results/pid_fusion_${tag}_only.json --steps 50 --seeds 0 1 2 3 --workers 8 \
    --out $R2/results/final/v2_offsim_${tag}_only_vs_${tag}_ref1_s0_1_2_3.json > logs/nano4/s4wcmp-$J-offsim.log 2>&1 || echo "pseudo-pool comparison failed"
  echo "== pseudo-pools, four seeds: ${fam}F alone (scale3b + scale4 accounts, window) vs its twin $twin alone (scale3b accounts, window)"
  grep -E '^(pub2|priv2|both): document weights|^   talt vs t50' logs/nano4/s4wcmp-$J-offsim.log | cut -c1-240
  echo "== old-pool simulation, 50 steps"
  python -I $R2/scripts/pid_v2_oldsim.py --docs ${tag}_ref1 ${tag}_only --step 50 --out $R2/results/final/v2_oldsim_${tag}_only.json || echo "old-pool simulation failed"
  return $rc
}
main "$@"; exit $?
