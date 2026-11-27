#!/usr/bin/env bash
# METHOD_NOTES 2.165: the chain of ONE scale3b family (the by-games family retrained with more external accounts) and its
# one-family comparison, in one job - so that it runs without the conversation.
#   pid_exp_s3b_chain.sh <o27|o28|o29|o30>        o27 <-> o22F, o28 <-> o23F, o29 <-> o19F, o30 <-> o21F
#   pid_exp_s3b_chain.sh all                      after the four: every original family replaced / every new family added
# Per family, against the final's document v2bgm_d2:
#   v2<fam>m_x     the final's composition with the original family replaced by the new one
#   v2<fam>m_only  the new family alone        v2<fam>m_ref1  the original family alone     (read one against the other)
# The files that a chain writes into upload/ are NOT candidates until the rule of notes 2.165 says so: all of them are moved
# to results/final/not_for_upload/ at the end (a passing candidate is rebuilt with pid_v2_make_file4.sh).
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 S=/work/u2507115/pid_v2/seqnet fam=$1 J=${SLURM_JOB_ID:-local} orig links="" i n free f
  local FINAL="o23F*18+o22F*18+o21F*18+o19F*18+o18F*9+o17F*9+o16F*9+o13F*9+o12F*9+o7F*9"
  free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')
  [ "$free" -ge 3000 ] || { echo "/work has only $free MB free: not started"; return 1; }
  mkdir -p $R2/results/final/not_for_upload
  if [ "$fam" = all ]; then
    for f in o27F o28F o29F o30F; do
      n=$(ls $S | grep -c "^ckpt_-1_${f}s[0-9]e\.pt$")
      [ "$n" -ge 7 ] || { echo "family $f has only $n networks linked"; return 1; }
    done
    local all="o28F*18+o27F*18+o30F*18+o29F*18+o18F*9+o17F*9+o16F*9+o13F*9+o12F*9+o7F*9"
    local add="o23F*9+o28F*9+o22F*9+o27F*9+o21F*9+o30F*9+o19F*9+o29F*9+o18F*9+o17F*9+o16F*9+o13F*9+o12F*9+o7F*9"
    REF=v2bgm_d2 bash $R2/scripts/nano4/pid_chain_job4.sh combo v2s3bm o27F "v2s3bm_all=$all" "v2s3bm_add=$add"
    local rc=$?
    for f in $R2/upload/pid_v2s3bm_*.csv; do [ -e "$f" ] && mv -f "$f" $R2/results/final/not_for_upload/; done
    return $rc
  fi
  case "$fam" in
    o27) orig=o22F; for i in 0 1 2 3; do links+=" o27F:$i=ckpt_-1_o27Ss${i}e.pt o27F:$((i + 4))=ckpt_-1_o27SMs${i}e.pt"; done ;;
    o28) orig=o23F ;;
    o29) orig=o19F ;;
    o30) orig=o21F ;;
    *) echo "usage: pid_exp_s3b_chain.sh <o27|o28|o29|o30|all>"; return 2 ;;
  esac
  if [ "$fam" != o27 ]; then
    for i in 0 1 2 3 4 5; do links+=" ${fam}F:$i=ckpt_-1_${fam}As${i}e.pt"; done
    links+=" ${fam}F:6=ckpt_-1_${fam}Cs0e.pt ${fam}F:7=ckpt_-1_${fam}Cs1e.pt"
  fi
  n=0; for i in $links; do [ -s "$S/${i#*=}" ] && n=$((n + 1)); done
  [ "$n" -ge 7 ] || { echo "only $n of the eight checkpoints of $fam exist: not started"; return 1; }
  local x=${FINAL/${orig}\*18/${fam}F*18} tag=v2${fam}m
  [ "$x" != "$FINAL" ] || { echo "the original family $orig is not in the final's composition"; return 1; }
  echo "family $fam ($n checkpoints) replaces $orig: $x"
  LINKS="$links" REF=v2bgm_d2 bash $R2/scripts/nano4/pid_chain_job4.sh combo $tag ${fam}F "${tag}_x=$x" "${tag}_only=${fam}F*18" "${tag}_ref1=${orig}*18"
  local rc=$?
  for f in $R2/upload/pid_${tag}_*.csv; do [ -e "$f" ] && mv -f "$f" $R2/results/final/not_for_upload/; done
  for f in ${tag}_ref1 ${tag}_only; do [ -s $R2/results/pid_fusion_$f.json ] || { echo "no document $f: the one-family comparison is skipped"; return 1; }; done
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -I $R2/scripts/pid_v2_offsim.py --fusion $R2/results/pid_fusion_${tag}_ref1.json --alt $R2/results/pid_fusion_${tag}_only.json --steps 50 --seeds 0 1 2 3 --workers 8 \
    --out $R2/results/final/v2_offsim_${tag}_only_vs_${tag}_ref1_s0_1_2_3.json > logs/nano4/s3bcmp-$J-offsim.log 2>&1 || echo "pseudo-pool comparison failed"
  echo "== pseudo-pools, four seeds: ${fam}F alone (more external accounts) vs $orig alone"
  grep -E '^(pub2|priv2|both): document weights|^   talt vs t50' logs/nano4/s3bcmp-$J-offsim.log | cut -c1-240
  echo "== old-pool simulation, 50 steps"
  python -I $R2/scripts/pid_v2_oldsim.py --docs ${tag}_ref1 ${tag}_only --step 50 --out $R2/results/final/v2_oldsim_${tag}_only.json || echo "old-pool simulation failed"
  return $rc
}
main "$@"; exit $?
