#!/usr/bin/env bash
# METHOD_NOTES 2.170: the chain of the window family, in one job (runs without the conversation).
#   pid_exp_o31_chain.sh h    after the first training group: o31H = o31Ss0-3 against o22H = o22Ss0-3 (an alias of the first four
#                             networks of o22F: their vectors are linked under the new name, nothing is recomputed)
#                               v2o31h_only = o31H alone      v2o31h_ref1 = o22H alone     (level 2 of the notes)
#                               v2o31h_x    = the final's composition with o22F*18 -> o22F*9 + o31H*9   (level 3)
#   pid_exp_o31_chain.sh m    after both groups: o31F = the eight networks
#                               v2o31m_x8 = o22F replaced by o31F     v2o31m_a8 = o22F*9 + o31F*9
# Against the final's document v2bgm_d2. Every file a chain writes into upload/ is moved to results/final/not_for_upload/.
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 S=/work/u2507115/pid_v2/seqnet mode=$1 J=${SLURM_JOB_ID:-local} i n f d links="" rc free
  local FINAL="o23F*18+o22F*18+o21F*18+o19F*18+o18F*9+o17F*9+o16F*9+o13F*9+o12F*9+o7F*9"
  free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')
  [ "$free" -ge 3000 ] || { echo "/work has only $free MB free: not started"; return 1; }
  mkdir -p $R2/results/final/not_for_upload
  case "$mode" in
    h)
      for i in 0 1 2 3; do links+=" o31H:$i=ckpt_-1_o31Ss${i}e.pt"; done
      n=0; for i in $links; do [ -s "$S/${i#*=}" ] && n=$((n + 1)); done
      [ "$n" -eq 4 ] || { echo "only $n of the four checkpoints of the first group exist: not started"; return 1; }
      n=0
      for d in $R2/data/seq_ft $R2/data/pid_v2_seq_ft_offsim $R2/data/pid_v2_seq_ft_oldsim; do
        for f in "$d"/*_net-1_o22Fs[0-3]e_t*.npz; do
          [ -e "$f" ] || continue
          ln -sfn "$(basename "$f")" "$d/$(basename "$f" | sed 's/_net-1_o22Fs/_net-1_o22Hs/')"; n=$((n + 1))
        done
      done
      echo "alias o22H: $n vector files of o22F seeds 0-3 linked"
      local x=${FINAL/o22F\*18/o22F*9+o31H*9} tag=v2o31h
      LINKS="$links" REF=v2bgm_d2 bash $R2/scripts/nano4/pid_chain_job4.sh combo $tag o31H "${tag}_x=$x" "${tag}_only=o31H*18" "${tag}_ref1=o22H*18"
      rc=$?
      for f in $R2/upload/pid_${tag}_*.csv; do [ -e "$f" ] && mv -f "$f" $R2/results/final/not_for_upload/; done
      for f in ${tag}_ref1 ${tag}_only; do [ -s $R2/results/pid_fusion_$f.json ] || { echo "no document $f: the one-family comparison is skipped"; return 1; }; done
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -I $R2/scripts/pid_v2_offsim.py --fusion $R2/results/pid_fusion_${tag}_ref1.json --alt $R2/results/pid_fusion_${tag}_only.json --steps 50 --seeds 0 1 2 3 --workers 8 \
        --out $R2/results/final/v2_offsim_${tag}_only_vs_${tag}_ref1_s0_1_2_3.json > logs/nano4/o31cmp-$J-offsim.log 2>&1 || echo "pseudo-pool comparison failed"
      echo "== pseudo-pools, four seeds: o31H alone (window input) vs o22H alone (the same four seeds without it)"
      grep -E '^(pub2|priv2|both): document weights|^   talt vs t50' logs/nano4/o31cmp-$J-offsim.log | cut -c1-240
      echo "== old-pool simulation, 50 steps"
      python -I $R2/scripts/pid_v2_oldsim.py --docs ${tag}_ref1 ${tag}_only --step 50 --out $R2/results/final/v2_oldsim_${tag}_only.json || echo "old-pool simulation failed"
      return $rc ;;
    m)
      for i in 0 1 2 3; do links+=" o31F:$i=ckpt_-1_o31Ss${i}e.pt o31F:$((i + 4))=ckpt_-1_o31SMs${i}e.pt"; done
      n=0; for i in $links; do [ -s "$S/${i#*=}" ] && n=$((n + 1)); done
      [ "$n" -ge 7 ] || { echo "only $n of the eight checkpoints exist: not started"; return 1; }
      local tag=v2o31m
      LINKS="$links" REF=v2bgm_d2 bash $R2/scripts/nano4/pid_chain_job4.sh combo $tag o31F "${tag}_x8=${FINAL/o22F\*18/o31F*18}" "${tag}_a8=${FINAL/o22F\*18/o22F*9+o31F*9}" "${tag}_only=o31F*18" "${tag}_ref1=o22F*18"
      rc=$?
      for f in $R2/upload/pid_${tag}_*.csv; do [ -e "$f" ] && mv -f "$f" $R2/results/final/not_for_upload/; done
      return $rc ;;
    *) echo "usage: pid_exp_o31_chain.sh h | m"; return 2 ;;
  esac
}
main "$@"; exit $?
