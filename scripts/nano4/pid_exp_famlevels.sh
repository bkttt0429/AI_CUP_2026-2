#!/usr/bin/env bash
# METHOD_NOTES 2.191: the family-level readings of a new family against the ORIGINAL family it replaces in the final's
# composition (the readings registered in notes 2.177; the twins' vectors no longer exist, so the original is the reference):
#   <tag>_x     the final's composition with the original family replaced     vs v2bgm_d2
#   <tag>_only  the new family alone (x18)                                    vs <tag>_ref0 = the original family alone (x18)
# Pseudo-pools, four hold-out seeds, 50 fine-tune steps (scripts/pid_v2_offsim_par.py: the 16 splits at the same time, then the
# statistics from the stored splits). Every vector must exist already; nothing is fine-tuned and no file is built.
#   pid_exp_famlevels.sh <new family, e.g. o32F> <original family, e.g. o22F> <tag, e.g. v2o32m>
# From the AI_CUP_2026-1 checkout, 8 GPUs (the embedding component only). Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 new=$1 orig=$2 tag=$3 J=${SLURM_JOB_ID:-local} i=0 p s pool pair ref alt sp=() rc=0
  local FINAL="o23F*18+o22F*18+o21F*18+o19F*18+o18F*9+o17F*9+o16F*9+o13F*9+o12F*9+o7F*9" D=/dev/shm/famlev_$J
  local x=${FINAL/${orig}\*18/${new}*18} G
  [ "$x" != "$FINAL" ] || { echo "$orig is not in the final's composition"; return 1; }
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  python -I $R2/data/day_scripts/build_v2h.py ${tag}_x "$x" && python -I $R2/data/day_scripts/build_v2h.py ${tag}_only "${new}*18" \
    && python -I $R2/data/day_scripts/build_v2h.py ${tag}_ref0 "${orig}*18" || return 1
  mkdir -p $D || return 1
  for pair in "v2bgm_d2:${tag}_x" "${tag}_ref0:${tag}_only"; do
    ref=${pair%%:*}; alt=${pair#*:}
    for pool in pub2 priv2; do
      for s in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=${G[$(( i % ${#G[@]} ))]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_v2_offsim_par.py --fusion $R2/results/pid_fusion_$ref.json \
          --alt $R2/results/pid_fusion_$alt.json --steps 50 --seeds 0 1 2 3 --workers 6 --only $pool:$s --dump $D/$alt --out /dev/null > logs/nano4/famlev-$J-$alt-$pool-$s.log 2>&1 &
        sp+=($!); i=$((i + 1))
      done
    done
  done
  for p in "${sp[@]}"; do wait "$p" || { echo "a split failed (logs/nano4/famlev-$J-*.log)"; rc=1; }; done
  echo "splits done at $(date +%T)"
  for pair in "v2bgm_d2:${tag}_x" "${tag}_ref0:${tag}_only"; do
    ref=${pair%%:*}; alt=${pair#*:}
    echo "== pseudo-pools, four seeds: $alt vs $ref"
    python -I $R2/scripts/pid_v2_offsim_par.py --fusion $R2/results/pid_fusion_$ref.json --alt $R2/results/pid_fusion_$alt.json --steps 50 --seeds 0 1 2 3 --from-dumps --dump $D/$alt \
      --out $R2/results/final/v2_offsim_${alt}_vs_${ref}_s0_1_2_3.json > logs/nano4/famlev-$J-$alt-stat.log 2>&1 || rc=1
    grep -E '^(pub2|priv2|both): document weights|^   talt vs t50' logs/nano4/famlev-$J-$alt-stat.log | cut -c1-240
  done
  find $D -delete 2> /dev/null
  return $rc
}
main "$@"; exit $?
