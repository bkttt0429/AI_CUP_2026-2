#!/usr/bin/env bash
# METHOD_NOTES 2.133: the o18 recipe (second supervised feature set "js", 40,000 steps on the 60,000-step cosine) with the
# competition players of an episode drawn BY GAMES (linear) - the longer-trained counterpart of o19 (30,000 steps), as
# o13F is of o12F and o18 of o16. By-games holds at the full official scale (notes 2.132: +0.0277 fused against the
# uniform control, about +0.017 of it from the sampling); the depth test left the recipe as it is (256 x 8, no second
# sides, external accounts uniform). No snapshot.
# Symmetry + ply masking 0.3, EMA only, competition folds 1-4 (never fold 0), 116,420 external accounts, half of an
# episode external.
#   o23As0-5  type A (the pooled half and four more input dimensions replaced by the 68 features)
#   o23Cs0-1  type C (the 68 features alone)
# No chain inside. From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj specs=() k
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local SUP="--sup-ply $P/ply_js $O/comp $O/ext --sup-stats $P/stats_js.npz"
  local C="--steps 40000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --sup-name js --sup-dims 0 68 --own-by-games linear"
  for k in 0 1 2 3 4 5; do specs+=("-1 o23As$k $C --seed $((k + 200))"); done
  for k in 0 1; do specs+=("-1 o23Cs$k $C --sup-only --seed $((k + 210))"); done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF $SUP" "${specs[@]}"
}
main "$@"; exit $?
