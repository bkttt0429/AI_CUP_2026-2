#!/usr/bin/env bash
# METHOD_NOTES 2.129: the o17 recipe (second supervised feature set "js" read by the bigger transformer, 512 x 10 layers,
# FFN 2048; 30,000 steps on the 60,000-step cosine) with the competition players of an episode drawn BY GAMES (linear).
# Started on the partial reading of the by-games test (two of its four networks: fused +0.0148 / +0.0120 against the
# uniform control on the two pools each, both intervals above 0); the registered four-pool verdict follows at ~17:10 and
# decides as written in notes 2.125 (both arms' fused point estimates <= 0 -> cancelled).
# o17 added +0.002 to the composition on top of o16 as "the same features read by a different network"; this is its
# by-games counterpart, as o19 is o16's. No snapshot (the /work quota is nearly full: notes 2.128).
# Symmetry + ply masking 0.3, EMA only, competition folds 1-4 (never fold 0), 116,420 external accounts, half of an
# episode external (drawn uniformly).
#   o21As0-5  type A (the pooled half and four more input dimensions replaced by the 68 features)
#   o21Cs0-1  type C (the 68 features alone)
# No chain inside. From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj specs=() k
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local SUP="--sup-ply $P/ply_js $O/comp $O/ext --sup-stats $P/stats_js.npz"
  local C="--steps 30000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --sup-name js --sup-dims 0 68 --arch 512,10,8,2048 --own-by-games linear"
  for k in 0 1 2 3 4 5; do specs+=("-1 o21As$k $C --seed $((k + 160))"); done
  for k in 0 1; do specs+=("-1 o21Cs$k $C --sup-only --seed $((k + 170))"); done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF $SUP" "${specs[@]}"
}
main "$@"; exit $?
