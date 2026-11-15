#!/usr/bin/env bash
# METHOD_NOTES 2.129: the o7 recipe (plain inputs - the randomly projected trunk vectors - 40,000 steps on the 60,000-step
# cosine, symmetry + ply masking 0.3, half of them with the opening hidden in 30 % of the games) with the competition
# players of an episode drawn BY GAMES (linear). The by-games held-out test (notes 2.124) was run on exactly this kind of
# network; its partial reading (two of four networks) is fused +0.0148 / +0.0120 with both intervals above 0. o7F carries
# weight 9 in every composition since 11-14: this is its by-games counterpart.
# The registered four-pool verdict (~17:10) decides as in notes 2.125 (both arms' fused point estimates <= 0 -> cancelled).
# EMA only, no snapshots (the /work quota is nearly full), competition folds 1-4 (never fold 0), 116,420 external accounts
# (o7 had 182,167: the third batch's features were deleted on 11-14; accounts beyond 116k added +0.002 per network).
#   o22Ss0-3   symmetry + ply masking 0.3
#   o22SMs0-3  + opening hidden in 30 % of the games
# No chain inside. From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox specs=() k
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local C="--steps 40000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --ema-only --own-by-games linear"
  for k in 0 1 2 3; do specs+=("-1 o22Ss$k $C --seed $((k + 180))"); done
  for k in 0 1 2 3; do specs+=("-1 o22SMs$k $C --drop-open 0.3 --seed $((k + 184))"); done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF" "${specs[@]}"
}
main "$@"; exit $?
