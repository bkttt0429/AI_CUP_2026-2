#!/usr/bin/env bash
# METHOD_NOTES 2.138: a second batch of by-games networks on the second supervised feature set ("js"), each pair with one
# more way of hiding information that passed its held-out test on its own (hidden opening, ply windows, dropout 0.2;
# notes 2.79 / 2.83 / 2.94 - stacking them did not help a single network, they are here as members that err differently,
# as the diversity batch o9D was). The o19 recipe otherwise: 30,000 steps on the 60,000-step cosine, symmetry + ply
# masking 0.3, half of an episode external, competition players by games (linear), EMA only, no snapshot.
# Competition folds 1-4 (never fold 0), 116,420 external accounts.
#   o25As0-1  type A + opening hidden in 30 % of the games
#   o25As2-3  type A + half of the games shown through a window of 64 / 128 plies
#   o25As4-5  type A + dropout 0.2
#   o25Cs0-1  type C (the 68 features alone) + opening hidden in 30 % of the games
# No chain inside. From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj specs=() k
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local SUP="--sup-ply $P/ply_js $O/comp $O/ext --sup-stats $P/stats_js.npz"
  local C="--steps 30000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --sup-name js --sup-dims 0 68 --own-by-games linear"
  for k in 0 1; do specs+=("-1 o25As$k $C --drop-open 0.3 --seed $((k + 240))"); done
  for k in 2 3; do specs+=("-1 o25As$k $C --win 0.5 --seed $((k + 240))"); done
  for k in 4 5; do specs+=("-1 o25As$k $C --drop 0.2 --seed $((k + 240))"); done
  for k in 0 1; do specs+=("-1 o25Cs$k $C --sup-only --drop-open 0.3 --seed $((k + 250))"); done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF $SUP" "${specs[@]}"
}
main "$@"; exit $?
