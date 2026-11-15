#!/usr/bin/env bash
# METHOD_NOTES 2.125: the o16 recipe (second supervised feature set "js", 30,000 steps on the 60,000-step cosine, a
# snapshot at 20,000) with the competition players of an episode drawn BY GAMES (pid_seq_train.py --own-by-games; the rank
# session's M161 idea, held-out test 779827 of notes 2.124). Started before that test's verdict because 34 nodes were idle
# (precedent: o16, notes 2.118); the reading at step 10,000 of the test was positive for unseen players in both folds. If
# the verdict fails the job is cancelled or the networks are judged by the composition rule like any other family.
# Symmetry + ply masking 0.3, EMA only, competition folds 1-4 (never fold 0), 116,420 external accounts, half of an
# episode external (drawn uniformly, as before).
#   o19As0-2  type A (the pooled half and four more input dimensions replaced by the 68 features), probability ~ games
#   o19As3-5  type A, probability ~ sqrt(games)
#   o19Cs0    type C (the 68 features alone), ~ games
#   o19Cs1    type C, ~ sqrt(games)
# No chain inside. From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj specs=() k
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local SUP="--sup-ply $P/ply_js $O/comp $O/ext --sup-stats $P/stats_js.npz"
  local C="--steps 30000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --snap 20000 --sup-name js --sup-dims 0 68"
  for k in 0 1 2; do specs+=("-1 o19As$k $C --own-by-games linear --seed $((k + 140))"); done
  for k in 3 4 5; do specs+=("-1 o19As$k $C --own-by-games sqrt --seed $((k + 140))"); done
  specs+=("-1 o19Cs0 $C --sup-only --own-by-games linear --seed 150")
  specs+=("-1 o19Cs1 $C --sup-only --own-by-games sqrt --seed 151")
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF $SUP" "${specs[@]}"
}
main "$@"; exit $?
