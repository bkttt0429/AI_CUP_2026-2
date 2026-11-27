#!/usr/bin/env bash
# METHOD_NOTES 2.121: supervised-input networks of a DIFFERENT SIZE. What added to the composition so far were members that
# are wrong in a different way: the bigger plain-input networks (o3BS: +0.002 with 4 networks) and the longer-trained
# supervised-input ones (o13F: +0.002). Here both: the second supervised feature set ("js", the only one whose training store
# still exists) read by the bigger transformer (512 x 10 layers, FFN 2048; the standard one is 384 x 8), 30,000 steps on the
# 60,000-step cosine with a snapshot at 20,000, symmetry + ply masking 0.3, half of an episode external, EMA only.
# Competition folds 1-4 (never fold 0), 116,420 external accounts.
#   o17As0-5  the pooled half and four more input dimensions replaced by the 68 features (type A)
#   o17Cs0-1  the 68 features alone (type C)
# No chain inside (the reference composition may change before it ends).
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj specs=() k
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local SUP="--sup-ply $P/ply_js $O/comp $O/ext --sup-stats $P/stats_js.npz"
  local C="--steps 30000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --snap 20000 --sup-name js --sup-dims 0 68 --arch 512,10,8,2048"
  for k in 0 1 2 3 4 5; do specs+=("-1 o17As$k $C --seed $((k + 100))"); done
  for k in 0 1; do specs+=("-1 o17Cs$k $C --sup-only --seed $((k + 110))"); done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF $SUP" "${specs[@]}"
}
main "$@"; exit $?
