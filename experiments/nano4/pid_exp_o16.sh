#!/usr/bin/env bash
# METHOD_NOTES 2.118: more official networks on the SECOND supervised feature set ("js"), trained longer - started before
# o14's verdict because the GPUs were idle (if o14 adds nothing these are not used).
# Competition folds 1-4 (never fold 0), 116,420 external accounts, symmetry + ply masking 0.3, half of an episode external,
# 30,000 steps on the 60,000-step cosine (the schedule of o12F), a snapshot at 20,000 steps (= o14 with new seeds), EMA only.
#   o16As0-5  the pooled half and four more input dimensions replaced by the 68 features (type A)
#   o16Cs0-1  the 68 features alone (type C)
# No chain inside: it is submitted separately once the reference composition is known.
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj specs=() k
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local SUP="--sup-ply $P/ply_js $O/comp $O/ext --sup-stats $P/stats_js.npz"
  local C="--steps 30000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --snap 20000 --sup-name js --sup-dims 0 68"
  for k in 0 1 2 3 4 5; do specs+=("-1 o16As$k $C --seed $((k + 80))"); done
  for k in 0 1; do specs+=("-1 o16Cs$k $C --sup-only --seed $((k + 90))"); done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF $SUP" "${specs[@]}"
}
main "$@"; exit $?
