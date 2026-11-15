#!/usr/bin/env bash
# METHOD_NOTES 2.123: the second supervised feature set ("js") trained as o13 was on the first one - 40,000 steps on the
# 60,000-step cosine with a snapshot at 30,000 (o12F -> o13F: single networks unchanged, +0.002 in the composition; js at
# 20k steps (o14) added +0.0009, at 30k (o16) +0.0021). Symmetry + ply masking 0.3, EMA only, competition folds 1-4 (never
# fold 0), 116,420 external accounts; as in o13 two of the type-A networks draw 70% of an episode from the external accounts.
#   o18As0-3  type A (the pooled half and four more input dimensions replaced by the 68 features), half of an episode external
#   o18As4-5  type A, 70% external
#   o18Cs0-1  type C (the 68 features alone)
# No chain inside. From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj specs=() k
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local SUP="--sup-ply $P/ply_js $O/comp $O/ext --sup-stats $P/stats_js.npz"
  local C="--steps 40000 --horizon 60000 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --snap 30000 --sup-name js --sup-dims 0 68"
  for k in 0 1 2 3; do specs+=("-1 o18As$k $C --ext-share 0.5 --seed $((k + 120))"); done
  for k in 4 5; do specs+=("-1 o18As$k $C --ext-share 0.7 --seed $((k + 120))"); done
  for k in 0 1; do specs+=("-1 o18Cs$k $C --ext-share 0.5 --sup-only --seed $((k + 130))"); done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF $SUP" "${specs[@]}"
}
main "$@"; exit $?
