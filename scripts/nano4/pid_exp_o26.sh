#!/usr/bin/env bash
# METHOD_NOTES 2.157: what is one competition fold worth ON THE OFFICIAL POOLS? The o22 family once more (plain inputs,
# 40,000 steps on the 60,000-step cosine, by-games sampling, the same eight seeds and settings) with ONE FOLD FEWER:
# competition folds 1-3 instead of 1-4 (--drop-fold 4; fold 0 is not touched). o26F against o22F, network by network and
# as a family, on the official pseudo-pools = the loss from 25 % fewer competition players; the gain from adding a fifth
# fold (fold 0, which needs the user's consent) is about 0.78 of it (ln(5/4) / ln(4/3)).
#   o26Ss0-3   symmetry + ply masking 0.3            (o22Ss0-3 with --drop-fold 4)
#   o26SMs0-3  + opening hidden in 30 % of the games  (o22SMs0-3 with --drop-fold 4)
# EMA only, no snapshots. No chain inside. From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox specs=() k d
  for d in $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d; do
    [ "$(ls "$d" 2> /dev/null | grep -c 'ply_x_.*[0-9]\.npz$')" -ge 200 ] || { echo "external ply folder $d is missing or incomplete"; return 1; }
  done
  echo "external parts: $(for d in $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d; do ls $d | grep -c 'ply_x_.*[0-9]\.npz$'; done | tr '\n' ' ') (expected 976 1504 2000 1392 208 248)"
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local C="--steps 40000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --ema-only --own-by-games linear --drop-fold 4"
  for k in 0 1 2 3; do specs+=("-1 o26Ss$k $C --seed $((k + 180))"); done
  for k in 0 1 2 3; do specs+=("-1 o26SMs$k $C --drop-open 0.3 --seed $((k + 184))"); done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF" "${specs[@]}"
}
main "$@"; exit $?
