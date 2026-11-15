#!/usr/bin/env bash
# METHOD_NOTES 2.138: plain-input networks of the BIGGER size (512 x 10 layers, FFN 2048; o3BS was this size with uniform
# sampling) with the competition players drawn BY GAMES (linear). By-games lifted the plain-input networks most (o22F
# against o7F: +0.04 on the official candidates' 1-game questions, notes 2.137), and members of another size added to the
# composition before (o3BS, o17). 30,000 steps on the 60,000-step cosine, symmetry + ply masking 0.3, half of an episode
# external, EMA only, no snapshot. Competition folds 1-4 (never fold 0), 116,420 external accounts.
#   o24Ss0-3   symmetry + ply masking 0.3
#   o24SMs0-3  + opening hidden in 30 % of the games
# No chain inside. From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox specs=() k
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local C="--steps 30000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --ema-only --own-by-games linear --arch 512,10,8,2048"
  for k in 0 1 2 3; do specs+=("-1 o24Ss$k $C --seed $((k + 220))"); done
  for k in 0 1 2 3; do specs+=("-1 o24SMs$k $C --drop-open 0.3 --seed $((k + 224))"); done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF" "${specs[@]}"
}
main "$@"; exit $?
