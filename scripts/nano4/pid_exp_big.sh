#!/usr/bin/env bash
# METHOD_NOTES 2.57: the bigger network (512 wide, 10 layers) passed the held-out test (+0.0098 fused, 1-game questions +0.026).
#   pid_exp_big.sh official   eight official networks of that size (net -1: competition folds 1-4, never fold 0; all 116,420
#                             external accounts incl. the high-dan batch; EMA), seeds 0-7 -> ckpt_-1_o3Bs<k>[e].pt
#   pid_exp_big.sh arch       held-out test of two more shapes on the x2 data (folds 1 and 2): yw = 640 wide x 8 layers,
#                             yd = 448 wide x 12 layers; then fold vectors and the comparison with x2e and ybe
# All networks of a job share one copy of the games (scripts/nano4/pid_seq_train_fork.sh).
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
X=/work/u2507115/ext_fox
S=/work/u2507115/pid_v2/seqnet
C="--steps 20000 --horizon 60000 --ext-share 0.5 --no-state"
case $1 in
  official)
    DATA="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
    specs=()
    for k in 0 1 2 3 4 5 6 7; do specs+=("-1 o3Bs$k $C --arch 512,10,8,2048 --seed $k"); done
    bash $R2/scripts/nano4/pid_seq_train_fork.sh "$DATA" "${specs[@]}" ;;
  arch)
    DATA="--ext-ply $X/pilot/ply $X/scale1/ply_a --ext-select $X/pilot/select.csv $X/scale1/select.csv"
    bash $R2/scripts/nano4/pid_seq_train_fork.sh "$DATA" "1 yw $C --arch 640,8,8,2560" "2 yw $C --arch 640,8,8,2560" "1 yd $C --arch 448,12,8,1792" "2 yd $C --arch 448,12,8,1792"
    for t in ywe yde yw yd; do
      python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _$t --name _n$t --out $R2/data/seq
    done
    python -I $R2/scripts/pid_v2_sim14.py --names x2e=_nx2e ybe=_nybe ywe=_nywe yde=_nyde yw=_nyw yd=_nyd --ref x2e --folds 1 2 --out $R2/results/final/v2_sim14arch.json ;;
esac
