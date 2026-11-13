#!/usr/bin/env bash
# METHOD_NOTES 2.58: board-symmetry augmentation passed the held-out test (fused +0.0295 per network, 1-game questions
# 0.510 -> 0.560). All networks of a job share one copy of the games (scripts/nano4/pid_seq_train_fork.sh).
#   pid_exp_sym.sh base     four official networks, usual size, with the augmentation      -> ckpt_-1_o3Ss<k>[e].pt
#   pid_exp_sym.sh big      four official networks, 512 wide x 10 layers, with it          -> ckpt_-1_o3BSs<k>[e].pt
#   pid_exp_sym.sh stack    held-out test (x2 data, folds 1 and 2) of what stacks on the augmentation:
#                           zs + the bigger network, zr + half of the episodes from one rank band, zg every game its own
#                           symmetry (instead of one per episode), zm + ply masking up to 0.3; then the comparison with yse
# Official networks: net -1 = competition folds 1-4, never fold 0; all 116,420 external accounts; EMA.
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
X=/work/u2507115/ext_fox
S=/work/u2507115/pid_v2/seqnet
C="--steps 20000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug"
OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
X2="--ext-ply $X/pilot/ply $X/scale1/ply_a --ext-select $X/pilot/select.csv $X/scale1/select.csv"
case $1 in
  base) bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF" "-1 o3Ss0 $C --seed 0" "-1 o3Ss1 $C --seed 1" "-1 o3Ss2 $C --seed 2" "-1 o3Ss3 $C --seed 3" ;;
  big)  B="$C --arch 512,10,8,2048"
        bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF" "-1 o3BSs0 $B --seed 0" "-1 o3BSs1 $B --seed 1" "-1 o3BSs2 $B --seed 2" "-1 o3BSs3 $B --seed 3" ;;
  stack)
    specs=()
    for v in 1 2; do
      specs+=("$v zs $C --arch 512,10,8,2048" "$v zr $C --rank-ep band --rank-ep-share 0.5" "$v zg $C --sym-game" "$v zm $C --mask 0.3")
    done
    bash $R2/scripts/nano4/pid_seq_train_fork.sh "$X2" "${specs[@]}"
    for t in zse zre zge zme zs zr zg zm; do
      python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _$t --name _n$t --out $R2/data/seq
    done
    python -I $R2/scripts/pid_v2_sim14.py --names yse=_nyse x2e=_nx2e ybe=_nybe zse=_nzse zre=_nzre zge=_nzge zme=_nzme zs=_nzs zr=_nzr zg=_nzg zm=_nzm --ref yse --folds 1 2 --out $R2/results/final/v2_sim14z.json ;;
esac
