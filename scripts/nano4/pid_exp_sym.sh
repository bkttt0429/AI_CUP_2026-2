#!/usr/bin/env bash
# METHOD_NOTES 2.58: board-symmetry augmentation passed the held-out test (fused +0.0295 per network, 1-game questions
# 0.510 -> 0.560). All networks of a job share one copy of the games (scripts/nano4/pid_seq_train_fork.sh).
#   pid_exp_sym.sh base     four official networks, usual size, with the augmentation      -> ckpt_-1_o3Ss<k>[e].pt
#   pid_exp_sym.sh base8    eight more seeds of the same (o3Ss4-11)
#   pid_exp_sym.sh long     eight official networks trained twice as long (o3SLs12-19; snapshots at 20k and 30k steps)
#   pid_exp_sym.sh longheld the same on the held-out folds, compared with yse on the simulated pools
#   pid_exp_sym.sh band     four official networks with the augmentation and rank-band episodes -> ckpt_-1_o3SRs<k>[e].pt
#   pid_exp_sym.sh big      four official networks, 512 wide x 10 layers, with it          -> ckpt_-1_o3BSs<k>[e].pt
#   pid_exp_sym.sh open     held-out test of hiding the opening in training (zo 30 %, zp 50 %, zq up to ply 160) and of
#                           more dropout (zd), all on top of the augmentation; comparison with yse
#   pid_exp_sym.sh bank     held-out test of cross-batch memory negatives (kb; kg and kt with one symmetry per game, kt only the
#                           2,048 hardest) and of a cosine margin (km); comparison with yse
#   pid_exp_sym.sh window   held-out test of random ply windows (wa half of the games, wb all, wc half + ply masking) and of the
#                           reworked cross-batch memory (kc); comparison with yse
#   pid_exp_sym.sh depth    held-out test of how to spend the disk on external accounts (METHOD_NOTES 2.69): d0 39,683 accounts
#                           x 8 games (episodes 256 x 8), d1 the same accounts x 4 games (512 x 4), d2 79,366 accounts x 4
#                           games (the same number of games as d0), d3 all 116,420 x 4; comparisons d1 / d2 with d0, d3 with d2
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
  base8) specs=(); for k in 4 5 6 7 8 9 10 11; do specs+=("-1 o3Ss$k $C --seed $k"); done  # eight more seeds of the usual size
        bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF" "${specs[@]}" ;;
  long)  # METHOD_NOTES 2.62: with the augmentation the held-out curves still rise at 20,000 steps -> eight official networks
         # trained to 40,000 steps (same schedule), new seeds 12-19, snapshots at 20,000 (= further o3S seeds) and 30,000
    L="--steps 40000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --snap 20000 30000"
    specs=(); for k in 12 13 14 15 16 17 18 19; do specs+=("-1 o3SLs$k $L --seed $k"); done
    bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF" "${specs[@]}" ;;
  longheld)  # the same on the x2 data, folds 1 and 2, and the simulated-pool comparison of the 20k / 30k / 40k states with yse
    L="--steps 40000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --snap 20000 30000"
    bash $R2/scripts/nano4/pid_seq_train_fork.sh "$X2" "1 ysl $L" "2 ysl $L"
    for t in yslk20e yslk30e ysle; do
      python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _$t --name _n$t --out $R2/data/seq
    done
    python -I $R2/scripts/pid_v2_sim14.py --names yse=_nyse yslk20e=_nyslk20e yslk30e=_nyslk30e ysle=_nysle --ref yse --folds 1 2 --out $R2/results/final/v2_sim14l.json ;;
  band) R="$C --rank-ep band --rank-ep-share 0.5"  # four official networks with the augmentation AND half of the episodes from one rank band (ahead of the stack test's verdict)
        bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF" "-1 o3SRs0 $R --seed 0" "-1 o3SRs1 $R --seed 1" "-1 o3SRs2 $R --seed 2" "-1 o3SRs3 $R --seed 3" ;;
  big)  B="$C --arch 512,10,8,2048"
        bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF" "-1 o3BSs0 $B --seed 0" "-1 o3BSs1 $B --seed 1" "-1 o3BSs2 $B --seed 2" "-1 o3BSs3 $B --seed 3" ;;
  open)  # METHOD_NOTES 2.60: the other shortcut. Half of the 1-game score sits in the first 30 plies; hide the opening in training
    specs=()
    for v in 1 2; do
      specs+=("$v zo $C --drop-open 0.3" "$v zp $C --drop-open 0.5" "$v zq $C --drop-open 0.3 --drop-open-max 160" "$v zd $C --drop 0.2")
    done
    bash $R2/scripts/nano4/pid_seq_train_fork.sh "$X2" "${specs[@]}"
    for t in zoe zpe zqe zde zo zp zq zd; do
      python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _$t --name _n$t --out $R2/data/seq
    done
    python -I $R2/scripts/pid_v2_sim14.py --names yse=_nyse x2e=_nx2e zoe=_nzoe zpe=_nzpe zqe=_nzqe zde=_nzde zo=_nzo zp=_nzp zq=_nzq zd=_nzd --ref yse --folds 1 2 --out $R2/results/final/v2_sim14o.json ;;
  bank)  # METHOD_NOTES 2.64: cross-batch memory negatives (and a plain cosine margin), all on top of the augmentation
    specs=()
    for v in 1 2; do
      specs+=("$v kb $C --bank" "$v kg $C --sym-game --bank" "$v kt $C --sym-game --bank --bank-topk 2048" "$v km $C --margin 0.1")
    done
    bash $R2/scripts/nano4/pid_seq_train_fork.sh "$X2" "${specs[@]}"
    for t in kbe kge kte kme kb kg kt km; do
      python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _$t --name _n$t --out $R2/data/seq
    done
    python -I $R2/scripts/pid_v2_sim14.py --names yse=_nyse x2e=_nx2e kbe=_nkbe kge=_nkge kte=_nkte kme=_nkme kb=_nkb kg=_nkg kt=_nkt km=_nkm --ref yse --folds 1 2 --out $R2/results/final/v2_sim14k.json ;;
  window)  # METHOD_NOTES 2.66: random ply windows (the rank session's idea) and the reworked cross-batch memory, on top of the augmentation
    specs=()
    for v in 1 2; do
      specs+=("$v wa $C --win 0.5" "$v wb $C --win 1.0" "$v wc $C --win 0.5 --mask 0.2" "$v kc $C --bank --bank-ema --bank-start 8000 --bank-age 1500 --bank-ramp 2000")
    done
    bash $R2/scripts/nano4/pid_seq_train_fork.sh "$X2" "${specs[@]}"
    for t in wae wbe wce kce wa wb wc kc; do
      python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _$t --name _n$t --out $R2/data/seq
    done
    python -I $R2/scripts/pid_v2_sim14.py --names yse=_nyse x2e=_nx2e wae=_nwae wbe=_nwbe wce=_nwce kce=_nkce wa=_nwa wb=_nwb wc=_nwc kc=_nkc --ref yse --folds 1 2 --out $R2/results/final/v2_sim14w.json ;;
  depth)  # nested account subsets of one fixed order; competition players as always (>= 8 games); only the external side changes
    D4="--m-ep 4 --p-ep 512 --ext-cap 4 --min-games 4"
    specs=()
    for v in 1 2; do
      specs+=("$v d0 $C --ext-accounts 39683" "$v d1 $C --ext-accounts 39683 $D4" "$v d2 $C --ext-accounts 79366 $D4" "$v d3 $C $D4")
    done
    bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF" "${specs[@]}"
    for t in d0e d1e d2e d3e; do
      python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _$t --name _n$t --out $R2/data/seq
    done
    python -I $R2/scripts/pid_v2_sim14.py --names d0e=_nd0e yse=_nyse d1e=_nd1e d2e=_nd2e d3e=_nd3e --ref d0e --folds 1 2 --out $R2/results/final/v2_sim14d.json
    python -I $R2/scripts/pid_v2_sim14.py --names d2e=_nd2e d3e=_nd3e d1e=_nd1e --ref d2e --folds 1 2 --out $R2/results/final/v2_sim14d2.json ;;
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
