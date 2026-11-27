#!/usr/bin/env bash
# METHOD_NOTES 2.55, item 1: rank-matched episodes (hard negatives), held-out test. Data as x2 (competition folds without the
# held-out one and without fold 0, 39,683 external accounts in half of every episode); four ways of drawing the episodes:
#   hc  every episode from one rank class          hb  every episode from a class and its two neighbours
#   hm  half of the episodes as hc, half as before hn  half of the episodes as hb, half as before
# usage: pid_exp_hardneg.sh 1|2    the four networks that leave out fold 1 (or 2), one job with 4 GPUs
#        pid_exp_hardneg.sh eval   fold vectors of all eight and the simulated-pool comparison against x2e
# Four networks per node: eight on one node spent an hour reading their games (2026-11-14).
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
X=/work/u2507115/ext_fox
S=/work/u2507115/pid_v2/seqnet
COMMON="--steps 20000 --horizon 60000 --ext-ply $X/pilot/ply $X/scale1/ply_a --ext-select $X/pilot/select.csv $X/scale1/select.csv --ext-share 0.5 --no-state"
case $1 in
  1|2)
    v=$1
    bash $R2/scripts/nano4/pid_seq_train_multi.sh "$v hc $COMMON --rank-ep class" "$v hb $COMMON --rank-ep band" \
      "$v hm $COMMON --rank-ep class --rank-ep-share 0.5" "$v hn $COMMON --rank-ep band --rank-ep-share 0.5" ;;
  eval)
    for t in hce hbe hme hne hc hb hm hn; do
      python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _$t --name _n$t --out $R2/data/seq
    done
    python -I $R2/scripts/pid_v2_sim14.py --names x2e=_nx2e x2=_nx2 hce=_nhce hbe=_nhbe hme=_nhme hne=_nhne hc=_nhc hb=_nhb hm=_nhm hn=_nhn --ref x2e --folds 1 2 --out $R2/results/final/v2_sim14h.json ;;
esac
