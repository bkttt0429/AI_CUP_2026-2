#!/usr/bin/env bash
# METHOD_NOTES 2.55, item 1: rank-matched episodes (hard negatives). Eight held-out networks on one node (folds 1 and 2 x four
# ways of drawing the episodes; data as x2: competition folds without the held-out one and without fold 0, 39,683 external
# accounts in half of every episode), then their fold vectors and the simulated-pool comparison against x2e.
#   hc  every episode from one rank class          hb  every episode from a class and its two neighbours
#   hm  half of the episodes as hc, half as before hn  half of the episodes as hb, half as before
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
X=/work/u2507115/ext_fox
S=/work/u2507115/pid_v2/seqnet
COMMON="--steps 20000 --horizon 60000 --ext-ply $X/pilot/ply $X/scale1/ply_a --ext-select $X/pilot/select.csv $X/scale1/select.csv --ext-share 0.5 --no-state"
bash $R2/scripts/nano4/pid_seq_train_multi.sh \
  "1 hc $COMMON --rank-ep class" "2 hc $COMMON --rank-ep class" \
  "1 hb $COMMON --rank-ep band" "2 hb $COMMON --rank-ep band" \
  "1 hm $COMMON --rank-ep class --rank-ep-share 0.5" "2 hm $COMMON --rank-ep class --rank-ep-share 0.5" \
  "1 hn $COMMON --rank-ep band --rank-ep-share 0.5" "2 hn $COMMON --rank-ep band --rank-ep-share 0.5"
for t in hce hbe hme hne hc hb hm hn; do
  python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _$t --name _n$t --out $R2/data/seq
done
python -I $R2/scripts/pid_v2_sim14.py --names x2e=_nx2e x2=_nx2 hce=_nhce hbe=_nhbe hme=_nhme hne=_nhne hc=_nhc hb=_nhb hm=_nhm hn=_nhn --ref x2e --folds 1 2 --out $R2/results/final/v2_sim14h.json
