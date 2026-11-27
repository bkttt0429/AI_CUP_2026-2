#!/usr/bin/env bash
# METHOD_NOTES 2.56: the training objective of face / speaker recognition. Eight held-out networks (folds 1 and 2 x four
# objectives; data as x2), started in two waves of four, then their fold vectors and the simulated-pool comparison with x2e.
#   ca  classification over all training players with a cosine margin 0.2 (no episode loss)
#   cb  episode loss + that classification loss (margin 0.2)
#   cc  episode loss + classification with margin 0.1
#   cd  cb with half of the episodes drawn from one rank band (hard negatives inside the episode as well)
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
X=/work/u2507115/ext_fox
S=/work/u2507115/pid_v2/seqnet
COMMON="--steps 20000 --horizon 60000 --ext-ply $X/pilot/ply $X/scale1/ply_a --ext-select $X/pilot/select.csv $X/scale1/select.csv --ext-share 0.5 --no-state"
STAGGER=4 bash $R2/scripts/nano4/pid_seq_train_multi.sh \
  "1 ca $COMMON --cls-loss cosface --no-episode-loss" "1 cb $COMMON --cls-loss cosface" "1 cc $COMMON --cls-loss cosface --cls-margin 0.1" "1 cd $COMMON --cls-loss cosface --rank-ep band --rank-ep-share 0.5" \
  "2 ca $COMMON --cls-loss cosface --no-episode-loss" "2 cb $COMMON --cls-loss cosface" "2 cc $COMMON --cls-loss cosface --cls-margin 0.1" "2 cd $COMMON --cls-loss cosface --rank-ep band --rank-ep-share 0.5"
for t in cae cbe cce cde ca cb cc cd; do
  python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _$t --name _n$t --out $R2/data/seq
done
python -I $R2/scripts/pid_v2_sim14.py --names x2e=_nx2e x2=_nx2 cae=_ncae cbe=_ncbe cce=_ncce cde=_ncde ca=_nca cb=_ncb cc=_ncc cd=_ncd --ref x2e --folds 1 2 --out $R2/results/final/v2_sim14c.json
