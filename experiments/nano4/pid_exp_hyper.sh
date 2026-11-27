#!/usr/bin/env bash
# METHOD_NOTES 2.91: held-out test of four settings never varied so far, each on top of the symmetry augmentation + ply masking
# 0.3 (folds 1 and 2, the x2 data; against zme, the held-out networks of that recipe):
#   ha  30 % of an episode's players external (now 50 %)        hb  70 % external
#   hc  episodes of 384 players x 6 games (now 256 x 8)         hd  noise on the per-ply vectors, 0.25 standard deviations
# Result results/final/v2_sim14h.json. The body is a function called on the last line with exit.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox S=/work/u2507115/pid_v2/seqnet
  local X2="--ext-ply $X/pilot/ply $X/scale1/ply_a --ext-select $X/pilot/select.csv $X/scale1/select.csv"
  local C="--steps 20000 --horizon 60000 --no-state --sym-aug --mask 0.3"
  local specs=() v t
  for v in 1 2; do
    specs+=("$v ha $C --ext-share 0.3" "$v hb $C --ext-share 0.7" "$v hc $C --ext-share 0.5 --p-ep 384 --m-ep 6" "$v hd $C --ext-share 0.5 --jitter 0.25")
  done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$X2" "${specs[@]}"
  for t in hae hbe hce hde; do
    python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _$t --name _n$t --out $R2/data/seq
  done
  python -I $R2/scripts/pid_v2_sim14.py --names zme=_nzme yse=_nyse hae=_nhae hbe=_nhbe hce=_nhce hde=_nhde --ref zme --folds 1 2 --out $R2/results/final/v2_sim14h.json
}
main "$@"; exit $?
