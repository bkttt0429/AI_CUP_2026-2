#!/usr/bin/env bash
# METHOD_NOTES 2.124: held-out test of drawing the competition players of an episode BY GAMES (the rank session's M161:
# +0.07 held-out top-1 for players the network has seen). Here the question is the unseen player: folds 1 and 2, the x2
# data (competition folds other than the held-out one + 39,683 external accounts, half of an episode external and drawn
# uniformly as before), symmetry augmentation + ply masking 0.3, 20,000 steps on the 60,000-step cosine - the recipe of the
# control zme, whose fold vectors exist (data/seq/fold{1,2}_nzme.npz).
#   ga  competition players drawn with probability proportional to their number of games
#   gb  proportional to the square root of it
# Then the fold vectors and the simulated pools (scripts/pid_v2_sim14.py) against zme: results/final/v2_sim14g.json.
# From the AI_CUP_2026-1 checkout, 4 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox S=/work/u2507115/pid_v2/seqnet
  local X2="--ext-ply $X/pilot/ply $X/scale1/ply_a --ext-select $X/pilot/select.csv $X/scale1/select.csv"
  local C="--steps 20000 --horizon 60000 --no-state --sym-aug --mask 0.3 --ext-share 0.5 --ema-only"
  local specs=() v t
  for v in 1 2; do specs+=("$v ga $C --own-by-games linear" "$v gb $C --own-by-games sqrt"); done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$X2" "${specs[@]}" || { echo "training failed"; return 1; }
  for t in gae gbe; do
    python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _$t --name _n$t --out $R2/data/seq || echo "embedding of $t failed"
  done
  python -I $R2/scripts/pid_v2_sim14.py --names zme=_nzme yse=_nyse gae=_ngae gbe=_ngbe --ref zme --folds 1 2 --out $R2/results/final/v2_sim14g.json
}
main "$@"; exit $?
