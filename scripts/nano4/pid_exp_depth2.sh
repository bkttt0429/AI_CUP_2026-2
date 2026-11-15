#!/usr/bin/env bash
# METHOD_NOTES 2.126: held-out tests around "many games of one person", on top of drawing the competition players by games
# (notes 2.124). Folds 1 and 2, the full official external data (116,420 accounts) with the second sides of the selected
# games loaded (scripts/ext_fox_census.py: a selected game whose other player is a selected account is read once more from
# that side - 10.0 -> 17.9 games per account at no disk cost), symmetry + ply masking 0.3, half of an episode external,
# 20,000 steps on the 60,000-step cosine, EMA only.
#   uc  control: own games only (--no-second), competition players by games (linear), external accounts uniform, 256 x 8
#   ud  + second sides (external accounts uniform)
#   ue  + second sides, external accounts drawn by games as well (linear)
#   uf  control with episodes of 204 players x 10 games (so far only fewer games per player were tried: 512 x 4 and
#       384 x 6 were worse than 256 x 8, notes 2.89 / 2.105)
# Then the fold vectors and the simulated pools against uc: results/final/v2_sim14dep.json.
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox S=/work/u2507115/pid_v2/seqnet
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local C="--steps 20000 --horizon 60000 --no-state --sym-aug --mask 0.3 --ext-share 0.5 --ema-only --own-by-games linear"
  local specs=() v t
  for v in 1 2; do
    specs+=("$v uc $C --no-second" "$v ud $C" "$v ue $C --ext-by-games linear" "$v uf $C --no-second --p-ep 204 --m-ep 10")
  done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF --ext-second $X/census/second.csv" "${specs[@]}" || { echo "training failed"; return 1; }
  for t in uce ude uee ufe; do
    python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _$t --name _n$t --out $R2/data/seq || echo "embedding of $t failed"
  done
  python -I $R2/scripts/pid_v2_sim14.py --names uce=_nuce ude=_nude uee=_nuee ufe=_nufe zme=_nzme --ref uce --folds 1 2 --out $R2/results/final/v2_sim14dep.json
}
main "$@"; exit $?
