#!/usr/bin/env bash
# METHOD_NOTES 2.132: held-out test of DEEPER external accounts (scripts/nano4/ext_fox_deep.sh): do networks that read the
# second supervised feature set alone (type C) gain from accounts with ~70 games instead of ~10, and does drawing the
# external accounts by games then help?  Folds 1 and 2, the full official external data + the deeper selection <name>,
# symmetry + ply masking 0.3, half of an episode external, competition players by games (linear), 20,000 steps on the
# 60,000-step cosine, EMA only.
#   wc  control: the deeper games left out (--no-deep), external accounts uniform
#   wd  + deeper games, external accounts uniform
#   we  + deeper games, external accounts drawn by games (linear)
#   wf  + deeper games, external accounts drawn by the square root of their games
# Then the fold vectors and the simulated pools against wc: results/final/v2_sim14deep_<name>.json.
#   pid_exp_deep1.sh <name of the deeper selection, e.g. deep1>
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox S=/work/u2507115/pid_v2/seqnet P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj name=${1:-deep1}
  [ -s "$X/$name/select.csv" ] && [ -d "$X/$name/ply" ] && [ -d "$O/ext_$name" ] || { echo "the deeper selection $name is not complete"; return 1; }
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d $X/$name/ply --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv $X/$name/select.csv --ext-deep $X/$name/select.csv"
  local SUP="--sup-ply $P/ply_js $O/comp $O/ext $O/ext_$name --sup-stats $P/stats_js.npz"
  local C="--steps 20000 --horizon 60000 --no-state --sym-aug --mask 0.3 --ext-share 0.5 --ema-only --sup-require --sup-name js --sup-dims 0 68 --sup-only --own-by-games linear"
  local specs=() v t
  for v in 1 2; do
    specs+=("$v wc $C --no-deep" "$v wd $C" "$v we $C --ext-by-games linear" "$v wf $C --ext-by-games sqrt")
  done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$OFF $SUP" "${specs[@]}" || echo "a trainer failed (the networks that were written are still scored)"
  for t in wce wde wee wfe; do
    python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _$t --name _n$t --sup-ply $P/ply_js $O/comp --sup-dims 0 68 --sup-only --sup-stats $P/stats_js.npz --out $R2/data/seq || echo "embedding of $t failed"
  done
  python -I $R2/scripts/pid_v2_sim14.py --names wce=_nwce wde=_nwde wee=_nwee wfe=_nwfe --ref wce --folds 1 2 --out $R2/results/final/v2_sim14deep_$name.json
}
main "$@"; exit $?
