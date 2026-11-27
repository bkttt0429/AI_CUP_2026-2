#!/usr/bin/env bash
# METHOD_NOTES 2.131: four held-out tests on top of "competition players drawn by games" (notes 2.124 / 2.129). Folds 1
# and 2, the x2 data, symmetry + ply masking 0.3, 20,000 steps on the 60,000-step cosine, EMA only, by-games (linear) - the
# recipe of arm ga, which is the control here (its fold vectors: data/seq/fold{1,2}_ngae.npz, complete at ~17:10).
#   va  one competition fold fewer (held-out fold 1: folds 3 4 only; held-out fold 2: folds 3 4 only): ga - va measures what
#       one more fold of competition players is worth under the present recipe (the question behind fold 0)
#   vb  30 % of an episode external (ga: 50 %)
#   vc  70 % of an episode external
#   vd  episodes of 384 players x 8 games (ga: 256 x 8), the same number of steps
# Then the fold vectors and the simulated pools against ga: results/final/v2_sim14bg2.json.
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox S=/work/u2507115/pid_v2/seqnet
  local X2="--ext-ply $X/pilot/ply $X/scale1/ply_a --ext-select $X/pilot/select.csv $X/scale1/select.csv"
  local C="--steps 20000 --horizon 60000 --no-state --sym-aug --mask 0.3 --ema-only --own-by-games linear"
  local specs=() v t
  for v in 1 2; do
    specs+=("$v va $C --ext-share 0.5 --drop-fold $((3 - v))" "$v vb $C --ext-share 0.3" "$v vc $C --ext-share 0.7" "$v vd $C --ext-share 0.5 --p-ep 384")
  done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$X2" "${specs[@]}" || echo "a trainer failed (the networks that were written are still scored)"
  for t in vae vbe vce vde; do
    python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _$t --name _n$t --out $R2/data/seq || echo "embedding of $t failed"
  done
  for v in 1 2; do [ -s $R2/data/seq/fold${v}_ngae.npz ] || echo "the control's vectors of fold $v are missing (job 780230)"; done
  python -I $R2/scripts/pid_v2_sim14.py --names gae=_ngae vae=_nvae vbe=_nvbe vce=_nvce vde=_nvde zme=_nzme --ref gae --folds 1 2 --out $R2/results/final/v2_sim14bg2.json
}
main "$@"; exit $?
