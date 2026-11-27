#!/usr/bin/env bash
# METHOD_NOTES 2.140: held-out test of LOOK-ALIKE episodes (pid_seq_train.py --nn-ep). The remaining errors are near misses
# against the same few similar candidates, shared by every network group (notes 2.139); in a random episode of 256 players
# the most similar people are almost never present, so the network is not asked to tell them apart. Here part of the steps
# take their 128 competition players as clusters: seeds drawn by games as usual, each with its nearest competition players
# by the prototypes of their last visit. The external half of the episode stays random (an external account may be the same
# person as another one; competition ids are distinct people).
# Folds 1 and 2, the x2 data, symmetry + ply masking 0.3, half of an episode external, competition players by games
# (linear), 20,000 steps on the 60,000-step cosine, EMA only - the recipe of arm ga, the control (data/seq/fold{1,2}_ngae.npz).
#   na  half of the steps, clusters of 8          nb  every step, clusters of 8
#   nc  half of the steps, clusters of 4          nd  half of the steps, clusters of 16
# Then the fold vectors and the simulated pools against ga: results/final/v2_sim14nn.json.
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox S=/work/u2507115/pid_v2/seqnet
  local X2="--ext-ply $X/pilot/ply $X/scale1/ply_a --ext-select $X/pilot/select.csv $X/scale1/select.csv"
  local C="--steps 20000 --horizon 60000 --no-state --sym-aug --mask 0.3 --ext-share 0.5 --ema-only --own-by-games linear"
  local specs=() v t
  python -I -m py_compile $R2/scripts/pid_seq_train.py || { echo "the trainer does not compile"; return 1; }
  for v in 1 2; do
    specs+=("$v na $C --nn-ep 0.5 --nn-k 8" "$v nb $C --nn-ep 1.0 --nn-k 8" "$v nc $C --nn-ep 0.5 --nn-k 4" "$v nd $C --nn-ep 0.5 --nn-k 16")
  done
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "$X2" "${specs[@]}" || echo "a trainer failed (the networks that were written are still scored)"
  for t in nae nbe nce nde; do
    python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix _$t --name _n$t --out $R2/data/seq || echo "embedding of $t failed"
  done
  python -I $R2/scripts/pid_v2_sim14.py --names gae=_ngae nae=_nnae nbe=_nnbe nce=_nnce nde=_nnde zme=_nzme --ref gae --folds 1 2 --out $R2/results/final/v2_sim14nn.json
}
main "$@"; exit $?
