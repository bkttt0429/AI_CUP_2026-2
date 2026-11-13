#!/usr/bin/env bash
# METHOD_NOTES 2.75 / 2.77: the embfull component trained on every extracted training row (folds 1-4), scored in the whole
# pipeline on the official pseudo-pools; four variants side by side on this job's GPUs (one each):
#   a  full rows (1,536), the first reading's settings          b  full + phase means (3,840)
#   c  as b, 1,024 hidden units, 6,000 steps                    d  as a, at most 16 games of a player (the first reading's amount)
# From the AI_CUP_2026-1 checkout, 4 GPUs.   pid_exp_embfull2.sh
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 D=/work/u2507115/pid_v2 J=${SLURM_JOB_ID:-local}
  local T="--train-rows $D/fullrows_train/rows --pool-rows $D/fullrows_off2"
  IFS=, read -r -a G <<< "$CUDA_VISIBLE_DEVICES"
  local i=0 name opts
  while read -r name opts; do
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${G[$((i % ${#G[@]}))]} OMP_NUM_THREADS=8 python -I $R2/scripts/pid_v2_offsim_embfull.py $T $opts --save $D/embfull_$name.pt \
      --out $R2/results/final/v2_offsim_embfull_$name.json > logs/nano4/embfull2-$J-$name.log 2>&1 &
    i=$((i + 1))
  done <<'ARMS'
a
b --phase
c --phase --hidden 1024 --emb-steps 6000
d --max-games 16
ARMS
  wait
  for name in a b c d; do
    echo "== $name: $(grep -E '^training rows' logs/nano4/embfull2-$J-$name.log | cut -c1-120)"
    grep -E "^both pools, alone|^fused gain|^new component" logs/nano4/embfull2-$J-$name.log | cut -c1-400
  done
}
main "$@"; exit $?
