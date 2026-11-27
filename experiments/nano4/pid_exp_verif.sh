#!/usr/bin/env bash
# METHOD_NOTES 2.209: feasibility probe of the research agent's frozen conditional behaviour policy as a verifier of the
# fusion's first choices. One pseudo-pool split (official candidates, 7 reference games, 3 held-out games as 1-game questions):
#   1. scripts/pid_verif_scores.py, 8 shards (one per GPU; every shard builds the memory of all 600 candidates itself, then
#      scores its query games) -> /dev/shm/verif_<job>/scores_<i>.npz on the node;
#      next to them the split of the reference document v2fin_x (scripts/pid_v2_offsim_par.py --only, CPU) -> .../dump
#   2. scripts/pid_verif_eval.py: D0 / D1 / D2 -> results/final/v2_verif_<pool>_s<seed>.json
#   pid_exp_verif.sh <pub2|priv2> <hold-out seed> [<limit>]
# With <limit>: a smoke test on the first <limit> query games (result file ..._smoke.json; the memory is still the full one).
# Nothing is trained, no vector and no submission file is written; the /dev/shm folder is removed at the end.
# From the AI_CUP_2026-1 checkout, 8 GPUs, -c 96 (every shard starts VERIF_WORKERS = 7 replay processes). Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 pool=${1:-} seed=${2:-} lim=${3:-} J=${SLURM_JOB_ID:-local} ref=v2fin_x
  local D G i p sp out sfx="" rc=0 extra=() pids=() t0=$SECONDS
  case "$pool" in pub2|priv2) ;; *) echo "usage: pid_exp_verif.sh <pub2|priv2> <hold-out seed> [<limit>]"; return 2 ;; esac
  [[ "$seed" =~ ^[0-9]+$ && "$lim" =~ ^[0-9]*$ ]] || { echo "usage: pid_exp_verif.sh <pub2|priv2> <hold-out seed> [<limit>]"; return 2; }
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "run inside a Slurm allocation"; return 1; }
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ "${#G[@]}" -eq 8 ] || { echo "expected eight GPUs"; return 1; }
  [ -n "$lim" ] && { sfx=_smoke; extra=(--limit-queries "$lim"); }
  D=/dev/shm/verif_$J
  out=$R2/results/final/v2_verif_${pool}_s${seed}${sfx}.json
  [ -s $R2/results/pid_fusion_$ref.json ] || { echo "no document $ref"; return 1; }
  python -I -m py_compile $R2/scripts/pid_verif_scores.py $R2/scripts/pid_verif_eval.py || return 1
  mkdir -p $D/dump || return 1
  echo "verifier probe: $pool, hold-out seed $seed${lim:+, smoke test on $lim query games}; reference $ref; node $(hostname); $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=${G[0]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_v2_offsim_par.py --fusion $R2/results/pid_fusion_$ref.json \
    --ft-dir $R2/data/pid_v2_seq_ft_offsim --steps 50 --seeds "$seed" --workers 6 --only "$pool:$seed" --dump $D/dump --out /dev/null < /dev/null > logs/nano4/verif-$J-split.log 2>&1 &
  sp=$!
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=${G[$i]} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -I $R2/scripts/pid_verif_scores.py --pool "$pool" --holdout-seed "$seed" \
      --shard $i --nshards 8 ${extra[@]+"${extra[@]}"} --workers "${VERIF_WORKERS:-7}" --out $D/scores_$i.npz < /dev/null > logs/nano4/verif-$J-$i.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "shards done at $(date +%T) ($(( (SECONDS - t0) / 60 )) min): $(ls $D | grep -c '^scores_[0-7]\.npz$') of 8 files"
  grep -h '^memory:' logs/nano4/verif-$J-0.log | cut -c1-200
  [ $rc = 0 ] || { echo "a shard failed (logs/nano4/verif-$J-[0-7].log):"; tail -q -n 2 logs/nano4/verif-$J-[0-7].log | cut -c1-240; }
  wait "$sp" || { echo "the split of the reference document failed (logs/nano4/verif-$J-split.log):"; tail -n 3 logs/nano4/verif-$J-split.log | cut -c1-240; rc=1; }
  if [ $rc = 0 ]; then
    python -I $R2/scripts/pid_verif_eval.py --pool "$pool" --holdout-seed "$seed" --scores $D/scores_[0-7].npz --fusion $R2/results/pid_fusion_$ref.json \
      --ft-dir $R2/data/pid_v2_seq_ft_offsim --step 50 --dump $D/dump --out "$out" < /dev/null > logs/nano4/verif-$J-eval.log 2>&1 || rc=1
    grep -E '^D[0-2] ' logs/nano4/verif-$J-eval.log | cut -c1-400
    if [ $rc = 0 ]; then echo "result: $out"; else echo "the evaluation failed (logs/nano4/verif-$J-eval.log):"; tail -n 5 logs/nano4/verif-$J-eval.log | cut -c1-300; fi
  fi
  find $D -delete 2> /dev/null
  echo "done at $(date +%T) ($(( (SECONDS - t0) / 60 )) min)"
  return $rc
}
main "$@"; exit $?
