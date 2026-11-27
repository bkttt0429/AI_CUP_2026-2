#!/usr/bin/env bash
# METHOD_NOTES 2.147: the pair network on the OLD-POOL simulation (scripts/pid_pair_old.py) - a second, independent reading
# of the re-ranking: another population of official players, 10-game reference sets, questions of 1 / 2 / 3 / 5 games.
# Nothing is chosen here; the settings come from the pseudo-pools (notes 2.144).
#   pid_exp_pair_old.sh <output tag> "<network tags, e.g. pa pb pc>" ["<form beta ...>", default "z 0.5 z 1 raw 0.5"] [label=net,net ...]
#   pid_exp_pair_old.sh build ""            only the tasks (CPU)
# Pair scores already written are not computed again. From the AI_CUP_2026-1 checkout. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 W=/work/u2507115/pid_v2 tag=$1 g nets=() names=() fixed=() GPUS=() ALL
  local TASKS=$W/pair_eval/tasks_old_v2bgm_d2_k10.pkl
  read -r -a names <<< "${2:-}"
  read -r -a fixed <<< "${3:-z 0.5 z 1 raw 0.5}"
  shift; shift || true; shift || true
  python -I -m py_compile $R2/scripts/pid_pair_old.py $R2/scripts/pid_pair_eval.py $R2/scripts/pid_pair_train.py || { echo "a script does not compile"; return 1; }
  mkdir -p $W/pair_eval
  if [ ! -s "$TASKS" ]; then
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_pair_old.py --build --doc v2bgm_d2 --tasks "$TASKS" || { echo "the old-pool tasks were not built"; return 1; }
  fi
  for g in "${names[@]}"; do
    if [ -s $W/seqnet/pair_$g.pt ]; then nets+=("$W/seqnet/pair_$g.pt"); else echo "no network pair_$g"; fi
  done
  [ "${#nets[@]}" -ge 1 ] || { echo "tasks only"; return 0; }
  IFS=, read -r -a ALL <<< "${CUDA_VISIBLE_DEVICES}"
  for g in "${ALL[@]}"; do
    if CUDA_VISIBLE_DEVICES="$g" python -I -c "import torch; torch.zeros(1).cuda()" > /dev/null 2>&1; then GPUS+=("$g"); else echo "GPU $g on $(hostname) is not usable, skipped"; fi
  done
  [ "${#GPUS[@]}" -ge 1 ] || { echo "no usable GPU"; return 1; }
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_pair_old.py --tasks "$TASKS" --nets "${nets[@]}" --gpus "${GPUS[@]}" --ft 0 50 150 \
    --fixed "${fixed[@]}" --mean "$@" --out $R2/results/final/v2_pair_old_$tag.json
}
main "$@"; exit $?
