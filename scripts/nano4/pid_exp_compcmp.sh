#!/usr/bin/env bash
# METHOD_NOTES 2.194: compositions built from vectors that exist already, each against one reference document:
#   pseudo-pools of the four hold-out seeds 0-3 (scripts/pid_v2_offsim_par.py: every split of every candidate at the same
#   time, then the statistics from the stored splits) and the old-pool simulation (50 steps, scripts/pid_v2_oldsim.py).
# Nothing is trained or fine-tuned, no vector and no submission file is written.
#   pid_exp_compcmp.sh <reference document> <name>=<composition> [<name>=<composition> ...]
# Results: results/final/v2_offsim_<name>_vs_<reference>_s0_1_2_3.json, results/final/v2_oldsim_cmp_<first name>.json.
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 J=${SLURM_JOB_ID:-local} ref=$1
  shift
  local D=/dev/shm/compcmp_$J G i=0 c n names=() pids=() pool s rc=0
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ -s $R2/results/pid_fusion_$ref.json ] || { echo "no document $ref"; return 1; }
  for c in "$@"; do
    n=${c%%=*}
    python -I $R2/data/day_scripts/build_v2h.py "$n" "${c#*=}" || return 1
    names+=("$n")
  done
  for n in "${names[@]}"; do
    mkdir -p $D/$n || return 1
    for pool in pub2 priv2; do for s in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=${G[$(( i % ${#G[@]} ))]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_v2_offsim_par.py --fusion $R2/results/pid_fusion_$ref.json \
        --alt $R2/results/pid_fusion_$n.json --steps 50 --seeds 0 1 2 3 --workers 6 --only $pool:$s --dump $D/$n --out /dev/null > logs/nano4/compcmp-$J-$n-$pool-$s.log 2>&1 &
      pids+=($!); i=$((i + 1))
    done; done
  done
  echo "${#pids[@]} splits started at $(date +%T)"
  for i in "${pids[@]}"; do wait "$i" || { echo "a split failed (logs/nano4/compcmp-$J-*.log)"; rc=1; }; done
  echo "splits done at $(date +%T)"
  for n in "${names[@]}"; do
    echo "== pseudo-pools, four seeds: $n vs $ref"
    python -I $R2/scripts/pid_v2_offsim_par.py --fusion $R2/results/pid_fusion_$ref.json --alt $R2/results/pid_fusion_$n.json --steps 50 --seeds 0 1 2 3 --from-dumps --dump $D/$n \
      --out $R2/results/final/v2_offsim_${n}_vs_${ref}_s0_1_2_3.json > logs/nano4/compcmp-$J-$n-stat.log 2>&1 || rc=1
    grep -E '^(pub2|priv2|both): document weights|^   talt vs t50' logs/nano4/compcmp-$J-$n-stat.log | cut -c1-240
  done
  find $D -delete 2> /dev/null
  echo "== old-pool simulation, 50 steps ($(date +%T))"
  python -I $R2/scripts/pid_v2_oldsim.py --docs "$ref" "${names[@]}" --step 50 --out $R2/results/final/v2_oldsim_cmp_${names[0]}.json || { echo "old-pool simulation failed"; rc=1; }
  echo "done at $(date +%T), rc $rc"
  return $rc
}
main "$@"; exit $?
