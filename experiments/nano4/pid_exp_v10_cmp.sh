#!/usr/bin/env bash
# METHOD_NOTES 2.230: the splits of ONE comparison (alternative vs reference) for the hold-out seeds whose vectors are on
# THIS node, reduced to per-question scores in a shared folder (scripts/pid_v10_sc.py; a split itself is ~160 MB and stays
# in /dev/shm until it is reduced). With STAT="<all seeds>" the pooled statistics follow as soon as the per-split files of
# all those seeds are present (the other node's task writes its own; waits up to STAT_WAIT minutes, default 240).
#   pid_exp_v10_cmp.sh <vector folder | shm:<name>> <reference> <alternative> "<seeds of this node>" [<shared folder>]
#     shm:<name> = /dev/shm/f0c_<name>/ft (written by pid_exp_v10_vec.sh on this node)
#     shared folder: default results/final/v10_sc of the AICUP_2026-2 checkout
# Nothing is written to /work. From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 J=${SLURM_JOB_ID:-local} ft=$1 r=$2 al=$3 sd=$4 sh=${5:-/home/u2507115/BK_AICUP_2026-2/results/final/v10_sc}
  local G D pool s x=0 e=0 q=() k n t0=$SECONDS all miss
  case "$ft" in shm:*) ft=/dev/shm/f0c_${ft#shm:}/ft ;; esac
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ -d "$ft" ] || { echo "no vector folder $ft on $(hostname)"; return 1; }
  for k in $r $al; do [ -s $R2/results/pid_fusion_$k.json ] || { echo "no document $k"; return 1; }; done
  D=/dev/shm/v10dump_$J
  mkdir -p $D "$sh" || return 1
  echo "comparison $al vs $r, seeds $sd, vectors $ft ($(ls "$ft" | wc -l) files) on $(hostname), start $(date +%T)"
  for pool in pub2 priv2; do for s in $sd; do
    (
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=${G[$(( x % ${#G[@]} ))]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_v2_offsim_par.py --fusion $R2/results/pid_fusion_$r.json \
        --alt $R2/results/pid_fusion_$al.json --ft-dir "$ft" --steps 50 --seeds $sd --workers 6 --only $pool:$s --dump $D/${al}_$pool$s --out /dev/null \
      && python -I $R2/scripts/pid_v10_sc.py --fusion $R2/results/pid_fusion_$r.json --alt $R2/results/pid_fusion_$al.json --split $D/${al}_$pool$s/split_${pool}_$s.pkl --out "$sh/sc_${al}_vs_${r}_${pool}_$s.npz"
      rc=$?
      find $D/${al}_$pool$s -delete 2> /dev/null
      exit $rc
    ) > logs/nano4/v10cmp-$J-$al-$pool-$s.log 2>&1 &
    q+=($!); x=$((x + 1))
    if [ $(( x % ${#G[@]} )) -eq 0 ]; then for k in "${q[@]}"; do wait "$k" || e=1; done; q=(); fi
  done; done
  for k in "${q[@]}"; do wait "$k" || e=1; done
  find $D -delete 2> /dev/null
  grep -h -E 'split [0-9]+: [0-9]+ questions, reference' logs/nano4/v10cmp-$J-$al-*.log | sort | cut -c1-200
  echo "splits of seeds $sd done at $(date +%T) ($(( (SECONDS - t0) / 60 )) min)$([ $e = 0 ] || echo '  FAILED (logs/nano4/v10cmp-'$J'-'$al'-*.log)')"
  [ "$e" = 0 ] || return 1
  [ -n "${STAT:-}" ] || return 0
  all=$STAT
  while :; do
    miss=0
    for pool in pub2 priv2; do for s in $all; do [ -s "$sh/sc_${al}_vs_${r}_${pool}_$s.npz" ] || miss=$((miss + 1)); done; done
    [ "$miss" = 0 ] && break
    [ $(( (SECONDS - t0) / 60 )) -lt "${STAT_WAIT:-240}" ] || { echo "statistics: $miss per-split files of seeds $all are still missing after ${STAT_WAIT:-240} min"; return 1; }
    sleep 30
  done
  # shellcheck disable=SC2086
  python -I $R2/scripts/pid_v10_pool.py --dir "$sh" --ref $r --alt $al --seeds $all --out $R2/results/final/v10_pool_${al}_vs_${r}_s${all// /_}.json 2>&1 | cut -c1-260
  return ${PIPESTATUS[0]}
}
main "$@"; exit $?
