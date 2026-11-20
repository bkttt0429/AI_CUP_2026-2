#!/usr/bin/env bash
# METHOD_NOTES 2.242: diagnostic - fusion weights per question size against one common weight vector, on the seen hold-out
# seeds 0-3 of a document (default w11c). One node: the splits of scripts/pid_v2_offsim_par.py go to /dev/shm (about 160 MB
# each), scripts/pid_v11_sizefit.py reads them, they are removed. Nothing here chooses or writes a file.
#   pid_exp_sizefit.sh [<document> [<pseudo-pool vector folder>]]
# From the AI_CUP_2026-1 checkout. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 J=${SLURM_JOB_ID:-local} d=${1:-w11c} ft=${2:-/work/u2507115/pid_v2/vtmp8/ft} G D pool s x=0 e=0 q=() k
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ -s $R2/results/pid_fusion_$d.json ] || { echo "no document $d"; return 1; }
  D=/dev/shm/sizefit_$J
  mkdir -p $D || return 1
  for pool in pub2 priv2; do for s in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=${G[$(( x % ${#G[@]} ))]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_v2_offsim_par.py --fusion $R2/results/pid_fusion_$d.json \
      --ft-dir "$ft" --steps 50 --seeds 0 1 2 3 --workers 6 --only $pool:$s --dump $D --out /dev/null > logs/nano4/sizefit-$J-$pool-$s.log 2>&1 &
    q+=($!); x=$((x + 1))
    if [ $(( x % ${#G[@]} )) -eq 0 ]; then for k in "${q[@]}"; do wait "$k" || e=1; done; q=(); fi
  done; done
  for k in "${q[@]}"; do wait "$k" || e=1; done
  echo "splits: $(ls $D | grep -c '^split_.*\.pkl$') of 8 ($(du -sh $D | cut -f1)); $(date +%T)$([ $e = 0 ] || echo '  FAILED (logs/nano4/sizefit-'$J'-*.log)')"
  [ "$e" = 0 ] || { find $D -delete 2> /dev/null; return 1; }
  python -I $R2/scripts/pid_v11_sizefit.py --fusion $R2/results/pid_fusion_$d.json --dump $D --seeds 0 1 2 3 --out $R2/results/final/v11_sizefit_$d.json 2>&1 | cut -c1-330
  e=${PIPESTATUS[0]}
  find $D -delete 2> /dev/null
  echo "done at $(date +%T), rc $e"
  return $e
}
main "$@"; exit $?
