#!/usr/bin/env bash
# METHOD_NOTES 2.214: the fusion weights of a document re-checked with two factors (emb x e, seq x l): the splits of the
# document for the hold-out seeds 0-5 of both pools (scripts/pid_v2_offsim_par.py --only, vectors from <ft dir>), then
# scripts/pid_v2_wcheck.py (chosen on seeds 0 1, tested on 2 3, then the fresh seeds 4 5).
#   pid_exp_wcheck.sh <document> [<ft dir>]        default ft dir: data/pid_v2_f0tmp/ft (links to every store + its own files)
# Result: results/final/v2_wcheck_<document>.json. The split files live in /dev/shm of the node and are removed.
# From the AI_CUP_2026-1 checkout, 8 GPUs, -c 96. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 doc=$1 ft=${2:-/home/u2507115/BK_AICUP_2026-2/data/pid_v2_f0tmp/ft} J=${SLURM_JOB_ID:-local}
  local D=/dev/shm/wcheck_$J G pool s sd i=0 rc=0 pids=() t0=$SECONDS
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ -s $R2/results/pid_fusion_$doc.json ] || { echo "no document $doc"; return 1; }
  python -I -m py_compile $R2/scripts/pid_v2_wcheck.py || return 1
  mkdir -p $D || return 1
  echo "weight check of $doc; vectors $ft; node $(hostname); $(date '+%F %T')"
  for pool in pub2 priv2; do for s in 0 1 2 3 4 5; do
    if [ "$s" -le 3 ]; then sd="0 1 2 3"; else sd="4 5"; fi
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${G[$(( i % ${#G[@]} ))]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_v2_offsim_par.py --fusion $R2/results/pid_fusion_$doc.json \
      --ft-dir $ft --steps 50 --seeds $sd --workers 6 --only $pool:$s --dump $D --out /dev/null < /dev/null > logs/nano4/wcheck-$J-split-$pool-$s.log 2>&1 &
    pids+=($!); i=$((i + 1))
  done; done
  for i in "${pids[@]}"; do wait "$i" || rc=1; done
  echo "splits done at $(date +%T) ($(( (SECONDS - t0) / 60 )) min): $(ls $D | grep -c '^split_.*\.pkl$') of 12 files"
  [ $rc = 0 ] || { echo "a split failed (logs/nano4/wcheck-$J-split-*.log):"; tail -q -n 2 logs/nano4/wcheck-$J-split-*.log | cut -c1-200 | head -8; find $D -delete 2> /dev/null; return 1; }
  python -I $R2/scripts/pid_v2_wcheck.py --fusion $R2/results/pid_fusion_$doc.json --dump $D --out $R2/results/final/v2_wcheck_$doc.json 2>&1 | cut -c1-260 || rc=1
  find $D -delete 2> /dev/null
  echo "done at $(date +%T) ($(( (SECONDS - t0) / 60 )) min), rc $rc"
  return $rc
}
main "$@"; exit $?
