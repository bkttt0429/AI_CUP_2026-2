#!/usr/bin/env bash
# METHOD_NOTES 2.75: per-game rows with the UNPROJECTED trunk statistics (scripts/pid_trunk_extract.py --full), at scale.
#   pid_exp_fullrows2.sh <node k> <nodes M>
# node 0 first does the added official pools (/work/u2507115/pid_v2/fullrows_off2) and writes the key list of the training
# games of folds 1-4 that have no row yet; every node then extracts its share (8 shards of 8 x M) into
# /work/u2507115/pid_v2/fullrows_train. Resumable per part. 8 GPUs, from the AI_CUP_2026-1 checkout.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 D=/work/u2507115/pid_v2 k=$1 M=$2
  mkdir -p $D/fullrows_train
  if [ "$k" = 0 ]; then
    bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_trunk_extract.py --official2 --tests-dir $R2/data/tests_1111 --full --out $D/fullrows_off2 --workers 10 || echo "official pools FAILED"
    echo "official pools: $(ls $D/fullrows_off2 | grep -c npz) parts"
    # the first whole-pipeline check of the new component, beside the extraction (training rows: the probe's sample)
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_v2_offsim_embfull.py --save $D/embfull_probe.pt --out $R2/results/final/v2_offsim_embfull_probe.json \
      > logs/nano4/embfull-${SLURM_JOB_ID:-local}.log 2>&1 &
    [ -s $D/fullrows_train/keys.txt ] || python -I $R2/scripts/pid_fullrows_keys.py $D/fullrows_train/keys.txt $D/fullrows/rows
  fi
  local n=0
  until [ -s $D/fullrows_train/keys.txt ]; do sleep 30; n=$((n + 1)); [ $n -gt 60 ] && { echo "no key list"; return 1; }; done
  bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_trunk_extract.py --keys $D/fullrows_train/keys.txt --full --out $D/fullrows_train/rows --node "$k" --nodes "$M" --workers 10
  echo "node $k of $M done: $(ls $D/fullrows_train/rows | grep -c npz) parts in all; $(du -sh $D/fullrows_train/rows | cut -f1)"
}
main "$@"; exit $?
