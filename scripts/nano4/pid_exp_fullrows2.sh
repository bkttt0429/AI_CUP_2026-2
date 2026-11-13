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
    [ -s $D/fullrows_train/keys.txt ] || python -I $R2/scripts/pid_fullrows_keys.py $D/fullrows_train/keys.txt  # every game of folds 1-4 (the probe's rows have no phase means)
  fi
  local n=0
  until [ -s $D/fullrows_train/keys.txt ]; do sleep 30; n=$((n + 1)); [ $n -gt 60 ] && { echo "no key list"; return 1; }; done
  local fast=()
  if [ "$k" != 0 ]; then
    # the faster replay (target side only; 2026-11-14 06:55) checked on this node first: node 0's first part again, value by value
    n=0
    until [ -s $D/fullrows_train/rows/part_s000_p0000.npz ]; do sleep 20; n=$((n + 1)); [ $n -gt 60 ] && break; done
    rm -rf $D/fullrows_train/check_$k
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} python -I $R2/scripts/pid_trunk_extract.py --keys $D/fullrows_train/keys.txt --full --fast \
      --out $D/fullrows_train/check_$k --node 0 --nodes "$M" --shard 0 --nshards 8 --max-parts 1 --workers 20 > logs/nano4/frow-check-${SLURM_JOB_ID:-local}.log 2>&1
    if python -I - $D/fullrows_train/rows/part_s000_p0000.npz $D/fullrows_train/check_$k/part_s000_p0000.npz <<'PYEOF'
import sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
ok = list(a["keys"]) == list(b["keys"]) and np.array_equal(a["n_target"], b["n_target"])
d = {k: float(np.abs(a[k].astype(np.float32) - b[k].astype(np.float32)).max()) for k in ("feats", "full", "fullp", "phase")} if ok else {}
print("fast replay against the usual one, first part:", "keys and ply counts equal" if ok else "KEYS OR PLY COUNTS DIFFER", "| largest absolute differences", d)
sys.exit(0 if ok and max(d.values()) < 0.02 else 1)
PYEOF
    then fast=(--fast); else echo "the fast replay is NOT used on this node (see logs/nano4/frow-check-${SLURM_JOB_ID:-local}.log)"; fi
    rm -rf $D/fullrows_train/check_$k
  fi
  bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_trunk_extract.py --keys $D/fullrows_train/keys.txt --full "${fast[@]}" --out $D/fullrows_train/rows --node "$k" --nodes "$M" --workers 10
  echo "node $k of $M done: $(ls $D/fullrows_train/rows | grep -c npz) parts in all; $(du -sh $D/fullrows_train/rows | cut -f1)"
}
main "$@"; exit $?
