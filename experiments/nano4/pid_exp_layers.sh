#!/usr/bin/env bash
# METHOD_NOTES 2.172 (layer probe): how much identity information does each depth of the humanv0 trunk carry? Forward passes
# only - the cheap question before any fine-tuning of the trunk.
#   1. the probe sample of probe A (scripts/pid_probe_fullrows.py --stage keys: the simulated pools of folds 1 and 2, up to 16
#      games of every player of folds 3 and 4; never fold 0)
#   2. per-game means of the residual stream at eight depths (scripts/pid_layer_rows.py), one shard per GPU
#   3. the instrument of notes 2.88 on every depth and on the depths together (scripts/pid_probe_layers.py)
# One job with 8 GPUs, from the AI_CUP_2026-1 checkout. The rows (about 1 GB) are removed after a successful evaluation.
# Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 D=/work/u2507115/pid_v2/layerrows free
  python -I -m py_compile $R2/scripts/pid_layer_rows.py $R2/scripts/pid_probe_layers.py $R2/scripts/pid_probe_fullrows.py || return 1
  free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')
  [ "$free" -ge 5500 ] || { echo "/work has only $free MB free: not started (the rows need about 1 GB above the 4 GB floor)"; return 1; }
  mkdir -p $D
  [ -s $D/keys.txt ] || python -I $R2/scripts/pid_probe_fullrows.py --stage keys --out $D/keys.txt || return 1
  wc -l < $D/keys.txt
  bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_layer_rows.py --keys $D/keys.txt --out $D/rows --workers 10 || { echo "a shard FAILED"; grep -hE "check:|Error|error" logs/nano4/shard*-${SLURM_JOB_ID:-local}.log | sort -u | head -8; return 1; }
  grep -h "check:" logs/nano4/shard0-${SLURM_JOB_ID:-local}.log | head -1
  echo "rows: $(ls $D/rows | grep -c '[0-9]\.npz$') parts, $(du -sh $D/rows | cut -f1)"
  OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 python -I $R2/scripts/pid_probe_layers.py --rows $D/rows --out $R2/results/final/v2_probe_layers.json || { echo "the evaluation FAILED (rows kept in $D/rows)"; return 1; }
  find $D/rows -name 'layers_s*.npz' -delete; rmdir $D/rows 2> /dev/null
  df -BM --output=avail /work/u2507115 | tail -1
}
main "$@"; exit $?
