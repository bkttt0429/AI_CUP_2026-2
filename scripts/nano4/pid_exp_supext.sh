#!/usr/bin/env bash
# METHOD_NOTES 2.101: supervised per-ply features of the EXTERNAL games the official networks train on (the 116,420 accounts
# of the pilot, scale1 and scale2 selections; scale3 is left out, 2.81), so that networks on supervised input can be trained
# at the scale of the composition's members. int8 parts in the scale of the probe-B store, the subject's plies below 256.
#   pid_exp_supext.sh <first shard> <shards in total>     8 shards per job (a shard = the games whose key hashes to it)
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox B=/work/u2507115/pid_v2/probeb OUT=/work/u2507115/pid_v2/sup_ext/ply
  mkdir -p $OUT
  bash $R2/scripts/nano4/shards_offset.sh 8 "$1" "$2" python -I $R2/scripts/pid_sup_ply_extract.py \
    --select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv \
    --proj $B/proj.npz --stats $B/stats.npz --out $OUT --workers 10 --max-plies 256
  local rc=$?
  echo "parts: $(ls $OUT | grep -c npz), $(du -sh $OUT | cut -f1)"
  return $rc
}
main "$@"; exit $?
