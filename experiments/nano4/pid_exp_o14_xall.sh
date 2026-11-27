#!/usr/bin/env bash
# The external games of the second feature set (scripts/nano4/pid_exp_o14_x.sh ext) as ONE job that works through several
# rounds of 8 shards (of 32) in the given order - the account's submit limit does not leave room for four separate jobs.
# Two such jobs with opposite orders meet in the middle (finished parts are skipped).
#   pid_exp_o14_xall.sh <first shard of a round> [<first shard of the next round> ...]      e.g. 0 8 16 24   or   24 16 8 0
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj f rc=0
  mkdir -p $O
  if [ ! -s $P/stats_js.npz ]; then
    python -I -c "import sys; sys.path.insert(0, '$R2/scripts'); import pid_sup; print('stats over', pid_sup.write_stats(['$P/ply_js'], '$P/stats_js.$$.npz'), 'plies')" && mv $P/stats_js.$$.npz $P/stats_js.npz
  fi
  for f in "$@"; do
    echo "== round: shards $f-$((f + 7)) of 32 at $(date +%H:%M:%S)"
    bash $R2/scripts/nano4/shards_offset.sh 8 "$f" 32 python -I $R2/scripts/pid_sup_ply_extract.py --select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv \
      --max-plies 256 --proj $P/proj3.npz --scalars --stats $P/stats_js.npz --workers 10 --out $O/ext || rc=1
    echo "parts so far: ext $(ls $O/ext 2>/dev/null | grep -c 'npz$'); $(du -sh $O | cut -f1)"
  done
  return $rc
}
main "$@"; exit $?
