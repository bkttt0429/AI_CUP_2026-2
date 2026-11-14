#!/usr/bin/env bash
# METHOD_NOTES 2.110: the SECOND supervised feature set ("js": the projection fitted jointly over pooled / played-point /
# policy-expected vectors, 64 dims, + 4 policy scalars; probe P6) for every game the official networks use. int8 parts in
# the scale of the probe store /work/u2507115/pid_v2/probep/ply_js (float, the 115,011 probe-B games, kept as it is).
#   pid_exp_o14_x.sh comp <first shard> <shards in total>   the other 679,039 competition games of folds 1-4 -> supj/comp
#   pid_exp_o14_x.sh pools                                   the official pools' games                       -> supj/pools
#   pid_exp_o14_x.sh ext <first shard> <shards in total>    the 1,261,070 games of the 116,420 external accounts -> supj/ext
# 8 shards per job. From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox B=/work/u2507115/pid_v2/probeb P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj rc
  mkdir -p $O
  if [ ! -s $P/stats_js.npz ]; then  # one scale for the whole set: the probe store's (several jobs may write the same file)
    python -I -c "import sys; sys.path.insert(0, '$R2/scripts'); import pid_sup; print('stats over', pid_sup.write_stats(['$P/ply_js'], '$P/stats_js.$$.npz'), 'plies')" && mv $P/stats_js.$$.npz $P/stats_js.npz
  fi
  local common="--proj $P/proj3.npz --scalars --stats $P/stats_js.npz --workers 10"
  case $1 in
    comp)  # shellcheck disable=SC2086
      bash $R2/scripts/nano4/shards_offset.sh 8 "$2" "$3" python -I $R2/scripts/pid_sup_ply_extract.py --keys /work/u2507115/pid_v2/fullrows_train/keys.txt --skip-keys $B/keys.txt $common --out $O/comp; rc=$? ;;
    pools)  # shellcheck disable=SC2086
      bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_sup_ply_extract.py --pools $R2/data/tests_1111 $common --out $O/pools; rc=$? ;;
    ext)  # shellcheck disable=SC2086
      bash $R2/scripts/nano4/shards_offset.sh 8 "$2" "$3" python -I $R2/scripts/pid_sup_ply_extract.py --select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv --max-plies 256 $common --out $O/ext; rc=$? ;;
    *) echo "unknown case $1"; return 2 ;;
  esac
  echo "parts: comp $(ls $O/comp 2>/dev/null | grep -c 'npz$'), pools $(ls $O/pools 2>/dev/null | grep -c 'npz$'), ext $(ls $O/ext 2>/dev/null | grep -c 'npz$'); $(du -sh $O | cut -f1)"
  return $rc
}
main "$@"; exit $?
