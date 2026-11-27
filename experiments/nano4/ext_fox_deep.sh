#!/usr/bin/env bash
# METHOD_NOTES 2.132: MORE games of external accounts that train already ("deeper" accounts). The selections so far gave
# every account about 10 games; the accounts chosen here get <games> further games each, for networks that read the second
# supervised feature set alone (type C): only those features (about 6 KB per game) and the plies' points are stored.
#   ingest the folders again (archives /work/u2507115/ext_fox/raw_<folder>, removed after the ingest; their overlap gate was
#     run when they were first used) -> compact parts
#   -> selection: accounts of the earlier selections with at least <need> eligible games over these folders, <accounts> of
#      them at random, <games> games each that no earlier selection holds (scripts/ext_fox_select.py --deepen --joint)
#   -> the compact parts are removed
#   -> light per-ply parts (points and movers, no trunk): /work/u2507115/ext_fox/<name>/ply
#   -> the 68 features of every game: /work/u2507115/pid_v2/supj/ext_<name>
# usage: ext_fox_deep.sh <name> <accounts> <games> <need> <seed> <folder> [<folder> ...]
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local X=/work/u2507115/ext_fox R2=/home/u2507115/BK_AICUP_2026-2 P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj J=${SLURM_JOB_ID:-local}
  local PY="$X/tools_venv/bin/python -I" name=$1 accounts=$2 games=$3 need=$4 seed=$5 r named=() pids=() rc=0
  shift 5
  df -h /work/u2507115 /home/u2507115 | awk 'NR>1{print "free:", $4, $6}'
  # the options added for this line have not run yet: fail here, not after the extraction, if one of the scripts does not load
  python -I -m py_compile $R2/scripts/pid_seq_train.py $R2/scripts/pid_ext_ply_extract.py $R2/scripts/ext_fox_select.py || { echo "a script does not compile"; return 1; }
  python -I $R2/scripts/pid_seq_train.py --help | grep -cE -- '--(ext-deep|no-deep|ext-by-games|drop-fold)' || { echo "the trainer does not show the new options"; return 1; }
  for r in "$@"; do
    named+=("$r=$X/parts_$r")
    if [ -d "$X/raw_$r" ]; then
      (
        rm -rf "${X:?}/parts_${r:?}"
        if $PY $R2/scripts/ext_fox_ingest.py --stage ingest --archives "$X/raw_$r" --out "$X/parts_$r" > "logs/nano4/deep-$J-ingest-$r.log" 2>&1; then
          rm -rf "${X:?}/raw_${r:?}"; echo "$r: $(tail -1 "logs/nano4/deep-$J-ingest-$r.log" | cut -c1-160)"
        else
          echo "$r: INGEST FAILED (logs/nano4/deep-$J-ingest-$r.log)"
        fi
      ) &
      pids+=($!)
    fi
  done
  for r in "${pids[@]}"; do wait "$r"; done
  for r in "$@"; do [ -d "$X/parts_$r" ] || { echo "no parts for $r"; return 1; }; done
  $PY $R2/scripts/ext_fox_select.py --deepen --joint --by-account --parts "${named[@]}" \
    --skip $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv \
    --accounts "$accounts" --games "$games" --need "$need" --seed "$seed" --out "$X/$name" || { echo "selection failed"; return 1; }
  for r in "$@"; do rm -rf "${X:?}/parts_${r:?}"; done
  echo "selection: $(($(wc -l < "$X/$name/select.csv") - 1)) games, $(du -sh "$X/$name/select.csv" | cut -f1); $(df -h /work/u2507115 | awk 'NR==2{print "work free:", $4}')"
  echo "== light per-ply parts at $(date +%H:%M:%S)"
  bash $R2/scripts/nano4/shards_offset.sh 8 0 8 python -I $R2/scripts/pid_ext_ply_extract.py --light --select "$X/$name/select.csv" --out "$X/$name/ply" --workers 10 || rc=1
  echo "light parts: $(ls "$X/$name/ply" 2>/dev/null | grep -c 'npz$') files, $(du -sh "$X/$name/ply" 2>/dev/null | cut -f1)"
  echo "== the 68 features at $(date +%H:%M:%S)"
  bash $R2/scripts/nano4/shards_offset.sh 8 0 8 python -I $R2/scripts/pid_sup_ply_extract.py --select "$X/$name/select.csv" --max-plies 256 --proj $P/proj3.npz --scalars --stats $P/stats_js.npz \
    --workers 10 --out "$O/ext_$name" || rc=1
  echo "features: $(ls "$O/ext_$name" 2>/dev/null | grep -c 'npz$') parts, $(du -sh "$O/ext_$name" 2>/dev/null | cut -f1); $(df -h /work/u2507115 | awk 'NR==2{print "work free:", $4}') at $(date +%H:%M:%S)"
  return $rc
}
main "$@"; exit $?
