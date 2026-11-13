#!/usr/bin/env bash
# Next batch of external accounts after the user freed /work on 2026-11-13 23:55 (METHOD_NOTES 2.52). Runs on the login node
# (submit / query / link only) in the background, next to the batch job that ingests 3d, 1d, 1k, 3k, 4k for "midrank":
#   1. when that job has ingested all five folders, keep their compact parts (hard links: the job removes its own copies)
#   2. when the job has finished and its overlap gate let the selection through: selection "scale3" over the five folders
#      jointly (accounts with >= 12 eligible games, 8 games each, at most 125,000 accounts; accounts and games of every
#      earlier selection left out)
#   3. feature extraction of the selection in two halves, 8 GPUs each -> /work/u2507115/ext_fox/scale3/ply_a, ply_b
# Jobs are submitted only when the shared queue (13 submitted jobs per user) has room.
set -uo pipefail
cd /home/u2507115/BK_AICUP_2026-1
X=/work/u2507115/ext_fox
R2=/home/u2507115/BK_AICUP_2026-2
EX="--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155,177]"
BATCH=772273
FOLDERS="3d 1d 1k 3k 4k"
inq() { squeue -h -j "$1" 2>/dev/null | grep -q .; }
gonewait() { local gone=0; while [ $gone -lt 3 ]; do if inq "$1"; then gone=0; else gone=$((gone + 1)); fi; sleep 20; done; }
submit() {  # retried until the queue takes it
  local out="" n
  while [ -z "$out" ]; do
    n=$(squeue -h -u u2507115 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ] && [ "$n" -lt 13 ]; then out=$(sbatch --parsable "$@" 2>/dev/null); fi
    [ -z "$out" ] && sleep 45
  done
  echo "$out"
}
raw_left() { local r; for r in $FOLDERS; do [ -d "$X/raw_$r" ] && return 0; done; return 1; }
while inq $BATCH && raw_left; do sleep 5; done
if raw_left; then echo "batch job $BATCH ended before all folders were ingested; nothing kept"; exit 1; fi
for r in $FOLDERS; do [ -d "$X/keep_parts_$r" ] || cp -al "$X/parts_$r" "$X/keep_parts_$r"; done
echo "$(date +%H:%M) parts kept: $(du -sh --total $X/keep_parts_* | tail -1 | cut -f1) in $(ls $X/keep_parts_*/ | grep -c npz) files"
gonewait $BATCH
state=$(sacct -j $BATCH -X -n -o State | tr -d ' ')
if [ "$state" != COMPLETED ] || [ ! -s $X/midrank/select.csv ]; then echo "batch job $BATCH: $state, no midrank selection -> scale3 not started (overlap gate or failure; see its log)"; exit 1; fi
echo "$(date +%H:%M) batch job $BATCH completed; overlap gate passed"
named=(); for r in $FOLDERS; do named+=("$r=$X/keep_parts_$r"); done
S=$(submit $EX --gres=gpu:1 -c 12 -t 01:00:00 -J p_extsel5 scripts/nano4/run.sbatch $X/tools_venv/bin/python -I $R2/scripts/ext_fox_select.py --joint --by-account \
  --parts "${named[@]}" --skip $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv $X/midrank/select.csv \
  --accounts 125000 --games 8 --need 12 --seed 5 --out $X/scale3)
echo "$(date +%H:%M) selection job $S"
gonewait "$S"
n=$(python3 -c "import json; print(json.load(open('$X/scale3/select.json'))['games_written'])" 2>/dev/null || echo 0)
if [ "$n" -lt 1000 ]; then echo "selection failed (rows $n); see logs/nano4/p_extsel5-$S.out"; exit 1; fi
h=$(( n / 2 ))
echo "$(date +%H:%M) scale3: $n games selected; halves at row $h"
A=$(submit $EX --gres=gpu:8 -c 96 -t 08:00:00 -J p_extplyE scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_ext_ply_extract.py --select $X/scale3/select.csv --workers 10 --start 0 --stop $h --out $X/scale3/ply_a)
B=$(submit $EX --gres=gpu:8 -c 96 -t 08:00:00 -J p_extplyF scripts/nano4/run.sbatch bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_ext_ply_extract.py --select $X/scale3/select.csv --workers 10 --start $h --stop $n --out $X/scale3/ply_b)
echo "$(date +%H:%M) extraction jobs $A (rows 0-$h) and $B (rows $h-$n)"
echo "$A $B" > $X/scale3/jobs_extract.txt
