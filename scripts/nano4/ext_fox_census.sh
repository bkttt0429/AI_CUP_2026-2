#!/usr/bin/env bash
# METHOD_NOTES 2.73: ingest again the rank folders used so far (archives /work/u2507115/ext_fox/raw_<folder>, removed after
# each ingest; the overlap gate of these folders was run when they were first used) and run scripts/ext_fox_census.py:
# games per account, the second side of the selected games, the cost of selecting the accounts with few games.
# The compact parts STAY (parts_<folder>) for the selection that follows. Up to PAR folders are ingested at a time.
#   [PAR=6] ext_fox_census.sh <folder> [<folder> ...]
set -uo pipefail
X=/work/u2507115/ext_fox
R2=/home/u2507115/BK_AICUP_2026-2
PY="$X/tools_venv/bin/python -I"
PAR=${PAR:-6}
named=()
running=0
for r in "$@"; do
  named+=("$r=$X/parts_$r")
  if [ -d "$X/raw_$r" ]; then
    (
      rm -rf "$X/parts_$r"
      if $PY $R2/scripts/ext_fox_ingest.py --stage ingest --archives $X/raw_$r --out $X/parts_$r > logs/nano4/census-${SLURM_JOB_ID:-local}-$r.log 2>&1; then
        rm -rf "$X/raw_$r"; echo "$r: $(tail -1 logs/nano4/census-${SLURM_JOB_ID:-local}-$r.log | cut -c1-160)"
      else
        echo "$r: INGEST FAILED (logs/nano4/census-${SLURM_JOB_ID:-local}-$r.log)"
      fi
    ) &
    running=$((running + 1))
    if [ "$running" -ge "$PAR" ]; then wait -n; running=$((running - 1)); fi
  fi
done
wait
for r in "$@"; do [ -d "$X/parts_$r" ] || { echo "no parts for $r"; exit 1; }; done
sels=()
for s in $X/*/select.csv; do [ -s "$s" ] && [ "$s" != "$X/scale4a/select.csv" ] && sels+=("$s"); done  # scale4a was selected but never extracted: its accounts count as free
$PY $R2/scripts/ext_fox_census.py --parts "${named[@]}" --selections "${sels[@]}" --out $X/census
df -h /work/u2507115 | awk 'NR==2{print "work free:", $4}'
