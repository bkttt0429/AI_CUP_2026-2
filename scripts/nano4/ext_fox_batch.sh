#!/usr/bin/env bash
# One batch of rank folders of the Fox Go Dataset (METHOD_NOTES 2.46, 2.50), from downloaded archives to a selection:
#   ingest every folder (archives /work/u2507115/ext_fox/raw_<folder> -> compact parts parts_<folder>; the archives are removed)
#   -> overlap gate against the competition's training games (counts only; no fold 0, no test sets, no account read)
#   -> if the gate has hits: the pair probe; a pair that is the same record, or one record being the start of the other,
#      stops the job here (nothing is selected, the parts stay for inspection)
#   -> selection (per folder: <accounts> accounts x <games> games, accounts with at least <need> eligible games; accounts and
#      games of the earlier selections are left out) -> the compact parts are removed.
# usage: [JOINT=1] ext_fox_batch.sh <name> <accounts> <games> <need> <seed> <folder> [<folder> ...]
# Output: /work/u2507115/ext_fox/<name>/select.csv, select.json; results/final/ext_fox_overlap_<name>.json (and ext_fox_match_<name>.json).
set -euo pipefail
X=/work/u2507115/ext_fox
R2=/home/u2507115/BK_AICUP_2026-2
PY="$X/tools_venv/bin/python -I"
name=$1; accounts=$2; games=$3; need=$4; seed=$5
shift 5
parts=(); named=()
for r in "$@"; do
  $PY $R2/scripts/ext_fox_ingest.py --stage ingest --archives $X/raw_$r --out $X/parts_$r
  rm -rf "$X/raw_$r"
  parts+=("$X/parts_$r"); named+=("$r=$X/parts_$r")
done
gate=$R2/results/final/ext_fox_overlap_$name.json
$PY $R2/scripts/ext_fox_ingest.py --stage overlap --parts "${parts[@]}" --out $gate
hits=$(python3 -c "import json; print(json.load(open('$gate'))['matches'])")
echo "overlap gate: $hits competition game(s) share their first 60 moves with an external record"
if [ "$hits" != 0 ]; then
  probe=$R2/results/final/ext_fox_match_$name.json
  $PY $R2/scripts/ext_fox_match_probe.py --parts "${named[@]}" --out $probe
  same=$(python3 -c "
import json
d = json.load(open('$probe'))
print(sum(1 for x in d if x['whole_record_equal'] or x['moves_equal_from_start'] >= min(x['competition_moves'], x['external_moves'])))")
  echo "pairs that are the same record or one the start of the other: $same"
  if [ "$same" != 0 ]; then echo "OVERLAP: stopped before the selection; tell the user"; exit 1; fi
fi
skip=()
for s in $X/*/select.csv; do [ -s "$s" ] && [ "$s" != "$X/$name/select.csv" ] && skip+=("$s"); done  # every earlier selection: their accounts and games are left out
joint=(); [ "${JOINT:-0}" = 1 ] && joint=(--joint)  # JOINT=1: count an account's games over all the folders of the batch
$PY $R2/scripts/ext_fox_select.py --by-account "${joint[@]}" --parts "${named[@]}" --skip "${skip[@]}" --accounts "$accounts" --games "$games" --need "$need" --seed "$seed" --out $X/$name
rm -rf "${parts[@]}"
df -h /work/u2507115 | awk 'NR==2{print "work free:", $4}'
