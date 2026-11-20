#!/usr/bin/env bash
# METHOD_NOTES 2.165: more external accounts ("scale3b") - the accounts of the rank folders 3d 1d 1k 3k 4k of the Fox Go
# Dataset with at least 12 eligible games that no earlier selection holds, 8 games each (the selection rule of scale3,
# notes 2.52; that batch's features were deleted on 11-14 for space). Stages, one job each:
#   ingest <folder>       downloaded archives /work/u2507115/ext_fox/raw_<folder> -> compact parts parts_<folder>; archives removed
#   select                overlap gate against the competition's training games (counts only; no fold 0, no test sets, no
#                         account read; a hit that is the same record stops the job) -> selection scale3b/select.csv -> parts removed
#   ply <first shard> work|home    per-ply trunk vectors (plain inputs), 8 of 48 shards, written compressed to
#                         /work/u2507115/ext_fox/scale3b/ply  or  ~/BK_AICUP_2026-2/data/ext_ply2/scale3b
#   sup <first shard>     second feature set (js), 8 of 32 shards -> ~/BK_AICUP_2026-2/data/supj_ext/scale3b
#   check                 parts and games of the three stores against the selection
# From the AI_CUP_2026-1 checkout. Do not edit while a job runs it.
set -uo pipefail
main() {
  local X=/work/u2507115/ext_fox R2=/home/u2507115/BK_AICUP_2026-2 P=/work/u2507115/pid_v2/probep mode=$1 r s NAME=scale3b
  local PY="$X/tools_venv/bin/python -I" FOLDERS="3d 1d 1k 3k 4k"
  local PLY_W=$X/$NAME/ply PLY_H=$R2/data/ext_ply2/$NAME SUP_H=$R2/data/supj_ext/$NAME SEL=$X/$NAME/select.csv
  case "$mode" in
    ingest)
      r=$2
      [ -d "$X/raw_$r" ] || { echo "no $X/raw_$r"; return 1; }
      $PY $R2/scripts/ext_fox_ingest.py --stage ingest --archives $X/raw_$r --out $X/parts_$r || { echo "ingest of $r FAILED (archives kept)"; return 1; }
      rm -rf "$X/raw_$r"
      echo "$r: parts $(du -sh $X/parts_$r | cut -f1); work free $(df -BM --output=avail /work/u2507115 | tail -1)" ;;
    select)
      local parts=() named=() gate=$R2/results/final/ext_fox_overlap_$NAME.json probe=$R2/results/final/ext_fox_match_$NAME.json hits same
      for r in $FOLDERS; do [ -d "$X/parts_$r" ] || { echo "no parts of $r"; return 1; }; parts+=("$X/parts_$r"); named+=("$r=$X/parts_$r"); done
      $PY $R2/scripts/ext_fox_ingest.py --stage overlap --parts "${parts[@]}" --out $gate || return 1
      hits=$(python3 -c "import json; print(json.load(open('$gate'))['matches'])")
      echo "overlap gate: $hits competition game(s) share their first 60 moves with an external record"
      if [ "$hits" != 0 ]; then
        $PY $R2/scripts/ext_fox_match_probe.py --parts "${named[@]}" --out $probe || return 1
        same=$(python3 -c "
import json
d = json.load(open('$probe'))
print(sum(1 for x in d if x['whole_record_equal'] or x['moves_equal_from_start'] >= min(x['competition_moves'], x['external_moves'])))")
        echo "pairs that are the same record or one the start of the other: $same"
        if [ "$same" != 0 ]; then echo "OVERLAP: stopped before the selection; tell the user"; return 1; fi
      fi
      $PY $R2/scripts/ext_fox_select.py --joint --by-account --parts "${named[@]}" --skip $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv \
        --accounts "${ACCOUNTS:-90000}" --games 8 --need 12 --seed 5 --out $X/$NAME || return 1
      rm -rf "${parts[@]}"
      echo "selection: $(($(wc -l < $SEL) - 1)) games, $(du -sh $SEL | cut -f1); $(head -c 600 $X/$NAME/select.json | tr '\n' ' ')"
      df -h /work/u2507115 /home/u2507115 | tail -2 ;;
    ply)
      local out=$PLY_W; [ "${3:-work}" = home ] && out=$PLY_H
      [ -s "$SEL" ] || { echo "no selection"; return 1; }
      mkdir -p "$out"
      bash $R2/scripts/nano4/shards_offset.sh 8 "$2" 48 python -I $R2/scripts/pid_ext_ply_extract.py --workers 10 --compress --select $SEL --out "$out" || { echo "a shard failed"; return 1; }
      echo "parts in $out: $(ls $out | grep -c '[0-9]\.npz$'), $(du -sh $out | cut -f1)"; df -h /work/u2507115 /home/u2507115 | tail -2 ;;
    sup)
      [ -s "$SEL" ] || { echo "no selection"; return 1; }
      mkdir -p "$SUP_H"
      bash $R2/scripts/nano4/shards_offset.sh 8 "$2" 32 python -I $R2/scripts/pid_sup_ply_extract.py --select $SEL --max-plies 256 --proj $P/proj3.npz --scalars --stats $P/stats_js.npz --workers 10 --compress --out $SUP_H || { echo "a shard failed"; return 1; }
      echo "parts in $SUP_H: $(ls $SUP_H | grep -c '[0-9]\.npz$'), $(du -sh $SUP_H | cut -f1)"; df -h /work/u2507115 /home/u2507115 | tail -2 ;;
    check)
      python -I - "$SEL" "$PLY_W" "$PLY_H" "$SUP_H" <<'EOF'
import csv, glob, sys
import numpy as np
sel, ply_w, ply_h, sup = sys.argv[1:5]
csv.field_size_limit(10 ** 9)
with open(sel, newline="") as fh:
    rows = [(r["game_key"], r["account"]) for r in csv.DictReader(fh)]
want = [k for k, _ in rows]
print(f"selection: {len(want)} games ({len(set(want))} distinct keys), {len(set(a for _, a in rows))} accounts")
got = []
for d in (ply_w, ply_h):
    fs = sorted(f for f in glob.glob(d + "/ply_x_*.npz") if not f.endswith(".tmp.npz"))
    ks = np.concatenate([np.load(f)["key"].astype(str) for f in fs]) if fs else np.array([], str)
    got.append(ks)
    print(f"{d}: {len(fs)} parts, {len(ks)} games")
allk = np.concatenate(got)
print(f"ply: {len(allk)} games, {len(set(allk.tolist()))} distinct, {len(set(want) - set(allk.tolist()))} selected games without a part row")
fs = sorted(f for f in glob.glob(sup + "/sup_*.npz") if not f.endswith(".tmp.npz"))
n = 0; with_plies = 0; keys = set()
for f in fs:
    z = np.load(f)
    k = z["keys"].astype(str); keys.update(k.tolist()); n += len(k); with_plies += int((z["n"] > 0).sum())
print(f"{sup}: {len(fs)} parts, {n} games ({with_plies} with plies), {len(set(want) - keys)} selected games without a row")
ok = len(set(want) - set(allk.tolist())) < 0.001 * len(want) and len(set(want) - keys) < 0.001 * len(want)
print("CHECK", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
EOF
      ;;
    *) echo "usage: ext_fox_scale3b.sh ingest <folder> | select | ply <first shard> work|home | sup <first shard> | check"; return 2 ;;
  esac
}
main "$@"; exit $?
