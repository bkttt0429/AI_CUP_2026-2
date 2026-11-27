#!/usr/bin/env bash
# METHOD_NOTES 2.186: the next batch of external accounts ("scale4") - the accounts of the Fox Go Dataset
# (github.com/featurecat/go-dataset, GPL-3.0) with at least 8 eligible games that no earlier selection holds, counted over
# ALL fifteen rank folders used so far (an account's games are spread over neighbouring rank folders), 8 games each, at most
# ACCOUNTS accounts (default 50,000: the disk space). Stages, one job each:
#   select                         everything temporary lives in /dev/shm of the node (nothing on /work but the result):
#                                  download the fifteen folders (4.7 GB; sizes checked against the repository listing; data
#                                  only, nothing executed) -> ingest (account names become salted keys; the salt never
#                                  leaves /work/u2507115/ext_fox) -> overlap gate against the competition's training games
#                                  (counts only; no fold 0, no test sets; a hit that is the same record stops the job)
#                                  -> joint selection -> /work/u2507115/ext_fox/scale4/select.csv
#   ply <first shard> <out dir>    per-ply trunk vectors (plain inputs), 8 of 48 shards, compressed
#   sup <first shard> <out dir>    second feature set (js), 8 of 32 shards, compressed
#   check <dir> [<dir> ...]        parts and games of the ply and sup folders against the selection
# From the AI_CUP_2026-1 checkout. select: 4 GPUs' worth of cores (-c 48); ply / sup: 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
select_stage() {  # everything under $T (in /dev/shm); the caller removes $T whatever happens here
  local X=$1 R2=$2 NAME=$3 J=$4 T=$5 SEL=$6 PY=$7 FOLDERS=$8
  local PAR=${PAR:-4} parts=() named=() n ok fails=0 hits same r d gate=$R2/results/final/ext_fox_overlap_$NAME.json probe=$R2/results/final/ext_fox_match_$NAME.json
  [ -e "$SEL" ] && { echo "$SEL exists already"; return 1; }
  [ "$(df -BM --output=avail /dev/shm | tail -1 | tr -dc '0-9')" -ge 60000 ] || { echo "/dev/shm of $(hostname) has less than 60 GB free"; return 1; }
  mkdir -p $T/tmp $X/$NAME || return 1
  # 1. download (four folders at a time)
  fetch() {
    local r=$1 d=$T/raw_$1 n ok=0 name size url
    mkdir "$d" || return 1
    curl -sS -A "curl" --retry 3 "https://api.github.com/repos/featurecat/go-dataset/contents/$r" > "$d/.listing.json" || { echo "$r: listing failed"; return 1; }
    python3 -I -c 'import json,sys
for x in json.load(open(sys.argv[1])):
    n = x["name"]
    if "/" in n or n.startswith("."):
        continue
    print(n, x["size"], x["download_url"])' "$d/.listing.json" > "$d/.files" || { echo "$r: listing unreadable: $(head -c 200 "$d/.listing.json")"; return 1; }
    n=$(wc -l < "$d/.files")
    while read -r name size url; do
      curl -sS -L -A "curl" --retry 3 -o "$d/$name" "$url" && [ "$(stat -c %s "$d/$name")" = "$size" ] && ok=$((ok + 1)) || echo "$r: $name incomplete"
    done < "$d/.files"
    rm -f "$d/.listing.json" "$d/.files"
    echo "$r: $ok of $n files, $(du -sh "$d" | cut -f1) ($(date +%T))"
    [ "$n" -gt 0 ] && [ "$ok" = "$n" ]
  }
  local pids=()
  for r in $FOLDERS; do
    fetch $r > $T/fetch_$r.log 2>&1 &
    pids+=($!)
    if [ "${#pids[@]}" -ge 4 ]; then for d in "${pids[@]}"; do wait "$d" || fails=$((fails + 1)); done; pids=(); fi
  done
  for d in "${pids[@]}"; do wait "$d" || fails=$((fails + 1)); done
  cat $T/fetch_*.log
  [ "$fails" = 0 ] || { echo "$fails folder(s) were not downloaded completely: stopped"; return 1; }
  echo "downloaded: $(du -sh --total $T/raw_* | tail -1 | cut -f1) at $(date +%T)"
  # 2. ingest, PAR folders at a time; the archives go as soon as their folder is read
  pids=(); fails=0
  for r in $FOLDERS; do
    ( $PY $R2/scripts/ext_fox_ingest.py --stage ingest --archives $T/raw_$r --out $T/parts_$r --tmp $T/tmp --workers 12 > logs/nano4/scale4-$J-ingest-$r.log 2>&1 \
        && { rm -rf "$T/raw_$r"; echo "$r: $(tail -1 logs/nano4/scale4-$J-ingest-$r.log | cut -c1-160)"; } \
        || { echo "$r: INGEST FAILED (logs/nano4/scale4-$J-ingest-$r.log)"; exit 1; } ) &
    pids+=($!)
    if [ "${#pids[@]}" -ge "$PAR" ]; then for d in "${pids[@]}"; do wait "$d" || fails=$((fails + 1)); done; pids=(); fi
  done
  for d in "${pids[@]}"; do wait "$d" || fails=$((fails + 1)); done
  [ "$fails" = 0 ] || { echo "$fails ingest(s) failed: stopped"; return 1; }
  for r in $FOLDERS; do parts+=("$T/parts_$r"); named+=("$r=$T/parts_$r"); done
  echo "parts: $(du -sh --total "${parts[@]}" | tail -1 | cut -f1) at $(date +%T)"
  # 3. overlap gate
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
  # 4. selection (the earlier selections' accounts and games are left out)
  $PY $R2/scripts/ext_fox_select.py --joint --by-account --parts "${named[@]}" \
    --skip $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv $X/scale3b/select.csv \
    --accounts "${ACCOUNTS:-50000}" --games 8 --need 8 --seed 6 --out $X/$NAME || return 1
  echo "selection: $(($(wc -l < $SEL) - 1)) games, $(du -sh $SEL | cut -f1); $(head -c 700 $X/$NAME/select.json | tr '\n' ' ')"
  df -h /work/u2507115 /home/u2507115 | tail -2
}
main() {
  local X=/work/u2507115/ext_fox R2=/home/u2507115/BK_AICUP_2026-2 P=/work/u2507115/pid_v2/probep mode=${1:?stage} NAME=scale4 J=${SLURM_JOB_ID:-local}
  local PY="$X/tools_venv/bin/python -I" FOLDERS="6d 5k 8k 2k 2d 4d 7d 8d 9d 5d 3d 1d 1k 3k 4k" SEL=$X/$NAME/select.csv r d
  case "$mode" in
    select)
      local T=/dev/shm/fox_$J rc
      select_stage "$X" "$R2" "$NAME" "$J" "$T" "$SEL" "$PY" "$FOLDERS"; rc=$?
      find "$T" -delete 2> /dev/null
      echo "/dev/shm cleaned: $(ls -d $T 2> /dev/null | wc -l) left"
      return $rc ;;
    ply)
      local out=${3:?out dir}
      [ -s "$SEL" ] && head -1 "$SEL" | grep -q sgf_content || { echo "no selection with the SGF text ($SEL)"; return 1; }
      mkdir -p "$out"
      bash $R2/scripts/nano4/shards_offset.sh 8 "$2" 48 python -I $R2/scripts/pid_ext_ply_extract.py --workers 10 --compress --select $SEL --out "$out" || { echo "a shard failed"; return 1; }
      echo "parts in $out: $(ls $out | grep -c '[0-9]\.npz$'), $(du -sh $out | cut -f1)"; df -h /work/u2507115 /home/u2507115 | tail -2 ;;
    sup)
      local out=${3:?out dir}
      [ -s "$SEL" ] && head -1 "$SEL" | grep -q sgf_content || { echo "no selection with the SGF text ($SEL)"; return 1; }
      mkdir -p "$out"
      bash $R2/scripts/nano4/shards_offset.sh 8 "$2" 32 python -I $R2/scripts/pid_sup_ply_extract.py --select $SEL --max-plies 256 --proj $P/proj3.npz --scalars --stats $P/stats_js.npz --workers 10 --compress --out "$out" || { echo "a shard failed"; return 1; }
      echo "parts in $out: $(ls $out | grep -c '[0-9]\.npz$'), $(du -sh $out | cut -f1)"; df -h /work/u2507115 /home/u2507115 | tail -2 ;;
    check)
      shift
      python -I - "$SEL" "$@" <<'EOF'
import csv, glob, sys
import numpy as np
sel, dirs = sys.argv[1], sys.argv[2:]
csv.field_size_limit(10 ** 9)
with open(sel, newline="") as fh:
    rows = [(r["game_key"], r["account"]) for r in csv.DictReader(fh)]
want = set(k for k, _ in rows)
print(f"selection: {len(rows)} games ({len(want)} distinct keys), {len(set(a for _, a in rows))} accounts")
ply, sup = set(), set()
for d in dirs:
    fp = sorted(f for f in glob.glob(d + "/ply_x_*.npz") if not f.endswith(".tmp.npz"))
    fs = sorted(f for f in glob.glob(d + "/sup_*.npz") if not f.endswith(".tmp.npz"))
    for f in fp:
        ply.update(np.load(f)["key"].astype(str).tolist())
    n = 0
    for f in fs:
        z = np.load(f); k = z["keys"].astype(str)
        sup.update(k[z["n"] > 0].tolist()); n += len(k)
    print(f"{d}: {len(fp)} ply parts, {len(fs)} sup parts")
mp, ms = len(want - ply), len(want - sup)
print(f"ply: {len(ply & want)} of {len(want)} selected games ({mp} missing); sup: {len(sup & want)} ({ms} missing)")
ok = mp < 0.001 * len(want) and ms < 0.001 * len(want)
print("CHECK PASSED" if ok else "CHECK FAILED")
sys.exit(0 if ok else 1)
EOF
      [ $? = 0 ] && touch $X/$NAME/CHECK_PASSED ;;
    *) echo "usage: ext_fox_scale4.sh select | ply <first shard> <out dir> | sup <first shard> <out dir> | check <dir> ..."; return 2 ;;
  esac
}
main "$@"; exit $?
