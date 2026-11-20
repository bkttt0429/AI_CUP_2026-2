#!/usr/bin/env bash
# METHOD_NOTES 2.185 (space plan): the external selections' select.csv files carry the whole SGF text of every selected game
# (2.1 GB together). The trainers read only game_key and account from them; the SGF text is needed again only for a new
# feature extraction. Per selection this job
#   1. writes select_full.csv.gz next to the file and checks that it decompresses to the identical bytes,
#   2. writes the slim file (game_key, account, color; same rows, same order) and checks keys / accounts against the original,
#   3. replaces select.csv by the slim file in one rename.
# Lossless: `gunzip -c select_full.csv.gz > select.csv` restores the original. An extraction (pid_ext_ply_extract.py,
# pid_sup_ply_extract.py --select) needs the full file again.
#   pid_select_slim.sh [<dir> ...]      default: the five selections in use
# From any directory, CPU only. Do not edit while a job runs it.
set -uo pipefail
main() {
  local X=/work/u2507115/ext_fox d f rc=0 before after
  local dirs=("$@")
  [ "${#dirs[@]}" -gt 0 ] || dirs=($X/pilot $X/scale1 $X/scale2_top $X/scale2_5d $X/scale3b)
  before=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')
  [ "$before" -ge 5500 ] || { echo "/work has only $before MB free: not started (the job needs about 0.6 GB for a moment)"; return 1; }
  for d in "${dirs[@]}"; do
    f=$d/select.csv
    [ -s "$f" ] || { echo "$d: no select.csv"; rc=1; continue; }
    if head -c 200 "$f" | head -1 | grep -q "sgf_content"; then :; else echo "$d: select.csv has no sgf_content column (already slim?) - left alone"; continue; fi
    [ -e "$d/select_full.csv.gz" ] && { echo "$d: select_full.csv.gz exists already - left alone"; rc=1; continue; }
    gzip -6 -c "$f" > "$d/select_full.csv.gz.tmp" || { echo "$d: gzip failed"; rm -f "$d/select_full.csv.gz.tmp"; rc=1; continue; }
    if ! gzip -dc "$d/select_full.csv.gz.tmp" | cmp -s - "$f"; then echo "$d: the compressed copy does NOT decompress to the original - nothing changed"; rm -f "$d/select_full.csv.gz.tmp"; rc=1; continue; fi
    python3 -I - "$f" "$d/select.csv.slim.tmp" <<'PY' || { echo "$d: the slim file could not be written - nothing changed"; rm -f "$d/select_full.csv.gz.tmp" "$d/select.csv.slim.tmp"; rc=1; continue; }
import csv, sys
csv.field_size_limit(10 ** 9)
src, dst = sys.argv[1:3]
n = 0
with open(src, newline="") as fh, open(dst, "w", newline="") as out:
    r = csv.DictReader(fh)
    w = csv.writer(out)
    w.writerow(["game_key", "account", "color"])
    for row in r:
        w.writerow([row["game_key"], row["account"], row["color"]])
        n += 1
with open(src, newline="") as a, open(dst, newline="") as b:
    for x, y in zip(csv.DictReader(a), csv.DictReader(b)):
        assert (x["game_key"], x["account"], x["color"]) == (y["game_key"], y["account"], y["color"])
        n -= 1
assert n == 0, "row counts differ"
print(f"slim file checked row by row against the original")
PY
    mv -f "$d/select_full.csv.gz.tmp" "$d/select_full.csv.gz" && mv -f "$d/select.csv.slim.tmp" "$f" || { echo "$d: rename failed"; rc=1; continue; }
    echo "$d: select.csv $(du -m "$f" | cut -f1) MB (slim) + select_full.csv.gz $(du -m "$d/select_full.csv.gz" | cut -f1) MB"
  done
  sleep 60
  after=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')
  echo "/work free: $before MB -> $after MB"
  echo "compute node reaches github.com: $(curl -sI -m 15 https://github.com 2> /dev/null | head -1 | tr -d '\r' || true)"
  return $rc
}
main "$@"; exit $?
