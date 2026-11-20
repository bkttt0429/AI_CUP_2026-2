#!/usr/bin/env bash
# METHOD_NOTES 2.148: rebuild the external per-ply parts deleted on 2026-11-16 04:17 (/work/u2507115/ext_fox/ext_ply_home:
# scale1_c, scale2_top, scale2_5d - the only copy; ~/BK_AICUP_2026-2/data/ext_ply is a link to it). Nothing is downloaded:
# the selection files hold the game records, and the commands are the original ones (jobs 769445 and 771730).
#   ext_fox_rebuild.sh c      scale1 rows 700,000-977,750 -> scale1_c   (1,392 parts; 2 h 35 min on 8 GPUs the first time)
#   ext_fox_rebuild.sh d      scale2_top (208 parts) and scale2_5d (248 parts)   (51 min)
# Ends with a count of the parts and of the games against the selection. From the AI_CUP_2026-1 checkout, 8 GPUs.
# Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox which=$1 rc=0 spec name sel rng
  local H=$X/ext_ply_home
  mkdir -p "$H"
  [ "$(readlink -f $R2/data/ext_ply)" = "$H" ] || { echo "$R2/data/ext_ply does not point to $H"; return 1; }
  df -h /work/u2507115 | tail -1
  local jobs=()
  case "$which" in
    c) jobs=("scale1_c $X/scale1/select.csv --start 700000 --stop 977750") ;;
    d) jobs=("scale2_top $X/scale2_top/select.csv" "scale2_5d $X/scale2_5d/select.csv") ;;
    *) echo "usage: ext_fox_rebuild.sh c|d"; return 2 ;;
  esac
  for spec in "${jobs[@]}"; do
    read -r name sel rng <<< "$spec"
    # shellcheck disable=SC2086
    bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_ext_ply_extract.py --workers 10 --select "$sel" $rng --out "$H/$name" || { echo "extraction of $name FAILED"; rc=1; }
    # shellcheck disable=SC2086
    python -I - "$H/$name" "$sel" $rng <<'EOF' || rc=1
import csv, glob, sys
import numpy as np
out, sel, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
start = int(rest[rest.index("--start") + 1]) if "--start" in rest else 0
stop = int(rest[rest.index("--stop") + 1]) if "--stop" in rest else 0
csv.field_size_limit(10 ** 9)
with open(sel, newline="") as fh:
    want = [r["game_key"] for r in csv.DictReader(fh)]
want = want[start:(stop or len(want))]
files = sorted(f for f in glob.glob(out + "/ply_x_*.npz") if not f.endswith(".tmp.npz"))
got = np.concatenate([np.load(f)["key"].astype(str) for f in files]) if files else np.array([], str)
miss = len(set(want) - set(got.tolist()))
print(f"{out}: {len(files)} parts, {len(got)} games ({len(set(got.tolist()))} distinct) of {len(want)} selected; {miss} selected games without a part row (records the replay cannot follow are left out)")
sys.exit(0 if files and len(got) == len(set(got.tolist())) and miss < 0.01 * len(want) else 1)
EOF
  done
  du -sh "$H"/* 2> /dev/null; df -h /work/u2507115 | tail -1
  return $rc
}
main "$@"; exit $?
