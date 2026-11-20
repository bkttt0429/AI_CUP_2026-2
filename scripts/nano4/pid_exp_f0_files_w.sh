#!/usr/bin/env bash
# METHOD_NOTES 2.233: the submission file of a composition whose FUSION WEIGHTS come from another document (weights re-fitted
# on the pools' own candidates) instead of v2ge's. = scripts/nano4/pid_exp_f0_files.sh with one step in between:
#   1. data/day_scripts/build_v2h.py <name> <composition>      (real-pool document, weights of v2ge)
#   2. the 9 weights are replaced by those of results/pid_fusion_<weights document>.json (same components, checked)
#   3. scripts/nano4/pid_exp_compfile.sh <name> <file to compare with>   -> results/final/not_for_upload/pid_<name>_a2k_p3.csv
#   pid_exp_f0_files_w.sh <name>=<composition> <weights document> [<file in upload/ to compare with>]
# A file goes to upload/ by hand only after its candidate has passed the rule. From the AI_CUP_2026-1 checkout.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 c=$1 w=$2 cmp=${3:-pid_v2fin9_w_a2k_p3.csv} n
  n=${c%%=*}
  [ -s $R2/results/pid_fusion_$w.json ] || { echo "no weights document $w"; return 1; }
  [ -e $R2/results/pid_fusion_$n.json ] && { echo "document $n exists already"; return 1; }
  python -I $R2/data/day_scripts/build_v2h.py "$n" "${c#*=}" || { echo "document $n was not built"; return 1; }
  python3 -I - "$R2/results/pid_fusion_$n.json" "$R2/results/pid_fusion_$w.json" <<'PY' || { echo "the weights were not replaced"; return 1; }
import json, os, sys
f, g = sys.argv[1], sys.argv[2]
d, s = json.load(open(f)), json.load(open(g))
assert list(d["components"]) == list(s["components"]) and len(s["weights"]) == len(d["components"])
old = d["weights"]
d["weights"] = [float(x) for x in s["weights"]]
d["note"] = (d.get("note") or "") + f" | fusion weights of {os.path.basename(g)} (METHOD_NOTES 2.233) in place of v2ge's"
json.dump(d, open(f + ".tmp", "w"), indent=1)
os.replace(f + ".tmp", f)
print("weights", dict(zip(d["components"], [round(x, 4) for x in old])), "->", dict(zip(d["components"], [round(x, 4) for x in d["weights"]])))
PY
  bash $R2/scripts/nano4/pid_exp_compfile.sh "$n" "$cmp"
  local rc=$?
  ls -la $R2/results/final/not_for_upload/ | grep -E "pid_${n}_a2k_p3\.csv"
  echo "done at $(date +%T), rc $rc"
  return $rc
}
main "$@"; exit $?
