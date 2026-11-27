#!/usr/bin/env bash
# Official pseudo-pool check (scripts/pid_v2_offsim.py --alt, METHOD_NOTES 2.42) of several candidate documents against one
# reference document, one after the other:   pid_v2_offsim_many.sh <reference name> <candidate name> ...
# Names are those of results/pid_fusion_<name>.json; results go to results/final/v2_offsim_<candidate>_vs_<reference>.json.
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
ref=$1
shift
for c in "$@"; do
  python -I $R2/scripts/pid_v2_offsim.py --fusion $R2/results/pid_fusion_$ref.json --alt $R2/results/pid_fusion_$c.json --steps 50 --workers 12 --out $R2/results/final/v2_offsim_${c}_vs_${ref}.json
done
