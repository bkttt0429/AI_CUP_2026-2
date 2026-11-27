#!/usr/bin/env bash
# Leave-games-out check on the official candidates for the two halves of a linked family (METHOD_NOTES 2.96):
#   pid_exp_split.sh <linked family> <name a> <seeds a, e.g. "0|1|2|3"> <name b> <seeds b>
# single networks and ensembles of each half, next to o3S (12 nets) and o3SL (8 nets, 40,000 steps). CPU only.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 S=/home/u2507115/BK_AICUP_2026-2/data/seq_ft fam=$1 a=$2 sa=$3 b=$4 sb=$5 g l k groups=()
  for g in o3S o3SL; do
    l=""; for k in $(seq 0 23); do [ -s $S/official_pub2_net-1_${g}s${k}e_t0.npz ] && [ -s $S/official_priv2_net-1_${g}s${k}e_t0.npz ] && l="$l|$k"; done
    groups+=("$g=$S/official_{pool}_net-1_${g}s{${l#|}}e_t0.npz")
  done
  groups+=("$a=$S/official_{pool}_net-1_${fam}s{$sa}e_t0.npz" "$b=$S/official_{pool}_net-1_${fam}s{$sb}e_t0.npz")
  python -I $R2/scripts/pid_v2_offcand2.py --groups "${groups[@]}" --comps "ref=o3S" "$a=$a" "$b=$b" "both=$a+$b" "o3SL=o3SL" "o3S_$a=o3S+$a" "o3S_$b=o3S+$b" --ref ref \
    --out $R2/results/final/v2_offcand2_split_${fam}.json
}
main "$@"; exit $?
