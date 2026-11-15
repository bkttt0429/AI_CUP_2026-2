#!/usr/bin/env bash
# A quick screen of compositions (scripts/pid_v2_offcand2.py): the sequence component alone, vectors before the pool
# fine-tune, the official candidates' held-out games (n = 1 and n = 3), both added pools, bootstrap over candidates.
# Minutes on CPU; no file is built and nothing is chosen here - it only says which compositions are worth the full chain.
#   pid_exp_screen.sh <out name> <reference composition> <name>=<composition> ...
# LIVE / BASE in a composition expand as in pid_chain_job4.sh (LIVE = the old final's composition o13F*9+o12F*9+BASE).
# Groups: every family with official vectors (as pid_v2_newnets3.sh), and two halves of o19F by sampling:
#   o19L = its four networks drawn by games linearly (seeds 0 1 2 6), o19Q = by the square root (seeds 3 4 5 7).
# From the AI_CUP_2026-1 checkout, CPU only. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 S=/home/u2507115/BK_AICUP_2026-2/data/seq_ft out=$1 ref=$2 g l k c comps=()
  shift 2
  local base="o7F*9+o9D*3+o3SL*3+o3S*3+o4S*3+o3BS*3+py*3+o1+o2+px2+pb+px+m101x"
  local live="o13F*9+o12F*9+$base"
  local SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
  list() {
    local o=() k
    for k in $SEEDS; do [ -s $S/official_pub2_net-1_${1}s${k}e_t0.npz ] && [ -s $S/official_priv2_net-1_${1}s${k}e_t0.npz ] && o+=("$k"); done
    (IFS='|'; echo "${o[*]}")
  }
  local groups=("m90=$S/official_{pool}_net{-1|1|2|3|4}_m90s{0|1|2}e_t0.npz" "m101x=$S/official_{pool}_net{-1|1|2|3|4}_m101x_t0.npz"
                "px=$S/official_{pool}_net{1|2}_xe_t0.npz,$S/official_{pool}_net1_xue_t0.npz" "px2=$S/official_{pool}_net{1|2}_x2e_t0.npz"
                "py=$S/official_{pool}_net{1|2}_yse_t0.npz" "pb=$S/official_{pool}_net{1|2}_ybe_t0.npz")
  for g in $(ls $S | sed -nE 's/^official_pub2_net-1_(o[0-9][A-Za-z0-9]*)s[0-9]+e_t0\.npz$/\1/p' | sort -u); do
    l=$(list $g)
    if [[ "$l" == *"|"* ]]; then groups+=("$g=$S/official_{pool}_net-1_${g}s{$l}e_t0.npz"); elif [ -n "$l" ]; then groups+=("$g=$S/official_{pool}_net-1_${g}s${l}e_t0.npz"); fi
  done
  if [ -s $S/official_pub2_net-1_o19Fs7e_t0.npz ]; then
    groups+=("o19L=$S/official_{pool}_net-1_o19Fs{0|1|2|6}e_t0.npz" "o19Q=$S/official_{pool}_net-1_o19Fs{3|4|5|7}e_t0.npz")
  fi
  ref=${ref//LIVE/$live}; ref=${ref//BASE/$base}
  for c in "$@"; do c=${c//LIVE/$live}; comps+=("${c//BASE/$base}"); done
  python -I $R2/scripts/pid_v2_offcand2.py --groups "${groups[@]}" --comps "ref=$ref" "${comps[@]}" --ref ref --out $R2/results/final/v2_offcand2_screen_$out.json
}
main "$@"; exit $?
