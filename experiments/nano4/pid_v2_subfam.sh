#!/usr/bin/env bash
# Which kind of supervised-input network is best (METHOD_NOTES 2.110): the leave-games-out check on the official candidates
# (scripts/pid_v2_offcand2.py, vectors before the pool fine-tune) with the o12 snapshot family split by what the network reads:
#   A = seeds 0-7 (pooled half replaced)   B = seeds 8-11 (played-point half replaced)   C = seeds 12-15 (supervised only)
# next to the plain symmetry family o3S and the long-trained o7F. No GPU work. From the AI_CUP_2026-1 checkout.
#   pid_v2_subfam.sh <family with 16 networks, e.g. o12K> <result name>
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 S=/home/u2507115/BK_AICUP_2026-2/data/seq_ft f=$1 out=$2
  python -I $R2/scripts/pid_v2_offcand2.py \
    --groups "A=$S/official_{pool}_net-1_${f}s{0|1|2|3|4|5|6|7}e_t0.npz" "B=$S/official_{pool}_net-1_${f}s{8|9|10|11}e_t0.npz" "C=$S/official_{pool}_net-1_${f}s{12|13|14|15}e_t0.npz" \
             "o3S=$S/official_{pool}_net-1_o3Ss{0|1|2|3|4|5|6|7|8|9|10|11}e_t0.npz" "o7F=$S/official_{pool}_net-1_o7Fs{0|1|2|3|4|5|6|7}e_t0.npz" \
    --comps "ref=o3S" "onlyA=A" "onlyB=B" "onlyC=C" "AB=A+B" "AC=A+C" "ABC=A+B+C" "o7=o7F" "o7ABC=o7F+A+B+C" \
    --ref ref --out $R2/results/final/v2_offcand2_$out.json
}
main "$@"; exit $?
