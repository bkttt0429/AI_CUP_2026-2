#!/usr/bin/env bash
# METHOD_NOTES 2.117: the opponent-side term in the whole pipeline. Opponent vectors of both official pools from 16 plain-input
# official networks as they are (scripts/pid_v2_opp_embed.py), then the pseudo-pool test on top of a fusion document
# (scripts/pid_v2_offsim_opp.py; default: the live composition v2c2_b).
#   pid_exp_opp.sh [document name]
# One GPU, 16 CPUs, from the AI_CUP_2026-1 checkout. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 doc=${1:-v2c2_b}
  local nets=(o7Fs0e o7Fs1e o7Fs2e o7Fs3e o7Fs4e o7Fs5e o7Fs6e o7Fs7e o3SLs12e o3SLs13e o3SLs14e o3SLs15e o3SLs16e o3SLs17e o3SLs18e o3SLs19e)
  if [ "$(ls /work/u2507115/pid_v2/seq_opp 2>/dev/null | grep -c '_opp_t0.npz$')" -lt $((2 * ${#nets[@]})) ]; then
    python -I $R2/scripts/pid_v2_opp_embed.py --nets "${nets[@]}" || return 1
  fi
  python -I $R2/scripts/pid_v2_offsim_opp.py --fusion $R2/results/pid_fusion_$doc.json --opp-nets "${nets[@]}" --out $R2/results/final/v2_offsim_opp_$doc.json
}
main "$@"; exit $?
