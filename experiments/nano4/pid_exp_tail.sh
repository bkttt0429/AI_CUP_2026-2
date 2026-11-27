#!/usr/bin/env bash
# METHOD_NOTES 2.117: tail terms from single reference games in the whole pipeline (scripts/pid_v2_offsim_tail.py): the same
# opponent again (S), one reference game that looks like the question game (U), the same pairing again (P), and S + U, on top
# of a fusion document (default: the live composition v2c2_b). Needs the opponent vectors of scripts/nano4/pid_exp_opp.sh.
#   pid_exp_tail.sh [document name]
# CPU work (12 workers); from the AI_CUP_2026-1 checkout. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 doc=${1:-v2c2_b}
  local nets=(o7Fs0e o7Fs1e o7Fs2e o7Fs3e o7Fs4e o7Fs5e o7Fs6e o7Fs7e o3SLs12e o3SLs13e o3SLs14e o3SLs15e o3SLs16e o3SLs17e o3SLs18e o3SLs19e)
  if [ "$(ls /work/u2507115/pid_v2/seq_opp 2>/dev/null | grep -c '_opp_t0.npz$')" -lt $((2 * ${#nets[@]})) ]; then
    python -I $R2/scripts/pid_v2_opp_embed.py --nets "${nets[@]}" || return 1
  fi
  python -I $R2/scripts/pid_v2_offsim_tail.py --fusion $R2/results/pid_fusion_$doc.json --nets "${nets[@]}" --out $R2/results/final/v2_offsim_tail_$doc.json
}
main "$@"; exit $?
