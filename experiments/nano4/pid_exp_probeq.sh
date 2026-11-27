#!/usr/bin/env bash
# METHOD_NOTES 2.117: probe Q, the opponent's side of the official candidates' reference games (scripts/pid_probe_opp.py).
# Plain-input official networks as they are (no pool fine-tune): the 8 long-trained mask-recipe networks and the 8 of 40k steps.
# One GPU, a few minutes; from the AI_CUP_2026-1 checkout. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 out=${1:-v2_probe_opp}
  shift || true
  local nets=("$@")
  if [ ${#nets[@]} -eq 0 ]; then
    nets=(o7Fs0e o7Fs1e o7Fs2e o7Fs3e o7Fs4e o7Fs5e o7Fs6e o7Fs7e o3SLs12e o3SLs13e o3SLs14e o3SLs15e o3SLs16e o3SLs17e o3SLs18e o3SLs19e)
  fi
  python -I $R2/scripts/pid_probe_opp.py --nets "${nets[@]}" --out $R2/results/final/$out.json
}
main "$@"; exit $?
