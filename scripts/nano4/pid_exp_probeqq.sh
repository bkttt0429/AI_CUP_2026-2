#!/usr/bin/env bash
# METHOD_NOTES 2.123, probe T: do the question games of one owner resemble each other more than his reference games
# (scripts/pid_probe_qq.py; real added pools, no answers), and what a link between a 1-game question and the multi-game
# questions would be worth (emulation on the real question games).
#   pid_exp_probeqq.sh [fusion document, default v2o13c_9] [further options of the script]
# From the AI_CUP_2026-1 checkout, CPU only. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 doc=${1:-v2o13c_9}
  shift || true
  python -I $R2/scripts/pid_probe_qq.py --doc "$doc" --out $R2/results/final/v2_probe_qq_$doc.json "$@"
}
main "$@"; exit $?
