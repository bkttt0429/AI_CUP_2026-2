#!/usr/bin/env bash
# Build and lint of the report page with the player-identification checks. Run through run.sbatch only.
set -uo pipefail
main() {
  local NODE; NODE=$(ls -t /home/u2507115/.vscode-server/cli/servers/*/server/node 2>/dev/null | head -n 1)
  [ -n "$NODE" ] && export PATH="$(dirname "$NODE"):$PATH"; echo "node: $(command -v node || echo none)"
  local G=/home/u2507115/.cache/pidrpt_wt/reports/go_story_pid P=/home/u2507115/.cache/pidrpt_wt/reports/pid_report rc=0
  python3 $P/check_numbers.py --selftest || { echo "selftest FAILED"; rc=1; }
  python3 $G/assemble.py || rc=1
  python3 $G/check.py || rc=1
  local files="$G/parts/c08_pid.html"
  [ -s $G/parts/c09_pidv2.html ] && files="$files $G/parts/c09_pidv2.html"
  [ -s $G/parts/a08b_pidv2.js ] && files="$files $G/parts/a08b_pidv2.js"
  # shellcheck disable=SC2086
  python3 $P/check_numbers.py --cards $files || rc=1
  [ $rc = 0 ] && echo "PID_BUILD OK" || echo "PID_BUILD FAILED"
  return $rc
}
main "$@"; exit $?
