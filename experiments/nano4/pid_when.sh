#!/usr/bin/env bash
# Run one command when enough files exist (METHOD_NOTES 2.106): the session that watches the training jobs can be interrupted,
# so the "when these checkpoints are there, submit their chain" steps are left with detached copies of this script on the
# login node. It only lists files and runs the given (submitting / linking) command once; nothing is computed here.
#   setsid nohup bash pid_when.sh <tag> <give up at, HH:MM today> <count> <directory> <egrep pattern> -- <command...> &
# Waits until at least <count> files of <directory> match the pattern and the newest of them is at least 90 s old (written
# completely), then runs the command from the AI_CUP_2026-1 checkout. Log: ~/BK_AICUP_2026-1/logs/nano4/when_<tag>.log
set -uo pipefail
main() {
  local tag=$1 until=$2 count=$3 dir=$4 pat=$5; shift 5
  [ "$1" = "--" ] && shift
  local log=/home/u2507115/BK_AICUP_2026-1/logs/nano4/when_$tag.log end now n newest
  end=$(date -d "today $until" +%s)
  cd /home/u2507115/BK_AICUP_2026-1 || return 1
  echo "$(date +%H:%M:%S) waiting for $count files matching '$pat' in $dir (until $until): $*" >> "$log"
  while :; do
    now=$(date +%s)
    if [ "$now" -ge "$end" ]; then echo "$(date +%H:%M:%S) gave up" >> "$log"; return 1; fi
    n=$(ls "$dir" 2>/dev/null | grep -cE "$pat")
    if [ "$n" -ge "$count" ]; then
      newest=$(ls "$dir" | grep -E "$pat" | while read -r f; do stat -c %Y "$dir/$f"; done | sort -n | tail -n 1)
      if [ $((now - newest)) -ge 90 ]; then break; fi
    fi
    sleep 30
  done
  echo "$(date +%H:%M:%S) $n files: running the command" >> "$log"
  "$@" >> "$log" 2>&1
  echo "$(date +%H:%M:%S) command exit $?" >> "$log"
}
main "$@"; exit $?
