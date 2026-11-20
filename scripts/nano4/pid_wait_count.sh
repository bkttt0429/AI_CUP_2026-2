#!/usr/bin/env bash
# Waits until a folder holds at least <count> names matching <regex> (and that list has not changed for a minute), then
# runs the command. The general form of pid_wait_files.sh (which looks at the checkpoint folder only): for steps that
# depend on files written by a task of ANOTHER queue (the queue's "needs" know the tasks of their own queue only).
#   pid_wait_count.sh <folder> <count> <regex> <timeout in minutes> <command> [<arguments> ...]
set -uo pipefail
main() {
  local dir=$1 n=$2 re=$3 lim=$4 t0=$SECONDS a b
  shift 4
  while :; do
    a=$(ls -l --time-style=+%s "$dir" 2> /dev/null | awk -v re="$re" '$NF ~ re {print $NF, $5, $6}')
    if [ "$(printf '%s\n' "$a" | grep -c .)" -ge "$n" ]; then
      sleep 60
      b=$(ls -l --time-style=+%s "$dir" 2> /dev/null | awk -v re="$re" '$NF ~ re {print $NF, $5, $6}')
      [ "$a" = "$b" ] && break
    fi
    [ $(( (SECONDS - t0) / 60 )) -lt "$lim" ] || { echo "timeout: $(printf '%s\n' "$a" | grep -c .) of $n names in $dir after $lim min"; return 1; }
    sleep 30
  done
  echo "$n names present in $dir after $(( (SECONDS - t0) / 60 )) min ($(date +%T)): $*"
  "$@"
}
main "$@"; exit $?
