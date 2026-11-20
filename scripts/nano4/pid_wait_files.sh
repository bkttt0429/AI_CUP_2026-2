#!/usr/bin/env bash
# Waits until <count> files whose names match <regex> exist in /work/u2507115/pid_v2/seqnet (and have not changed for a
# minute), then runs the command. For readings of SNAPSHOT checkpoints that a still-running training writes (METHOD_NOTES
# 2.217): the queue's "needs" know tasks, not files.
#   pid_wait_files.sh <count> <regex> <timeout in minutes> <command> [<arguments> ...]
set -uo pipefail
main() {
  local n=$1 re=$2 lim=$3 SN=/work/u2507115/pid_v2/seqnet t0=$SECONDS a b
  shift 3
  while :; do
    a=$(ls -l --time-style=+%s $SN | awk -v re="$re" '$NF ~ re {print $NF, $5, $6}')
    if [ "$(printf '%s\n' "$a" | grep -c .)" -ge "$n" ]; then
      sleep 60
      b=$(ls -l --time-style=+%s $SN | awk -v re="$re" '$NF ~ re {print $NF, $5, $6}')
      [ "$a" = "$b" ] && break
    fi
    [ $(( (SECONDS - t0) / 60 )) -lt "$lim" ] || { echo "timeout: $(printf '%s\n' "$a" | grep -c .) of $n files after $lim min"; return 1; }
    sleep 30
  done
  echo "$n files present after $(( (SECONDS - t0) / 60 )) min ($(date +%T)): $*"
  "$@"
}
main "$@"; exit $?
