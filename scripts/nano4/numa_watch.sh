#!/usr/bin/env bash
# METHOD_NOTES 2.200: a watchdog for the "slow trainer" fault found on 2026-11-17 (o36SMs3 on node 089, o41Ss2 / o41Ss3 on
# node 088): a trainer whose main thread is scheduled on the "wrong" NUMA side of a node spends more than half of its
# time in the kernel and runs at half speed or worse; pinning all its threads to the other side (taskset -acp) restores
# the speed within a minute. Which side is the bad one differs per node and the scheduler can move a thread later, so
# this loop looks at every trainer of a job every INTERVAL seconds and re-pins the ones that show the fault.
#   numa_watch.sh <job id> [interval, default 300]          runs on the LOGIN node (ssh to the job's nodes; no compute:
#                                                           a few /proc reads per trainer), ends when the job has ended
# A trainer is re-pinned when its main thread used more kernel than user time in a 5-second sample (and at least 1 s of
# kernel time); at most MAXPIN (3) times per trainer and not twice within 15 minutes. Everything is logged to
# logs/nano4/numa_watch-<job>.log. From the AI_CUP_2026-1 checkout. Do not edit while it runs.
set -uo pipefail
main() {
  local job=${1:?job id} every=${2:-300} log=logs/nano4/numa_watch-$1.log state n line now last cnt key pid side
  state=$(mktemp "${TMPDIR:-/tmp}/numa_watch_XXXXXX")
  echo "$(date '+%F %T') watch of job $job every $every s (state $state)" >> "$log"
  while [ "$(squeue -j "$job" -h -o %T 2> /dev/null)" = RUNNING ]; do
    for n in $(scontrol show hostnames "$(squeue -j "$job" -h -o %N)"); do
      # remote: one line per trainer "pid user sys cpu"
      timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no "$n" '
        par=$(pgrep -o -f "pid_seq_train_w[0-9a-z]*.py --fork") || exit 0
        for p in $(pgrep -P "$par"); do
          [ -r /proc/$p/task/$p/stat ] || continue
          grep -q python /proc/$p/comm 2> /dev/null || continue
          set -- $(awk "{print \$14, \$15}" /proc/$p/task/$p/stat); echo "$p $1 $2"
        done > /tmp/nw_$$
        sleep 5
        while read p u s; do
          [ -r /proc/$p/task/$p/stat ] || continue
          set -- $(awk "{print \$14, \$15, \$39}" /proc/$p/task/$p/stat); echo "$p $(( $1 - u )) $(( $2 - s )) $3"
        done < /tmp/nw_$$
        rm -f /tmp/nw_$$' 2> /dev/null | while read -r pid u s cpu; do
        [ -n "${cpu:-}" ] || continue
        if [ "$s" -ge 100 ] && [ "$s" -gt "$u" ]; then
          key="$n:$pid"; now=$(date +%s)
          last=$(awk -v k="$key" '$1 == k {t = $2} END {print t + 0}' "$state"); cnt=$(grep -c "^$key " "$state")
          if [ "$cnt" -ge "${MAXPIN:-3}" ] || [ $(( now - last )) -lt 900 ]; then
            echo "$(date '+%F %T') $n pid $pid: kernel $s / user $u ticks on cpu $cpu - not re-pinned (pinned $cnt times, last $(( now - last )) s ago)" >> "$log"
            continue
          fi
          if [ "$cpu" -lt 56 ]; then side=56-103; else side=0-47; fi
          if timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 "$n" "taskset -acp $side $pid > /dev/null"; then
            echo "$key $now" >> "$state"
            echo "$(date '+%F %T') $n pid $pid: kernel $s / user $u ticks on cpu $cpu -> pinned to CPUs $side" >> "$log"
          else
            echo "$(date '+%F %T') $n pid $pid: kernel $s / user $u ticks on cpu $cpu - taskset FAILED" >> "$log"
          fi
        fi
      done
    done
    sleep "$every"
  done
  echo "$(date '+%F %T') job $job is no longer running: the watch ends" >> "$log"
  rm -f "$state"
}
main "$@"; exit $?
