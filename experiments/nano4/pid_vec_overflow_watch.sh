#!/usr/bin/env bash
# METHOD_NOTES 2.202: moves the simulation vectors of o41F / o42F / o43F to /work (scripts/nano4/pid_vec_overflow.sh) as
# soon as the family's vector task of the queue job has ended, so that /home keeps room for the next family's vectors
# and for the verdict task. Reads the driver's log for "done vec_o41 / vec_o42 / vec_o43b".
#   pid_vec_overflow_watch.sh <job id>        on the LOGIN node (file management only); ends when the three families
#                                             are moved or the job has ended. Log: logs/nano4/vec_overflow_watch-<job>.log
# From the AI_CUP_2026-1 checkout. Do not edit while it runs.
set -uo pipefail
main() {
  local job=${1:?job id} R2=/home/u2507115/BK_AICUP_2026-2 L=logs/nano4/pidq1-$1.out log=logs/nano4/vec_overflow_watch-$1.log t fam seen=""
  echo "$(date '+%F %T') watch of job $job" >> "$log"
  while :; do
    for t in vec_o41:o41F vec_o42:o42F vec_o43b:o43F; do
      fam=${t#*:}
      case " $seen " in *" $fam "*) continue ;; esac
      if grep -q " done ${t%%:*} on " "$L" 2> /dev/null; then
        seen="$seen $fam"
        echo "$(date '+%F %T') ${t%%:*} has ended: moving the simulation vectors of $fam" >> "$log"
        bash $R2/scripts/nano4/pid_vec_overflow.sh "$fam" >> "$log" 2>&1 || echo "$(date '+%F %T') the move of $fam did NOT complete (see above); not tried again" >> "$log"
      fi
    done
    # shellcheck disable=SC2086
    [ "$(echo $seen | wc -w)" -ge 3 ] && { echo "$(date '+%F %T') the three families are handled: the watch ends" >> "$log"; break; }
    [ "$(squeue -j "$job" -h -o %T 2> /dev/null)" = RUNNING ] || { echo "$(date '+%F %T') job $job is no longer running: the watch ends" >> "$log"; break; }
    sleep 30
  done
}
main "$@"; exit $?
