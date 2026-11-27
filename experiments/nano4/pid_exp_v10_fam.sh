#!/usr/bin/env bash
# METHOD_NOTES 2.230: the vectors of ONE new family that the verdict and the file step need besides the untouched seeds:
#   1. pseudo-pools of the seen seeds 0-3 -> /work/u2507115/pid_v2/vtmp8/ft        (scripts/nano4/pid_exp_v10_vec.sh)
#   2. old-pool simulation               -> /work/u2507115/pid_v2/vtmp8/old       (scripts/nano4/pid_exp_bigft2.sh <family> old)
#   3. real pools                        -> data/seq_ft of the AICUP_2026-2 checkout (pid_exp_bigft2.sh <family> real)
#   pid_exp_v10_fam.sh <family>           e.g. o64F
# About 230 MB on /work and 75 MB on /home per family. From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 V=/work/u2507115/pid_v2/vtmp8 f=$1 free rc=0
  free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')
  [ "$free" -ge 4800 ] || { echo "/work has only $free MB free: not started (floor 4 GB)"; return 1; }
  free=$(df -BM --output=avail /home/u2507115 | tail -1 | tr -dc '0-9')
  [ "$free" -ge 1800 ] || { echo "/home has only $free MB free: not started (floor 1.5 GB)"; return 1; }
  bash $R2/scripts/nano4/pid_exp_v10_vec.sh $V/ft '0 1 2 3' "$f" || rc=1
  bash $R2/scripts/nano4/pid_exp_bigft2.sh "$f" old $V/old || rc=1
  bash $R2/scripts/nano4/pid_exp_bigft2.sh "$f" real || rc=1
  echo "family $f: seeds 0-3 $(ls $V/ft | grep -c "_h3s[0-3]_net-1_${f}s[0-9]*e_t50\.npz$") of 64, old pools $(ls $V/old | grep -c "_net-1_${f}s[0-9]*e_t50\.npz$") of 16, real pools $(ls $R2/data/seq_ft | grep -c "^official_.*_net-1_${f}s[0-9]*e_t50\.npz$") of 16; rc $rc; $(date '+%F %T')"
  return $rc
}
main "$@"; exit $?
