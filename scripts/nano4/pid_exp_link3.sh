#!/usr/bin/env bash
# METHOD_NOTES 2.167: do a linked candidate's training games help? (scripts/pid_probe_link3.py) - the registered test on the
# official pseudo-pools. CPU only (four worker processes). From the AI_CUP_2026-1 checkout. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 doc=${1:-v2bgm_d2}
  python -I -m py_compile $R2/scripts/pid_probe_link3.py || { echo "the script does not compile"; return 1; }
  [ -s /work/u2507115/pid_v2/pair_eval/link_protos.npz ] || { echo "no training prototypes"; return 1; }
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_probe_link3.py --doc "$doc" --out $R2/results/final/v2_probe_link3_$doc.json
}
main "$@"; exit $?
