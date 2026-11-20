#!/usr/bin/env bash
# METHOD_NOTES 2.153: cohort score normalisation per candidate (scripts/pid_probe_asnorm.py) - the registered test.
# The impostor trials of a candidate are the reference games of the other candidates of the same pool; the variant is
# chosen on hold-out seeds 0, 1, the verdict is read on seeds 2, 3, then the old-pool simulation at the chosen variant.
#   pid_exp_asnorm.sh [fusion document, default v2bgm_d2]
# From the AI_CUP_2026-1 checkout, CPU only (four worker processes). Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 doc=${1:-v2bgm_d2}
  python -I -m py_compile $R2/scripts/pid_probe_asnorm.py || { echo "the script does not compile"; return 1; }
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_probe_asnorm.py --doc "$doc" --out $R2/results/final/v2_probe_asnorm_$doc.json
}
main "$@"; exit $?
