#!/usr/bin/env bash
# METHOD_NOTES 2.168: the linked candidates' training games with the impostor calibration (scripts/pid_probe_link3.py --cal).
#   pid_exp_link4.sh design     hold-out seeds 0-3 only (0, 1 choose; 2, 3 were already looked at in notes 2.167: design run)
#   pid_exp_link4.sh verdict    seeds 0-5: the setting is chosen on seeds 0-3, the verdict is read on the untouched seeds 4, 5
# CPU only. From the AI_CUP_2026-1 checkout. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 mode=${1:-design}
  python -I -m py_compile $R2/scripts/pid_probe_link3.py || { echo "the script does not compile"; return 1; }
  case "$mode" in
    design) OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_probe_link3.py --cal --doc v2bgm_d2 --out $R2/results/final/v2_probe_link4_design.json ;;
    verdict) OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_probe_link3.py --cal --doc v2bgm_d2 --seeds 0 1 2 3 4 5 --select-seeds 0 1 2 3 --procs 6 --out $R2/results/final/v2_probe_link4_verdict.json ;;
    *) echo "usage: pid_exp_link4.sh design|verdict"; return 2 ;;
  esac
}
main "$@"; exit $?
