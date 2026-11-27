#!/usr/bin/env bash
# METHOD_NOTES 2.168: ONE combined candidate of the two small items that add information about the CANDIDATES - the cohort
# normalisation (notes 2.153; setting N=300, g=0.5 as chosen there) and the linked candidates' training games (notes 2.167;
# sixteen plain networks, impostor calibration) - scripts/pid_probe_link3.py --znorm --cal.
#   pid_exp_link5.sh design     hold-out seeds 0-3 (all of them already looked at: design only)
#   pid_exp_link5.sh verdict    (tau, lam, w) chosen on seeds 0-3, the verdict read ONCE on the untouched seeds 4, 5
# CPU only. From the AI_CUP_2026-1 checkout. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 W=/work/u2507115/pid_v2/pair_eval mode=${1:-design} P=${PROTOS:-link_protos16.npz}
  python -I -m py_compile $R2/scripts/pid_probe_link3.py || { echo "the script does not compile"; return 1; }
  [ -s $W/$P ] || { echo "no prototypes $W/$P"; return 1; }
  local C="--cal --znorm 300 0.5 --min-gain 0.002 --doc v2bgm_d2 --protos $W/$P"
  case "$mode" in
    design) OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_probe_link3.py $C --out $R2/results/final/v2_probe_link5_design.json ;;
    verdict) OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_probe_link3.py $C --seeds 0 1 2 3 4 5 --select-seeds 0 1 2 3 --procs 6 --out $R2/results/final/v2_probe_link5_verdict.json ;;
    *) echo "usage: pid_exp_link5.sh design|verdict"; return 2 ;;
  esac
}
main "$@"; exit $?
