#!/usr/bin/env bash
# METHOD_NOTES 2.168: training prototypes under SIXTEEN plain-input networks (o22F and o7F, the two plain families of the
# final's composition) for the linkage add-on - four networks made it too noisy to carry weight.
# From the AI_CUP_2026-1 checkout, one GPU, 48 cores for the loader. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 W=/work/u2507115/pid_v2/pair_eval
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_probe_link.py --protos $W/link_protos16.npz --out $R2/results/final/v2_probe_link16.json \
    --nets o22Ss0 o22Ss1 o22Ss2 o22Ss3 o22SMs0 o22SMs1 o22SMs2 o22SMs3 o7Ss0 o7Ss1k30 o7Ss2 o7Ss3 o7SMs0k30 o7SMs1 o7SMs2 o7SMs3k30 \
    --alias o22Fs0 o22Fs1 o22Fs2 o22Fs3 o22Fs4 o22Fs5 o22Fs6 o22Fs7 o7Fs0 o7Fs1 o7Fs2 o7Fs3 o7Fs4 o7Fs5 o7Fs6 o7Fs7
}
main "$@"; exit $?
