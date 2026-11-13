#!/usr/bin/env bash
# The projection probe (pid_exp_fullrows.sh, 8 GPUs) with the two inference-only checks (pid_exp_cheap.sh) running beside it
# on the same node: one job instead of two (the queue's limit is on the number of jobs). From the AI_CUP_2026-1 checkout.
set -uo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
J=${SLURM_JOB_ID:-local}
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 bash $R2/scripts/nano4/pid_exp_cheap.sh > logs/nano4/cheap-$J.log 2>&1 &
bash $R2/scripts/nano4/pid_exp_fullrows.sh 8
rc=$?
wait
echo "== cheap checks (logs/nano4/cheap-$J.log)"
grep -E "same-Top-1|^distinct|^independent|^fused a=|^alone a=|a chosen|FAILED|Traceback|Error" logs/nano4/cheap-$J.log | cut -c1-700
exit $rc
