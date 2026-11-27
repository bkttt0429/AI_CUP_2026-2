#!/usr/bin/env bash
# The two analysis probes of METHOD_NOTES 2.56 side by side on two GPUs of one job (the job is on 16gpus only because the
# 8gpus queue was full): where in a game the identity information sits, and move-level matching.
set -uo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
IFS=, read -r -a G <<< "$CUDA_VISIBLE_DEVICES"
CUDA_VISIBLE_DEVICES=${G[0]} python -I $R2/scripts/pid_v2_where.py --out $R2/results/final/v2_where_x2e.json > logs/nano4/probe_where-${SLURM_JOB_ID}.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=${G[1]} python -I $R2/scripts/pid_v2_latematch.py --out $R2/results/final/v2_latematch_x2e.json > logs/nano4/probe_latematch-${SLURM_JOB_ID}.log 2>&1 &
p2=$!
rc=0
wait $p1 || rc=1
echo "== where"; grep -vE "^fold" logs/nano4/probe_where-${SLURM_JOB_ID}.log | tail -30
wait $p2 || rc=1
echo "== move-level matching"; tail -30 logs/nano4/probe_latematch-${SLURM_JOB_ID}.log
exit $rc
