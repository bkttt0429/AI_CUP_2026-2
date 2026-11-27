#!/usr/bin/env bash
# Two inference-only checks on the official pseudo-pools, one job (1 GPU, 12 CPUs), from the AI_CUP_2026-1 checkout:
#   METHOD_NOTES 2.70  one-game questions: distinct owners? what an assignment among them is worth (pid_v2_assign1.py)
#   METHOD_NOTES 2.71  colour-matched candidate prototypes (pid_v2_offsim_colour.py)
set -uo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
python -I $R2/scripts/pid_v2_offsim_colour.py --out $R2/results/final/v2_offsim_colour.json || echo "colour check FAILED"
python -I $R2/scripts/pid_v2_assign1.py --out $R2/results/final/v2_assign1.json || echo "assignment check FAILED"
