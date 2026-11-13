#!/usr/bin/env bash
# METHOD_NOTES 2.69, probe A: what does the random projection (384 -> 64 per trunk vector) lose at the level of per-game rows?
#   1. the games: the simulated pools of folds 1 and 2, and up to 16 games of every player of folds 3 and 4 (never fold 0)
#   2. their per-game rows with the UNPROJECTED statistics (scripts/pid_trunk_extract.py --full), one shard per GPU
#   3. the same two learners on every representation (scripts/pid_probe_fullrows.py --stage eval)
# One job with 8 GPUs, from the AI_CUP_2026-1 checkout.   pid_exp_fullrows.sh [GPUs, default 8]
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
D=/work/u2507115/pid_v2/fullrows
N=${1:-8}
mkdir -p $D
[ -s $D/keys.txt ] || python -I $R2/scripts/pid_probe_fullrows.py --stage keys --out $D/keys.txt
bash scripts/nano4/shards.sh "$N" python -I $R2/scripts/pid_trunk_extract.py --keys $D/keys.txt --full --out $D/rows --workers 10
python -I $R2/scripts/pid_probe_fullrows.py --stage eval --rows $D/rows --out $R2/results/final/v2_probe_fullrows.json
