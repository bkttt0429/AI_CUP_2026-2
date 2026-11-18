#!/usr/bin/env bash
set -euo pipefail
study=$(cd "$(dirname "$0")/../.." && pwd)
export AICUP1_ROOT=${AICUP1_ROOT:-/home/u2507115/BK_AICUP_2026-1}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -I -m pytest -q "$study/tests"
python -I "$study/scripts/pid_local_choice_data.py"
python -I "$study/scripts/pid_local_choice.py" run
