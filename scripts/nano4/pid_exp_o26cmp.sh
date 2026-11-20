#!/usr/bin/env bash
# METHOD_NOTES 2.157: the one-family comparison after the o26 chain - o26F alone (folds 1-3) against o22F alone (folds 1-4),
# each with the non-sequence components of the final's document, official pseudo-pools, four hold-out seeds; and the
# old-pool simulation of the same two documents. CPU only. From the AI_CUP_2026-1 checkout. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 J=${SLURM_JOB_ID:-local}
  for d in v2o26m_ref1 v2o26m_only; do [ -s $R2/results/pid_fusion_$d.json ] || { echo "no document $d"; return 1; }; done
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -I $R2/scripts/pid_v2_offsim.py --fusion $R2/results/pid_fusion_v2o26m_ref1.json --alt $R2/results/pid_fusion_v2o26m_only.json --steps 50 --seeds 0 1 2 3 --workers 8 \
    --out $R2/results/final/v2_offsim_v2o26m_only_vs_v2o26m_ref1_s0_1_2_3.json > logs/nano4/o26cmp-$J-offsim.log 2>&1 || echo "pseudo-pool comparison failed"
  echo "== pseudo-pools, four seeds: o26F alone (folds 1-3) vs o22F alone (folds 1-4)"
  grep -E '^(pub2|priv2|both): document weights|^   talt vs t50' logs/nano4/o26cmp-$J-offsim.log | cut -c1-240
  echo "== old-pool simulation, 50 steps"
  python -I $R2/scripts/pid_v2_oldsim.py --docs v2o26m_ref1 v2o26m_only --step 50 --out $R2/results/final/v2_oldsim_v2o26m_only.json || echo "old-pool simulation failed"
}
main "$@"; exit $?
