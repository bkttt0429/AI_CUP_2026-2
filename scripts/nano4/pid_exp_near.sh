#!/usr/bin/env bash
# METHOD_NOTES 2.141: the nearest-game evidence as part of the sequence component (cfg["seq_near"] = {"k": 3, "weight": 0.5},
# aicup_pid.v2.components) - the registered test of the whole pipeline.
#   documents  v2o19d_jn = the final's document v2o19d_j + the setting;  v2bgm_d2n = the recommended candidate v2bgm_d2 + the setting
#   official pseudo-pools, hold-out seeds 0 1 / 2 3 / all four, each document against its own base
#   old-pool simulation (another population of official players; questions of 1 / 2 / 3 / 5 games), 50 steps and no fine-tune
#   files with the a2k assignment and the 1-game prior, for inspection (nothing is chosen here)
# From the AI_CUP_2026-1 checkout, CPU only (48 cores). Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 J=${SLURM_JOB_ID:-local} pair s b n pids=()
  python -I -m py_compile $R2/src/aicup_pid/v2.py $R2/scripts/pid_v2_offsim.py $R2/scripts/pid_v2_oldsim.py || { echo "a changed file does not compile"; return 1; }
  python -I - <<'EOF' || return 1
import json
from pathlib import Path
R = Path("/home/u2507115/BK_AICUP_2026-2/results")
for src, dst in (("v2o19d_j", "v2o19d_jn"), ("v2bgm_d2", "v2bgm_d2n")):
    d = json.loads((R / f"pid_fusion_{src}.json").read_text())
    d["cfg"] = {**d["cfg"], "seq_near": {"k": 3, "weight": 0.5}}
    (R / f"pid_fusion_{dst}.json").write_text(json.dumps(d, indent=1) + "\n")
    print("document", dst, "=", src, "+ seq_near", d["cfg"]["seq_near"])
EOF
  for pair in "v2o19d_j v2o19d_jn" "v2bgm_d2 v2bgm_d2n"; do
    read -r b n <<< "$pair"
    for s in "0 1" "2 3" "0 1 2 3"; do
      # shellcheck disable=SC2086
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -I $R2/scripts/pid_v2_offsim.py --fusion $R2/results/pid_fusion_$b.json --alt $R2/results/pid_fusion_$n.json --steps 50 --seeds $s --workers 6 \
        --out $R2/results/final/v2_offsim_${n}_vs_${b}_s${s// /_}.json > logs/nano4/near-$J-offsim-$n-s${s// /_}.log 2>&1 &
      pids+=($!)
    done
  done
  for s in "${pids[@]}"; do wait "$s" || echo "a pseudo-pool check failed (logs/nano4/near-$J-offsim-*.log)"; done
  for pair in "v2o19d_j v2o19d_jn" "v2bgm_d2 v2bgm_d2n"; do
    read -r b n <<< "$pair"
    for s in 0_1 2_3 0_1_2_3; do
      echo "== pseudo-pools, hold-out seeds ${s//_/ }: $n vs $b  $(grep -E '^   talt vs t50' logs/nano4/near-$J-offsim-$n-s$s.log | tail -1)"
      grep -E '^(pub2|priv2): document weights|^   talt vs t50' logs/nano4/near-$J-offsim-$n-s$s.log | head -4 | cut -c1-200 | sed 's/^/        /'
    done
  done
  for pair in "v2o19d_j v2o19d_jn" "v2bgm_d2 v2bgm_d2n"; do
    read -r b n <<< "$pair"
    echo "== old-pool simulation, 50 steps: $n vs $b"
    python -I $R2/scripts/pid_v2_oldsim.py --docs $b $n --step 50 --out $R2/results/final/v2_oldsim_near_$n.json || echo "old-pool simulation failed"
    echo "== old-pool simulation, no fine-tune: $n vs $b"
    python -I $R2/scripts/pid_v2_oldsim.py --docs $b $n --step 0 --out $R2/results/final/v2_oldsim_near_${n}_t0.json || echo "old-pool simulation failed"
  done
  bash $R2/scripts/nano4/pid_v2_make_file4.sh v2bgm_d2n pid_v2bgm_d2n_a2k_p3.csv 2 0.5 0.05 1.6 pid_v2bgm_d2_a2k_p3.csv > logs/nano4/near-$J-file-v2bgm_d2n.log 2>&1 || echo "file v2bgm_d2n not built"
  bash $R2/scripts/nano4/pid_v2_make_file4.sh v2o19d_jn pid_v2o19d_jn_a2k_p3.csv 2 0.5 0.05 1.6 pid_v2o19d_j_a2k_p3.csv > logs/nano4/near-$J-file-v2o19d_jn.log 2>&1 || echo "file v2o19d_jn not built"
  for n in v2bgm_d2n v2o19d_jn; do
    echo "== file pid_${n}_a2k_p3.csv: $(grep -E 'questions with a different Top-1' logs/nano4/near-$J-file-$n.log | tail -1) | md5 $(md5sum $R2/upload/pid_${n}_a2k_p3.csv 2>/dev/null | cut -c1-32)"
  done
}
main "$@"; exit $?
