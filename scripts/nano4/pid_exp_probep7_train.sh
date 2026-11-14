#!/usr/bin/env bash
# METHOD_NOTES 2.115, probe P7, steps 2-4 (after scripts/nano4/pid_exp_probep7_x.sh): do value scalars add to the second
# feature set? The recipe of probe B / P6 (folds 1 and 2 left out, symmetry + ply masking, 10,000 steps); the first 76 input
# dimensions = 64 joint-projection features + 4 policy scalars + 8 value scalars:
#   p7 / p7b / p7c   three seeds                      p7o   the 76 features alone (every other input dimension at its mean)
# then fold vectors, the simulated-pool comparisons against probe B's q2 (results/final/v2_sim14p7.json) and against P6's
# 68-dimension network p6s (v2_sim14p7b.json), and the row-embedding fusion (v2_probe_p7_fuse.json).
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 D=/work/u2507115/pid_v2/probep S=/work/u2507115/pid_v2/seqnet J=${SLURM_JOB_ID:-local}
  local C="--steps 10000 --horizon 30000 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --sup-dims 0 76" G p pids=() rc=0
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ "${#G[@]}" -ge 8 ] || { echo "need 8 GPUs, have ${#G[@]}"; return 1; }
  python -I - <<'PYEOF'
import glob, numpy as np
D = "/work/u2507115/pid_v2/probep"
worst, n = 0.0, 0
for f in sorted(glob.glob(f"{D}/ply_jv/sup_s00[0-7]_p0000.npz")):
    a, b = np.load(f), np.load(f.replace("ply_jv", "ply_js"))
    assert (a["keys"] == b["keys"]).all() and (a["t"] == b["t"]).all(), f
    d = np.abs(a["z"][:, :68].astype(np.float32) - b["z"].astype(np.float32))
    worst, n = max(worst, float(d.max())), n + len(d)
print(f"the first 68 dimensions against the stored second feature set: {n} plies of 8 parts, largest absolute difference {worst:.4f}")
PYEOF
  bash $R2/scripts/nano4/pid_seq_train_fork.sh "--sup-ply $D/ply_jv" \
    "1 p7 $C" "2 p7 $C" "1 p7b $C --seed 1" "2 p7b $C --seed 1" "1 p7c $C --seed 2" "2 p7c $C --seed 2" "1 p7o $C --sup-only" "2 p7o $C --sup-only" || { echo "a trainer FAILED"; rc=1; }
  emb() {  # <GPU> <tag> [more options]
    CUDA_VISIBLE_DEVICES=$1 python -I $R2/scripts/pid_v2_seq_embed.py --stage folds --folds 1 2 --ckpt-dir $S --ckpt-suffix "_$2" --name "_n$2" --out $R2/data/seq --sup-ply $D/ply_jv --sup-dims 0 76 "${@:3}" > logs/nano4/p7emb_$2-$J.log 2>&1
  }
  emb ${G[0]} p7e & pids+=($!)
  emb ${G[1]} p7be & pids+=($!)
  emb ${G[2]} p7ce & pids+=($!)
  emb ${G[3]} p7oe --sup-only & pids+=($!)
  for p in "${pids[@]}"; do wait "$p" || { echo "a fold-vector run FAILED"; rc=1; }; done
  local names="q2e=_nq2e q0e=_nq0e p6e=_np6e p6be=_np6be p6se=_np6se p6sbe=_np6sbe p7e=_np7e p7be=_np7be p7ce=_np7ce p7oe=_np7oe"
  # shellcheck disable=SC2086
  python -I $R2/scripts/pid_v2_sim14.py --names $names --ref q2e --folds 1 2 --out $R2/results/final/v2_sim14p7.json || rc=1
  # shellcheck disable=SC2086
  python -I $R2/scripts/pid_v2_sim14.py --names $names --ref p6se --folds 1 2 --out $R2/results/final/v2_sim14p7b.json | grep -E "vs p6se" || rc=1
  python -I $R2/scripts/pid_probe_b_fuse.py --tags p6se q2e p6sbe p7e p7be p7ce p7oe --ref p6se --out $R2/results/final/v2_probe_p7_fuse.json || rc=1
  return $rc
}
main "$@"; exit $?
