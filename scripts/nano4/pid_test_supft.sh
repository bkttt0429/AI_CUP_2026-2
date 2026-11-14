#!/usr/bin/env bash
# End-to-end test of the pool fine-tune for a network trained on supervised input (METHOD_NOTES 2.100): a copy of a probe-B
# checkpoint (ckpt_1_q2e.pt, trained with --sup-dims 0 64 before checkpoints recorded it) gets the record, then the three
# modes used by the chains run once each on it (real pool, pseudo-pool, old-pool simulation); the test files are removed.
# One GPU, from the AI_CUP_2026-1 checkout.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 S=/work/u2507115/pid_v2/seqnet T=/work/u2507115/pid_v2/supft_test rc=0
  mkdir -p $T
  python -I - <<'PYEOF'
import torch
S = "/work/u2507115/pid_v2/seqnet"
st = torch.load(f"{S}/ckpt_1_q2e.pt", map_location="cpu", weights_only=False)
st["sup"] = {"dims": [0, 64], "only": False}
torch.save(st, f"{S}/ckpt_1_q2Te.pt")
print("test checkpoint written; keys", sorted(st.keys()))
PYEOF
  local common="--net 1 --ckpt-dir $S --ckpt-suffix _q2Te --tag _q2Te --steps 0 50 --horizon 1000 --out $T"
  # shellcheck disable=SC2086
  python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --pool pub2 $common || rc=1
  # shellcheck disable=SC2086
  python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --pool priv2 --holdout 3 --holdout-seed 0 $common || rc=1
  # shellcheck disable=SC2086
  python -I $R2/scripts/pid_v2_seq_finetune.py --mode oldsim --side pub --holdout-seed 0 $common || rc=1
  ls -la $T | awk '{print $5, $9}' | tail -8
  python -I - <<'PYEOF'
import glob, numpy as np
for f in sorted(glob.glob("/work/u2507115/pid_v2/supft_test/*_t50.npz")):
    z = np.load(f); e = z["emb"].astype(np.float32)
    print(f.split("/")[-1], e.shape, "norms", float(np.linalg.norm(e, axis=1).mean()), "finite", bool(np.isfinite(e).all()))
PYEOF
  rm -rf $T $S/ckpt_1_q2Te.pt
  echo "fine-tune test for supervised-input networks: $([ $rc = 0 ] && echo PASSED || echo FAILED)"
  return $rc
}
main "$@"; exit $?
