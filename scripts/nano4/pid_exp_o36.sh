#!/usr/bin/env bash
# METHOD_NOTES 2.180: the window input with the EXACT board (scripts/pid_winx.py, trainer scripts/pid_seq_train_wx.py
# --shape-win N --shape-exact). Smoke test before any short run:
#   pid_exp_o36.sh smoke     1. pid_winx.py's self-test (removal plies and the window builder against a brute-force replay)
#                            2. two smoke trainings forked from one parent (the removal plies come from the parent's pool into
#                               shared memory): one with the exact board, one with the inexact window as before
#                            3. the pool loader on both checkpoints: the exact network needs packed colours and refuses plain
#                               ones; an old checkpoint loads as before
# From the AI_CUP_2026-1 checkout, 2 GPUs or more. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 S=/work/u2507115/pid_v2/seqnet mode=${1:-smoke} J=${SLURM_JOB_ID:-local} cards spec D
  IFS=, read -r -a cards <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ "$mode" = smoke ] || { echo "usage: pid_exp_o36.sh smoke"; return 2; }
  python -I -m py_compile $R2/scripts/pid_win.py $R2/scripts/pid_winx.py $R2/scripts/pid_seq_train_wx.py $R2/scripts/pid_v2_seq_embed.py $R2/scripts/pid_v2_seq_finetune.py || { echo "a script does not compile"; return 1; }
  D=/dev/shm/pid_o36smoke_$J
  mkdir -p $D || return 1
  python -I $R2/scripts/pid_winx.py || { echo "the self-test of pid_winx.py FAILED"; find $D -delete; return 1; }
  spec=$(mktemp "${TMPDIR:-/tmp}/o36_spec_XXXXXX")
  printf '%s\n' "${cards[0]} logs/nano4/o36smoke-$J-x.log --net -1 --tag wxsmoke --smoke --no-state --sym-aug --mask 0.3 --ema-only --shape-win 9 --shape-exact --out $D" \
                "${cards[1]} logs/nano4/o36smoke-$J-w.log --net -1 --tag wwsmoke --smoke --no-state --sym-aug --mask 0.3 --ema-only --shape-win 9 --out $D" > "$spec"
  REM_WORKERS=8 OMP_NUM_THREADS=4 python -I $R2/scripts/pid_seq_train_wx.py --fork "$spec" --smoke || { echo "a smoke training FAILED"; tail -20 logs/nano4/o36smoke-$J-x.log; rm -f "$spec"; find $D -delete; return 1; }
  rm -f "$spec"
  for f in x w; do grep -E "window input|exact board|removal plies|step +[0-9]+/|held-out" logs/nano4/o36smoke-$J-$f.log | cut -c1-200 | tail -6; done
  ls -la $D | cut -c1-120
  CUDA_VISIBLE_DEVICES=${cards[0]} python -I - $R2 $D $S <<'PY' || { echo "the loader check FAILED"; find $D -delete; return 1; }
import sys
from pathlib import Path
import numpy as np
import torch
R2, D, S = sys.argv[1:4]
sys.path[:0] = [R2 + "/scripts", R2 + "/src"]
import pid_v2_seq_embed as E
import pid_v2_seq_finetune as FT
import pid_winx
rng = np.random.default_rng(0)
n, T = 300, FT.m88.T_MAX
A = {"GL": np.zeros((n, T, 128), np.float16), "POS": np.full((n, T), 362, np.int16), "MOV": np.zeros((n, T), np.int8), "LEN": np.full(n, 200, np.int32), "SUBJ": np.zeros(n, np.int8)}
for i in range(n):  # random games in a corner: many captures
    A["POS"][i, :200] = rng.integers(0, 7, 200) * 19 + rng.integers(0, 7, 200); A["MOV"][i, :200] = np.arange(200) % 2
idx = np.arange(n)
for name, ck in (("exact", sorted(Path(D).glob("ckpt_-1_wxsmoke*.pt"))[0]), ("inexact", sorted(Path(D).glob("ckpt_-1_wwsmoke*.pt"))[0]), ("old", Path(S) / "ckpt_-1_o22Ss0e.pt")):
    net = FT.load_net(ck).eval()
    e, _ = E.load_net(ck)
    za = FT.embed(net, dict(A), idx)
    zb = E.embed(e, A["GL"], A["POS"], A["MOV"], A["LEN"], A["SUBJ"])
    print(f"{name} checkpoint {ck.name}: class {type(net).__name__}, exact {getattr(net, 'win_exact', False)}, the two loaders agree to {float(np.abs(za - zb).max()):.1e}, finite {bool(np.isfinite(za).all())}")
    if name == "exact":
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                net(torch.as_tensor(A["GL"][:4]).cuda(), torch.as_tensor(A["POS"][:4].astype(np.int64)).cuda(), torch.as_tensor(A["MOV"][:4].astype(np.int64)).cuda(),
                    torch.as_tensor(A["LEN"][:4].astype(np.int64)).cuda(), torch.as_tensor(A["SUBJ"][:4].astype(np.int64)).cuda())
            raise SystemExit("the exact network accepted plain colours")
        except ValueError as err:
            print("   plain colours are refused:", str(err)[:70])
        packed = pid_winx.pack_mov(A["POS"], A["MOV"], A["LEN"])
        zero = (A["MOV"].astype(np.int16) + pid_winx.FLAG).astype(np.int16)  # no removals: the inexact board
        z0 = FT.embed(net, dict(A, MOVX=zero), idx)
        print(f"   games with a capture {float(((packed - pid_winx.FLAG) >> 2 > 0).any(1).mean()):.2f}; vectors with / without the removals differ by up to {float(np.abs(za - z0).max()):.3f} (0 = the removals are not used)")
print("loader check passed")
PY
  find $D -delete
  return 0
}
main "$@"; exit $?
