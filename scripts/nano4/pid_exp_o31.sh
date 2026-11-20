#!/usr/bin/env bash
# METHOD_NOTES 2.170 (probe W): family o31 = the recipe, data and seeds of o22 (plain inputs, by-games, 40,000 steps) with ONE
# change - every ply also gets the stones in a 9 x 9 window around its point (scripts/pid_win.py, trainer
# scripts/pid_seq_train_w.py --shape-win 9). No new data, no feature extraction.
#   pid_exp_o31.sh smoke            the window builder against a plain replay; 300 smoke steps on two GPUs; both loaders on the
#                                   smoke checkpoint and on an old checkpoint (the chains of tonight use these loaders)
#   pid_exp_o31.sh train <0|1>      group 0 = o31Ss0-3 (seeds 180-183), group 1 = o31SMs0-3 (seeds 184-187, hidden openings);
#                                   four models x two GPUs (as scripts/nano4/pid_exp_s3b_train.sh)
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox S=/work/u2507115/pid_v2/seqnet mode=$1 J=${SLURM_JOB_ID:-local} cards pair slot i tag extra seed free lines=()
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local C="--steps 40000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --ema-only --own-by-games linear --shape-win 9"
  IFS=, read -r -a cards <<< "${CUDA_VISIBLE_DEVICES:?}"
  python -I -m py_compile $R2/scripts/pid_win.py $R2/scripts/pid_seq_train_w.py $R2/scripts/pid_v2_seq_embed.py $R2/scripts/pid_v2_seq_finetune.py || { echo "a script does not compile"; return 1; }
  case "$mode" in
    smoke)
      local D=/work/u2507115/pid_v2/seqnet_smoke_w
      mkdir -p $D
      python -I $R2/scripts/pid_win.py || { echo "the window builder FAILED its self-test"; return 1; }
      CUDA_VISIBLE_DEVICES=${cards[0]},${cards[1]} OMP_NUM_THREADS=8 python -I $R2/scripts/pid_seq_train_w.py --net -1 --tag wsmoke --smoke --no-state --sym-aug --mask 0.3 --ema-only --shape-win 9 --out $D \
        > logs/nano4/o31smoke-$J-train.log 2>&1 || { echo "the smoke training FAILED"; tail -30 logs/nano4/o31smoke-$J-train.log; return 1; }
      grep -E "window input|training forward|step +[0-9]+|held-out|wrote|saved" logs/nano4/o31smoke-$J-train.log | cut -c1-220 | tail -14
      CUDA_VISIBLE_DEVICES=${cards[0]} python -I - $R2 $D $S <<'EOF' || { echo "the loader check FAILED"; return 1; }
import sys, time
from pathlib import Path
import numpy as np
import torch
R2, D, S = sys.argv[1:4]
sys.path[:0] = [R2 + "/scripts", R2 + "/src"]
import pid_v2_seq_embed as E
import pid_v2_seq_finetune as FT
new = sorted(Path(D).glob("ckpt_-1_wsmoke*_smoke.pt"))[0]
old = Path(S) / "ckpt_-1_o22Ss0e.pt"
rng = np.random.default_rng(0)
B, T = 256, 256
pos = torch.as_tensor(rng.integers(0, 361, (B, T))).cuda(); mov = torch.as_tensor(np.tile(np.arange(T) % 2, (B, 1))).cuda()
gl = torch.zeros(B, T, 128).cuda(); ln = torch.full((B,), T).cuda(); sj = torch.zeros(B, dtype=torch.long).cuda()
for name, ck in (("new", new), ("old", old)):
    a, _ = E.load_net(ck)
    b = FT.load_net(ck).eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        za, zb = a(gl, pos, mov, ln, sj).float(), b(gl, pos, mov, ln, sj).float()
        t0 = time.time()
        for _ in range(5):
            a(gl, pos, mov, ln, sj)
        torch.cuda.synchronize()
    print(f"{name} checkpoint {ck.name}: class {type(a).__name__} / {type(b).__name__}, window module {hasattr(a, 'win')}, both loaders agree to {float((za - zb).abs().max()):.1e}, "
          f"finite {bool(torch.isfinite(za).all())}, forward of {B} games {1000 * (time.time() - t0) / 5:.0f} ms, GPU memory peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    if name == "new":
        w = a.win.out.weight.float().abs().mean().item()
        print(f"   the window term's output layer after the smoke steps: mean |weight| {w:.2e} (0 = never trained)")
        assert hasattr(a, "win") and w > 0
    else:
        assert not hasattr(a, "win")
print("loader check passed")
EOF
      find $D -name 'ckpt_-1_wsmoke*' -delete; rmdir $D 2> /dev/null
      return 0 ;;
    train)
      local grp=$2 d
      for d in $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d; do
        [ "$(ls "$d" 2> /dev/null | grep -c 'ply_x_.*[0-9]\.npz$')" -ge 200 ] || { echo "external ply folder $d is missing or incomplete"; return 1; }
      done
      free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')
      [ "$free" -ge 5000 ] || { echo "/work has only $free MB free: not started"; return 1; }
      [ "${#cards[@]}" -eq 8 ] || { echo "expected eight GPUs"; return 1; }
      for slot in 0 1 2 3; do
        pair=${cards[$((2 * slot))]},${cards[$((2 * slot + 1))]}
        CUDA_VISIBLE_DEVICES="$pair" timeout 600 python -u -c 'import torch; m=torch.nn.Linear(4,4).cuda(); y=torch.nn.DataParallel(m)(torch.ones(8,4,device="cuda")); y.sum().backward(); torch.cuda.synchronize()' \
          || { echo "GPU pair $pair on $(hostname) failed the two-card check"; return 1; }
      done
      for slot in 0 1 2 3; do
        if [ "$grp" = 0 ]; then tag=o31Ss$slot; extra=""; seed=$((180 + slot)); else tag=o31SMs$slot; extra="--drop-open 0.3"; seed=$((184 + slot)); fi
        [ -e $S/ckpt_-1_${tag}e.pt ] && { echo "ckpt of $tag exists already"; return 1; }
        pair=${cards[$((2 * slot))]},${cards[$((2 * slot + 1))]}
        lines+=("$pair logs/nano4/pidseq_-1_${tag}-$J.log --net -1 --tag $tag $C $extra --seed $seed")
      done
      local spec
      spec=$(mktemp "${TMPDIR:-/tmp}/o31_spec_XXXXXX")
      printf '%s\n' "${lines[@]}" > "$spec"
      cat "$spec"
      echo "node $(hostname): family o31 group $grp, four models x two GPUs; /work free $free MB"
      # shellcheck disable=SC2086
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_seq_train_w.py --fork "$spec" $OFF
      local rc=$?
      rm -f "$spec"
      grep -hE "training on|window input" logs/nano4/pidseq_-1_o31*-$J.log | sort -u | cut -c1-260
      ls $S | grep -c "ckpt_-1_o31.*e\.pt"
      return $rc ;;
    *) echo "usage: pid_exp_o31.sh smoke | train <0|1>"; return 2 ;;
  esac
}
main "$@"; exit $?
