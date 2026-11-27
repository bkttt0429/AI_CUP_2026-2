#!/usr/bin/env bash
# METHOD_NOTES 2.55, items 4 and 3. After the S2 pilot (its arm B = the present fine-tune of the three o1 networks at 0 / 50 /
# 100 / 200 steps on the official pseudo-pools):
#   item 4  the same fine-tune with a larger learning rate and / or a short warm-up: L3 (3e-4, warm-up 50), W10 (1e-4, warm-up
#           10), L3W10 (3e-4, warm-up 10); vectors after 10 / 25 / 50 / 100 steps; every arm and step against B at 50 steps
#   item 3  the fusion weights of the live composition (v2i_eq) refitted on each pool's own candidates (half / half)
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
FT=/work/u2507115/pid_v2/seq_ft_offsim
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
runs=()
for k in 0 1 2; do for pool in pub2 priv2; do for s in 0 1; do
  runs+=("--pool $pool --holdout-seed $s --ckpt-suffix _o1s${k}e --tag _o1s${k}eL3 --lr 3e-4")
  runs+=("--pool $pool --holdout-seed $s --ckpt-suffix _o1s${k}e --tag _o1s${k}eW10 --warm 10")
  runs+=("--pool $pool --holdout-seed $s --ckpt-suffix _o1s${k}e --tag _o1s${k}eL3W10 --lr 3e-4 --warm 10")
done; done; done
pids=()
for i in "${!GPUS[@]}"; do
  (
    for j in "${!runs[@]}"; do
      if [ $(( j % ${#GPUS[@]} )) -eq "$i" ]; then
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="${GPUS[$i]}" OMP_NUM_THREADS=8 python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --net -1 --steps 10 25 50 100 --horizon 1000 --holdout 3 \
          --ckpt-dir /work/u2507115/pid_v2/seqnet --out $FT ${runs[$j]}
      fi
    done
  ) > "logs/nano4/ftgrid-${SLURM_JOB_ID:-local}-$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "a fine-tune failed; see logs/nano4/ftgrid-${SLURM_JOB_ID:-local}-*.log"; exit 1; }
python3 - <<PY
import json
R = "$R2"
ref = json.load(open(f"{R}/results/pid_fusion_v2ge.json"))
for arm in ("L3", "W10", "L3W10"):
    files = [[f"{R}/data/seq_ft/official_{pool}_net-1_o1s{k}e{arm}_t50.npz" for pool in ("pub2", "priv2")] for k in (0, 1, 2)]
    json.dump({**ref, "seq_official": files, "composition": [f"o1 arm {arm}"], "note": "METHOD_NOTES 2.55 item 4: names only, used by pid_v2_offsim.py"}, open(f"{R}/results/pid_fusion_v2x_o1{arm}.json", "w"), indent=1)
PY
pids=()
for arm in L3 W10 L3W10; do for t in 10 25 50 100; do
  python -I $R2/scripts/pid_v2_offsim.py --fusion $R2/results/pid_fusion_v2x_o1B.json --alt $R2/results/pid_fusion_v2x_o1$arm.json --steps 50 --alt-step $t --workers 6 \
    --out $R2/results/final/v2_offsim_ftgrid_${arm}_t$t.json > "logs/nano4/ftgrid-${SLURM_JOB_ID:-local}-eval-${arm}-$t.log" 2>&1 &
  pids+=($!)
done; done
python -I $R2/scripts/pid_v2_offsim.py --fusion $R2/results/pid_fusion_v2i_eq.json --steps 50 --workers 6 --out $R2/results/final/v2_offsim_v2i_eq_calibration.json > "logs/nano4/ftgrid-${SLURM_JOB_ID:-local}-eval-calibration.log" 2>&1 &
pids+=($!)
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for arm in L3 W10 L3W10; do for t in 10 25 50 100; do
  echo "== $arm at $t steps against B at 50: $(grep -E '^   talt vs t50' logs/nano4/ftgrid-${SLURM_JOB_ID:-local}-eval-${arm}-$t.log | tail -1)"
done; done
echo "== calibration of the weights (v2i_eq):"; grep -E "^calibrated|^both" "logs/nano4/ftgrid-${SLURM_JOB_ID:-local}-eval-calibration.log"
exit "$fail"
