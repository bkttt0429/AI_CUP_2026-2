#!/usr/bin/env bash
# Pilot of METHOD_NOTES 2.54 (S2): does the pool fine-tune gain from the ORIGINAL pool's candidates of the same side?
#   1. per-ply vectors of 30 candidate games of every original-pool candidate (800 players) -> /work/u2507115/pid_v2/ply_original
#   2. pseudo-pool fine-tunes (3 games per candidate held out, splits 0 and 1) of the three o1 networks, vectors after
#      0 / 50 / 100 / 200 steps, two arms: B = the pool's own candidates only (as now), M = half of every episode from the
#      original pool of the same side
#   3. full-pipeline check on the pseudo-pools: arm B at every step, arm M at 50 / 100 / 200 steps against B at 50
# One job with 8 GPUs, run from the AI_CUP_2026-1 checkout through scripts/nano4/run.sbatch.
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
PLYO=/work/u2507115/pid_v2/ply_original
FT=/work/u2507115/pid_v2/seq_ft_offsim
bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_v2_ply_extract.py --tests-dir $R2/data/tests_1111 --original 30 --out $PLYO --workers 10
echo "original-pool parts: $(ls $PLYO | grep -c 'npz$'), $(du -sh $PLYO | cut -f1)"
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
runs=()
for k in 0 1 2; do for pool in pub2 priv2; do for s in 0 1; do
  runs+=("--pool $pool --holdout-seed $s --ckpt-suffix _o1s${k}e --tag _o1s${k}eB")
  runs+=("--pool $pool --holdout-seed $s --ckpt-suffix _o1s${k}e --tag _o1s${k}eM --mix-original $PLYO")
done; done; done
pids=()
for i in "${!GPUS[@]}"; do
  (
    for j in "${!runs[@]}"; do
      if [ $(( j % ${#GPUS[@]} )) -eq "$i" ]; then
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="${GPUS[$i]}" OMP_NUM_THREADS=8 python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --net -1 --steps 0 50 100 200 --horizon 1000 --holdout 3 \
          --ckpt-dir /work/u2507115/pid_v2/seqnet --out $FT ${runs[$j]}
      fi
    done
  ) > "logs/nano4/s2ft-${SLURM_JOB_ID:-local}-$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "a fine-tune failed; see logs/nano4/s2ft-${SLURM_JOB_ID:-local}-*.log"; exit 1; }
python3 - <<PY
import json
R = "$R2"
ref = json.load(open(f"{R}/results/pid_fusion_v2ge.json"))
for arm in "BM":
    files = [[f"{R}/data/seq_ft/official_{pool}_net-1_o1s{k}e{arm}_t50.npz" for pool in ("pub2", "priv2")] for k in (0, 1, 2)]
    doc = {**ref, "seq_official": files, "composition": [f"o1 arm {arm}"], "note": "pilot of METHOD_NOTES 2.54: names only, used by pid_v2_offsim.py"}
    json.dump(doc, open(f"{R}/results/pid_fusion_v2x_o1{arm}.json", "w"), indent=1)
PY
for t in 50 100 200; do
  python -I $R2/scripts/pid_v2_offsim.py --fusion $R2/results/pid_fusion_v2x_o1B.json --alt $R2/results/pid_fusion_v2x_o1M.json --steps 0 50 100 200 --alt-step $t --workers 12 \
    --out $R2/results/final/v2_offsim_s2pilot_t$t.json
done
