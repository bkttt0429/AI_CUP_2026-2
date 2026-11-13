#!/usr/bin/env bash
# Simulated pools from the ORIGINAL pools' candidates (METHOD_NOTES 2.55, item 7): fine-tune the networks of the given
# families on the simulated pools' candidate sets (both sides, split 0; vectors after 0 and 50 steps), then score documents.
#   pid_exp_oldsim.sh "<families>" "<documents, the first is the reference>" <result name> [--fit-weights]
# Families: m88 m90 m101x px px2, and any family of my official networks (o1 o2 o3 o3hm ...: every ckpt_-1_<family>s<k>e.pt).
# One job with up to 8 GPUs, from the AI_CUP_2026-1 checkout. Per-ply vectors of the original pools: scripts/pid_v2_ply_extract.py --original.
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
A1=/home/u2507115/BK_AICUP_2026-1
S=/work/u2507115/pid_v2/seqnet
OUT=/work/u2507115/pid_v2/seq_ft_oldsim
fams=$1; docs=$2; name=$3; extra=${4:-}
runs=()  # "<net> <tag or -> <checkpoint options>"
for f in $fams; do
  case $f in
    m88)   for v in -1 1 2 3 4; do runs+=("$v - --ckpt-dir $A1/results/m88"); done ;;
    m90)   for v in -1 1 2 3 4; do
             runs+=("$v _m90s0e --ckpt-dir /work/u2507115/pid_v2/m90_snap")
             runs+=("$v _m90s1e --ckpt-dir /work/u2507115/pid_v2/m90_snap --ckpt-suffix _s1")
             runs+=("$v _m90s2e --ckpt-dir /work/u2507115/pid_v2/m90_snap --ckpt-suffix _s2")
           done ;;
    m101x) for v in -1 1 2 3 4; do runs+=("$v _m101x --ckpt-dir $A1/results/m101 --ckpt-suffix _s3 --state last"); done ;;
    px)    runs+=("1 _xe --ckpt-dir $S --ckpt-suffix _xe" "2 _xe --ckpt-dir $S --ckpt-suffix _xe" "1 _xue --ckpt-dir $S --ckpt-suffix _xue") ;;
    px2)   runs+=("1 _x2e --ckpt-dir $S --ckpt-suffix _x2e" "2 _x2e --ckpt-dir $S --ckpt-suffix _x2e") ;;
    py)    runs+=("1 _yse --ckpt-dir $S --ckpt-suffix _yse" "2 _yse --ckpt-dir $S --ckpt-suffix _yse") ;;
    pb)    runs+=("1 _ybe --ckpt-dir $S --ckpt-suffix _ybe" "2 _ybe --ckpt-dir $S --ckpt-suffix _ybe") ;;
    *)     for k in 0 1 2 3 4 5 6 7; do [ -s $S/ckpt_-1_${f}s${k}e.pt ] && runs+=("-1 _${f}s${k}e --ckpt-dir $S --ckpt-suffix _${f}s${k}e"); done ;;
  esac
done
echo "${#runs[@]} networks to fine-tune on the simulated pools"
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
pids=()
for i in "${!GPUS[@]}"; do
  (
    for j in "${!runs[@]}"; do
      [ $(( j % ${#GPUS[@]} )) -eq "$i" ] || continue
      read -r v tag opts <<< "${runs[$j]}"
      [ "$tag" = "-" ] && tag=""
      for side in pub priv; do
        [ -s "$OUT/oldsim_${side}_s0_net${v}${tag}_t50.npz" ] && continue
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="${GPUS[$i]}" OMP_NUM_THREADS=8 python -I $R2/scripts/pid_v2_seq_finetune.py --mode oldsim --side $side --holdout-seed 0 --net $v --steps 0 50 --horizon 1000 \
          --out $OUT $opts ${tag:+--tag $tag}
      done
    done
  ) > "logs/nano4/oldsim-${SLURM_JOB_ID:-local}-$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "a fine-tune failed; see logs/nano4/oldsim-${SLURM_JOB_ID:-local}-*.log"; exit 1; }
# shellcheck disable=SC2086
python -I $R2/scripts/pid_v2_oldsim.py --docs $docs --step 50 $extra --out $R2/results/final/v2_oldsim_$name.json
# shellcheck disable=SC2086
python -I $R2/scripts/pid_v2_oldsim.py --docs $docs --step 0 --out $R2/results/final/v2_oldsim_${name}_t0.json
