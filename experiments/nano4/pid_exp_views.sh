#!/usr/bin/env bash
# METHOD_NOTES 2.59: test-time symmetry views. A network trained with the symmetry augmentation has seen every orientation;
# reading the whole pool under each of the 8 board symmetries gives 8 score matrices per network to average. Test with the
# two held-out augmented networks (yse, folds 1 and 2): vectors of the real pools, the official pseudo-pools and the old-pool
# simulation under 8 views; then, at EQUAL total weight of the yse networks,
#   v2yv_1 = v2i_eq's members + yse x 8 (identity view only)      v2yv_8 = the same members + the 8 views of each yse network
# on the official candidates (leave-games-out), the pseudo-pools (full pipeline) and the old-pool simulation.
set -uo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
S=$R2/data/seq_ft
SN=/work/u2507115/pid_v2/seqnet
IFS=, read -r -a G <<< "$CUDA_VISIBLE_DEVICES"
FT="python -I $R2/scripts/pid_v2_seq_finetune.py --horizon 1000 --ckpt-dir $SN --ckpt-suffix _yse --tag _yse --views 8"
i=0
for v in 1 2; do
  for pool in pub2 priv2; do
    CUDA_VISIBLE_DEVICES=${G[$((i % ${#G[@]}))]} $FT --mode official --pool $pool --net $v --steps 0 50 --out $S > logs/nano4/views-${SLURM_JOB_ID}-real-$v-$pool.log 2>&1 &
    i=$((i + 1))
    ( for s in 0 1; do CUDA_VISIBLE_DEVICES=${G[$((i % ${#G[@]}))]} $FT --mode official --pool $pool --net $v --steps 0 50 --holdout 3 --holdout-seed $s --out /work/u2507115/pid_v2/seq_ft_offsim; done ) > logs/nano4/views-${SLURM_JOB_ID}-offsim-$v-$pool.log 2>&1 &
    i=$((i + 1))
  done
done
wait
i=0
for v in 1 2; do for side in pub priv; do
  CUDA_VISIBLE_DEVICES=${G[$((i % ${#G[@]}))]} $FT --mode oldsim --side $side --holdout-seed 0 --net $v --steps 0 50 --out /work/u2507115/pid_v2/seq_ft_oldsim > logs/nano4/views-${SLURM_JOB_ID}-oldsim-$v-$side.log 2>&1 &
  i=$((i + 1))
done; done
wait
echo "view files: real $(ls $S | grep -c '_ysev[1-7]_t50'), pseudo-pools $(ls /work/u2507115/pid_v2/seq_ft_offsim | grep -c '_ysev[1-7]_t50'), old-pool simulation $(ls /work/u2507115/pid_v2/seq_ft_oldsim | grep -c '_ysev[1-7]_t50')"
g=("m90=$S/official_{pool}_net{-1|1|2|3|4}_m90s{0|1|2}e_t0.npz" "m101x=$S/official_{pool}_net{-1|1|2|3|4}_m101x_t0.npz" "px=$S/official_{pool}_net{1|2}_xe_t0.npz,$S/official_{pool}_net1_xue_t0.npz"
   "px2=$S/official_{pool}_net{1|2}_x2e_t0.npz" "o1=$S/official_{pool}_net-1_o1s{0|1|2}e_t0.npz" "py=$S/official_{pool}_net{1|2}_yse_t0.npz" "pyv=$S/official_{pool}_net{1|2}_yse{|v1|v2|v3|v4|v5|v6|v7}_t0.npz")
base=m90+m101x+px+o1+px2
python -I $R2/scripts/pid_v2_offcand2.py --groups "${g[@]}" --comps "ref=$base+py*8" "views=$base+pyv" "py_only=py" "pyv_only=pyv" --ref ref --out $R2/results/final/v2_offcand2_views.json
python -I $R2/data/day_scripts/build_v2h.py v2yv_1 "$base+py*8"
python -I $R2/data/day_scripts/build_v2h.py v2yv_8 "$base+pyv"
bash $R2/scripts/nano4/pid_v2_offsim_many.sh v2yv_1 v2yv_8
python -I $R2/scripts/pid_v2_oldsim.py --docs v2yv_1 v2yv_8 --step 50 --out $R2/results/final/v2_oldsim_views.json
