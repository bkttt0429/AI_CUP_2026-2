#!/usr/bin/env bash
# The held-out networks of tonight's two passing recipes as official members right away (METHOD_NOTES 2.58): py = folds 1 and 2
# with the symmetry augmentation (yse), pb = folds 1 and 2 of the bigger size (ybe). Like px / px2 they have seen three
# competition folds and 39,683 external accounts, no official player. Fine-tunes on several GPUs, then both instruments
# against v2i_eq, then the files.
set -uo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
S=$R2/data/seq_ft
IFS=, read -r -a G <<< "$CUDA_VISIBLE_DEVICES"
i=0
for spec in "1 _yse" "2 _yse" "1 _ybe" "2 _ybe"; do
  CUDA_VISIBLE_DEVICES=${G[$((i % ${#G[@]}))]} bash $R2/scripts/nano4/pid_v2_ft_many.sh real "$spec" > logs/nano4/py-${SLURM_JOB_ID}-real-$i.log 2>&1 &
  CUDA_VISIBLE_DEVICES=${G[$(((i + 4) % ${#G[@]}))]} bash $R2/scripts/nano4/pid_v2_ft_many.sh offsim "$spec" > logs/nano4/py-${SLURM_JOB_ID}-offsim-$i.log 2>&1 &
  i=$((i + 1))
done
wait
ls $S | grep -cE "_y[sb]e_t50" ; ls /work/u2507115/pid_v2/seq_ft_offsim | grep -cE "_y[sb]e_t50"
g=("m90=$S/official_{pool}_net{-1|1|2|3|4}_m90s{0|1|2}e_t0.npz" "m101x=$S/official_{pool}_net{-1|1|2|3|4}_m101x_t0.npz" "px=$S/official_{pool}_net{1|2}_xe_t0.npz,$S/official_{pool}_net1_xue_t0.npz"
   "px2=$S/official_{pool}_net{1|2}_x2e_t0.npz" "o1=$S/official_{pool}_net-1_o1s{0|1|2}e_t0.npz" "py=$S/official_{pool}_net{1|2}_yse_t0.npz" "pb=$S/official_{pool}_net{1|2}_ybe_t0.npz")
base=m90+m101x+px+o1+px2
cands=("v2y_eq=$base+py" "v2y_w=$base+py*3" "v2y_b=$base+py+pb" "v2y_bw=$base+py*3+pb*2" "v2y_new=m101x+px+o1+px2+py*2+pb")
python -I $R2/scripts/pid_v2_offcand2.py --groups "${g[@]}" --comps "ref=$base" "${cands[@]}" "py_only=py" "px2_only=px2" "pb_only=pb" --ref ref --out $R2/results/final/v2_offcand2_py.json
names=()
for c in "${cands[@]}"; do python -I $R2/data/day_scripts/build_v2h.py "${c%%=*}" "${c#*=}"; names+=("${c%%=*}"); done
bash $R2/scripts/nano4/pid_v2_offsim_many.sh v2i_eq "${names[@]}"
bash $R2/scripts/nano4/pid_exp_oldsim.sh "py pb" "v2i_eq ${names[*]}" py || echo "old-pool simulation failed"
for n in "${names[@]}"; do bash $R2/scripts/nano4/pid_v2_make_file.sh "$n" "pid_${n}s.csv"; done
