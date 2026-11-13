#!/usr/bin/env bash
# METHOD_NOTES 2.63: how much weight should the symmetry-trained networks get? Weight ladder with the two held-out ones (py),
# against v2y_new, on both instruments (pseudo-pools, old-pool simulation).
set -uo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
all=m90+m101x+px+o1+px2
new=m101x+px+o1+px2+pb
cands=("v2w_a5=$all+py*5" "v2w_a8=$all+py*8" "v2w_a12=$all+py*12" "v2w_n4=$new+py*4" "v2w_n6=$new+py*6" "v2w_n8=$new+py*8" "v2w_s4=o1+px2+py*4")
names=()
for c in "${cands[@]}"; do python -I $R2/data/day_scripts/build_v2h.py "${c%%=*}" "${c#*=}"; names+=("${c%%=*}"); done
python -I $R2/scripts/pid_v2_oldsim.py --docs v2y_new "${names[@]}" --step 50 --out $R2/results/final/v2_oldsim_weights.json
bash $R2/scripts/nano4/pid_v2_offsim_many.sh v2y_new "${names[@]}"
