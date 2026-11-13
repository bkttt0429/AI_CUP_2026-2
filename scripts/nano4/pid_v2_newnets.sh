#!/usr/bin/env bash
# A new group of my official sequence networks, from checkpoints to candidate files, unattended (METHOD_NOTES 2.51):
#   1. pool fine-tunes of every EMA checkpoint ckpt_-1_<group>s<k>e.pt that exists (real pools and official pseudo-pools)
#   2. leave-games-out check on the official candidates (pid_v2_offcand2.py): single networks by group, the reference
#      composition and every candidate composition
#   3. per candidate: fusion document, full-pipeline check on the official pseudo-pools against every reference document,
#      and the submission file upload/pid_<candidate>s.csv
# usage: pid_v2_newnets.sh <group> <reference document[,reference document...]> <candidate>=<composition> ...
#   e.g. pid_v2_newnets.sh o2 v2i_eq v2j_eq=m90+m101x+px+o1+px2+o2 v2j_new=m101x+px+o1+px2+o2
# Nothing here chooses a file: the rule is in the notes and is applied to the printed numbers.
set -euo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
S=$R2/data/seq_ft
grp=$1; refs=$2
shift 2
nets=(); seeds=()
for k in 0 1 2 3 4 5 6 7; do
  if [ -s /work/u2507115/pid_v2/seqnet/ckpt_-1_${grp}s${k}e.pt ]; then nets+=("-1 _${grp}s${k}e"); seeds+=("$k"); fi
done
echo "group $grp: ${#nets[@]} network(s), seeds ${seeds[*]}"
[ "${#nets[@]}" -gt 0 ] || { echo "no checkpoint of group $grp"; exit 1; }
bash $R2/scripts/nano4/pid_v2_ft_many.sh real "${nets[@]}"
bash $R2/scripts/nano4/pid_v2_ft_many.sh offsim "${nets[@]}"
list() {  # brace list of the seeds of a group that have official vectors
  local g=$1 out=() k
  for k in 0 1 2 3 4 5 6 7; do [ -s $S/official_pub2_net-1_${g}s${k}e_t0.npz ] && out+=("$k"); done
  (IFS='|'; echo "${out[*]}")
}
groups=("m90=$S/official_{pool}_net{-1|1|2|3|4}_m90s{0|1|2}e_t0.npz" "m101x=$S/official_{pool}_net{-1|1|2|3|4}_m101x_t0.npz"
        "px=$S/official_{pool}_net{1|2}_xe_t0.npz,$S/official_{pool}_net1_xue_t0.npz" "px2=$S/official_{pool}_net{1|2}_x2e_t0.npz")
for g in o1 o2 o3 o3hm o3hb o4 o4hm o4hb; do
  l=$(list $g)
  if [[ "$l" == *"|"* ]]; then groups+=("$g=$S/official_{pool}_net-1_${g}s{$l}e_t0.npz"); elif [ -n "$l" ]; then groups+=("$g=$S/official_{pool}_net-1_${g}s${l}e_t0.npz"); fi
done
first=${refs%%,*}
refcomp=$(python3 -c "import json; print('+'.join(json.load(open('$R2/results/pid_fusion_$first.json'))['composition']))")
python -I $R2/scripts/pid_v2_offcand2.py --groups "${groups[@]}" --comps "ref=$refcomp" "$@" "only=$grp" --ref ref --out $R2/results/final/v2_offcand2_$grp.json
names=()
for c in "$@"; do
  python -I $R2/data/day_scripts/build_v2h.py "${c%%=*}" "${c#*=}"
  names+=("${c%%=*}")
done
for r in ${refs//,/ }; do bash $R2/scripts/nano4/pid_v2_offsim_many.sh "$r" "${names[@]}"; done
for n in "${names[@]}"; do bash $R2/scripts/nano4/pid_v2_make_file.sh "$n" "pid_${n}s.csv"; done
# second instrument (METHOD_NOTES 2.55, item 7): simulated pools cut from the ORIGINAL pools' candidates, 10-game sets and 5-game questions
bash $R2/scripts/nano4/pid_exp_oldsim.sh "$grp" "$first ${names[*]}" "$grp" || echo "old-pool simulation failed"
