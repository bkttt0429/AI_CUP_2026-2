#!/usr/bin/env bash
# A group of my official sequence networks, from checkpoints to candidate files (parallel version of pid_v2_newnets.sh):
#   1. pool fine-tunes of every EMA checkpoint ckpt_-1_<group>s<k>e.pt (real pools, official pseudo-pools), spread over the
#      job's GPUs, one fine-tune per GPU at a time; the non-EMA checkpoints of the group are deleted (never used, disk)
#   2. leave-games-out check on the official candidates: single networks by group, the reference and the candidates
#   3. per candidate: fusion document; pseudo-pool check against the reference (all candidates side by side); old-pool
#      simulation (scripts/nano4/pid_exp_oldsim.sh); submission file upload/pid_<candidate>s.csv
# usage: [OUT=<name of the result files, default the group>] pid_v2_newnets2.sh <group> <reference document> <candidate>=<composition> ...
# Nothing here chooses a file: the rule is in the notes and is applied to the printed numbers.
set -uo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
S=$R2/data/seq_ft
SN=/work/u2507115/pid_v2/seqnet
J=${SLURM_JOB_ID:-local}
grp=$1; ref=$2
shift 2
out=${OUT:-$grp}
IFS=, read -r -a G <<< "$CUDA_VISIBLE_DEVICES"
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
nets=(); seeds=()
for k in $SEEDS; do
  if [ -s $SN/ckpt_-1_${grp}s${k}e.pt ]; then nets+=("-1 _${grp}s${k}e"); seeds+=("$k"); fi
done
echo "group $grp: ${#nets[@]} network(s), seeds ${seeds[*]}; ${#G[@]} GPU(s)"
[ "${#nets[@]}" -gt 0 ] || { echo "no checkpoint of group $grp"; exit 1; }
rm -f $SN/ckpt_-1_${grp}s[0-9].pt $SN/ckpt_-1_${grp}s[0-9][0-9].pt
tasks=()
for n in "${nets[@]}"; do tasks+=("real|$n" "offsim|$n"); done
pids=()
for i in "${!G[@]}"; do
  (
    for j in "${!tasks[@]}"; do
      [ $(( j % ${#G[@]} )) -eq "$i" ] || continue
      t=${tasks[$j]}
      CUDA_VISIBLE_DEVICES=${G[$i]} bash $R2/scripts/nano4/pid_v2_ft_many2.sh "${t%%|*}" "${t#*|}"
    done
  ) > logs/nano4/newnets-$J-ft-$i.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "a fine-tune worker failed (logs/nano4/newnets-$J-ft-*.log)"; done
echo "fine-tunes done: $(ls $S | grep -c "_${grp}s[0-9]*e_t50") official files, $(ls /work/u2507115/pid_v2/seq_ft_offsim/ | grep -c "_${grp}s[0-9]*e_t50") pseudo-pool files"
list() {  # brace list of the seeds of a group that have official vectors
  local g=$1 o=() k
  for k in $SEEDS; do [ -s $S/official_pub2_net-1_${g}s${k}e_t0.npz ] && o+=("$k"); done
  (IFS='|'; echo "${o[*]}")
}
groups=("m90=$S/official_{pool}_net{-1|1|2|3|4}_m90s{0|1|2}e_t0.npz" "m101x=$S/official_{pool}_net{-1|1|2|3|4}_m101x_t0.npz"
        "px=$S/official_{pool}_net{1|2}_xe_t0.npz,$S/official_{pool}_net1_xue_t0.npz" "px2=$S/official_{pool}_net{1|2}_x2e_t0.npz"
        "py=$S/official_{pool}_net{1|2}_yse_t0.npz" "pb=$S/official_{pool}_net{1|2}_ybe_t0.npz")
for g in o1 o2 o3S o3SR o3BS o3SL o3SK o3SM o4S o4SR; do
  l=$(list $g)
  if [[ "$l" == *"|"* ]]; then groups+=("$g=$S/official_{pool}_net-1_${g}s{$l}e_t0.npz"); elif [ -n "$l" ]; then groups+=("$g=$S/official_{pool}_net-1_${g}s${l}e_t0.npz"); fi
done
refcomp=$(python3 -c "import json; print('+'.join(json.load(open('$R2/results/pid_fusion_$ref.json'))['composition']))")
python -I $R2/scripts/pid_v2_offcand2.py --groups "${groups[@]}" --comps "ref=$refcomp" "$@" "only=$grp" --ref ref --out $R2/results/final/v2_offcand2_$out.json
names=()
for c in "$@"; do
  python -I $R2/data/day_scripts/build_v2h.py "${c%%=*}" "${c#*=}"
  names+=("${c%%=*}")
done
pids=()
for n in "${names[@]}"; do
  python -I $R2/scripts/pid_v2_offsim.py --fusion $R2/results/pid_fusion_$ref.json --alt $R2/results/pid_fusion_$n.json --steps 50 --workers 8 \
    --out $R2/results/final/v2_offsim_${n}_vs_${ref}.json > logs/nano4/newnets-$J-offsim-$n.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "a pseudo-pool check failed (logs/nano4/newnets-$J-offsim-*.log)"; done
for n in "${names[@]}"; do
  echo "== pseudo-pools: $n vs $ref  $(grep -E '^   talt vs t50' logs/nano4/newnets-$J-offsim-$n.log | tail -1)"
  echo "      $(grep -E '^both: document weights' logs/nano4/newnets-$J-offsim-$n.log | tail -1 | cut -c1-260)"
done
bash $R2/scripts/nano4/pid_exp_oldsim.sh "$grp" "$ref ${names[*]}" "$out" || echo "old-pool simulation failed"
pids=()
for n in "${names[@]}"; do
  bash $R2/scripts/nano4/pid_v2_make_file.sh "$n" "pid_${n}s.csv" > logs/nano4/newnets-$J-file-$n.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "a file was not built (logs/nano4/newnets-$J-file-*.log)"; done
for n in "${names[@]}"; do echo "== file pid_${n}s.csv: $(grep -E 'questions with a different Top-1' logs/nano4/newnets-$J-file-$n.log | head -1) | md5 $(md5sum $R2/upload/pid_${n}s.csv 2>/dev/null | cut -c1-12)"; done
