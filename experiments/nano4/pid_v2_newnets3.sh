#!/usr/bin/env bash
# As pid_v2_newnets2.sh (a group of my official sequence networks, from checkpoints to candidate files), with the standing
# rule of METHOD_NOTES 2.121 built in:
#   - official pseudo-pools on FOUR hold-out seeds: 0 1 (the seeds every earlier chain selected on) and 2 3 (fresh); the
#     check against the reference is printed for seeds 0 1, for seeds 2 3 and for all four
#   - two files per candidate: upload/pid_<candidate>s.csv (soft assignment lam 1, as every file up to 2026-11-15) and
#     upload/pid_<candidate>_a2k.csv (lam 2 for the Top-1 + Sinkhorn order tau 0.5, METHOD_NOTES 2.122)
# usage: [OUT=<name of the result files, default the group>] pid_v2_newnets3.sh <group> <reference document> <candidate>=<composition> ...
# Families other than <group> inside a composition must already have their vectors (real pools, pseudo-pools of all four
# seeds, old-pool simulation). Nothing here chooses a file: the rule is in the notes and is applied to the printed numbers.
# Do not edit while a job runs it; the body is a function called on the last line.
set -uo pipefail
main() {
local R2=/home/u2507115/BK_AICUP_2026-2 SN=/work/u2507115/pid_v2/seqnet O=/work/u2507115/pid_v2/seq_ft_offsim J=${SLURM_JOB_ID:-local}
local S=$R2/data/seq_ft grp=$1 ref=$2 out k n i j t p l g c s v sfx pool refcomp fams
shift 2
out=${OUT:-$grp}
local G; IFS=, read -r -a G <<< "$CUDA_VISIBLE_DEVICES"
local SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
local nets=() seeds=() tasks=() pids=() names=() groups=()
for k in $SEEDS; do
  if [ -s $SN/ckpt_-1_${grp}s${k}e.pt ]; then nets+=("-1 _${grp}s${k}e"); seeds+=("$k"); fi
done
echo "group $grp: ${#nets[@]} network(s), seeds ${seeds[*]}; ${#G[@]} GPU(s)"
[ "${#nets[@]}" -gt 0 ] || { echo "no checkpoint of group $grp"; return 1; }
for n in "${nets[@]}"; do tasks+=("real|$n" "offsim|$n" "fresh|$n"); done
for i in "${!G[@]}"; do
  (
    for j in "${!tasks[@]}"; do
      [ $(( j % ${#G[@]} )) -eq "$i" ] || continue
      t=${tasks[$j]}
      if [ "${t%%|*}" = fresh ]; then
        read -r v sfx <<< "${t#*|}"
        for pool in pub2 priv2; do for s in 2 3; do
          [ -s "$O/offsim_${pool}_h3s${s}_net${v}${sfx}_t50.npz" ] && continue
          CUDA_VISIBLE_DEVICES=${G[$i]} python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --pool "$pool" --net "$v" --horizon 1000 --ckpt-dir $SN --ckpt-suffix "$sfx" --tag "$sfx" \
            --steps 50 --holdout 3 --holdout-seed "$s" --out "$O" || echo "FAILED: net $v$sfx $pool seed $s"
        done; done
      else
        CUDA_VISIBLE_DEVICES=${G[$i]} bash $R2/scripts/nano4/pid_v2_ft_many2.sh "${t%%|*}" "${t#*|}" || echo "FAILED: $t"
      fi
    done
  ) > logs/nano4/newnets-$J-ft-$i.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "a fine-tune worker failed (logs/nano4/newnets-$J-ft-*.log)"; done
echo "fine-tunes done at $(date +%H:%M:%S): $(ls $S | grep -c "_${grp}s[0-9]*e_t50") official files, pseudo-pool files per seed: $(for s in 0 1 2 3; do ls $O | grep -c "_h3s${s}_net-1_${grp}s[0-9]*e_t50"; done | tr '\n' ' '); failures: $(grep -c FAILED logs/nano4/newnets-$J-ft-*.log | awk -F: '{n+=$2} END{print n+0}')"
list() {  # brace list of the seeds of a group that have official vectors
  local g=$1 o=() k
  for k in $SEEDS; do [ -s $S/official_pub2_net-1_${g}s${k}e_t0.npz ] && [ -s $S/official_priv2_net-1_${g}s${k}e_t0.npz ] && o+=("$k"); done
  (IFS='|'; echo "${o[*]}")
}
groups=("m90=$S/official_{pool}_net{-1|1|2|3|4}_m90s{0|1|2}e_t0.npz" "m101x=$S/official_{pool}_net{-1|1|2|3|4}_m101x_t0.npz"
        "px=$S/official_{pool}_net{1|2}_xe_t0.npz,$S/official_{pool}_net1_xue_t0.npz" "px2=$S/official_{pool}_net{1|2}_x2e_t0.npz"
        "py=$S/official_{pool}_net{1|2}_yse_t0.npz" "pb=$S/official_{pool}_net{1|2}_ybe_t0.npz")
fams=$(ls $S | sed -nE 's/^official_pub2_net-1_(o[0-9][A-Za-z0-9]*)s[0-9]+e_t0\.npz$/\1/p' | sort -u | tr '\n' ' ')
for g in $fams; do
  l=$(list $g)
  if [[ "$l" == *"|"* ]]; then groups+=("$g=$S/official_{pool}_net-1_${g}s{$l}e_t0.npz"); elif [ -n "$l" ]; then groups+=("$g=$S/official_{pool}_net-1_${g}s${l}e_t0.npz"); fi
done
refcomp=$(python3 -c "import json; print('+'.join(json.load(open('$R2/results/pid_fusion_$ref.json'))['composition']))")
python -I $R2/scripts/pid_v2_offcand2.py --groups "${groups[@]}" --comps "ref=$refcomp" "$@" "only=$grp" --ref ref --out $R2/results/final/v2_offcand2_$out.json
for c in "$@"; do
  python -I $R2/data/day_scripts/build_v2h.py "${c%%=*}" "${c#*=}"
  names+=("${c%%=*}")
done
pids=()
for n in "${names[@]}"; do
  for s in "0 1" "2 3" "0 1 2 3"; do
    # shellcheck disable=SC2086
    python -I $R2/scripts/pid_v2_offsim.py --fusion $R2/results/pid_fusion_$ref.json --alt $R2/results/pid_fusion_$n.json --steps 50 --seeds $s --workers 6 \
      --out $R2/results/final/v2_offsim_${n}_vs_${ref}_s${s// /_}.json > logs/nano4/newnets-$J-offsim-$n-s${s// /_}.log 2>&1 &
    pids+=($!)
  done
done
for p in "${pids[@]}"; do wait "$p" || echo "a pseudo-pool check failed (logs/nano4/newnets-$J-offsim-*.log)"; done
for n in "${names[@]}"; do
  for s in 0_1 2_3 0_1_2_3; do
    echo "== pseudo-pools, hold-out seeds ${s//_/ }: $n vs $ref  $(grep -E '^   talt vs t50' logs/nano4/newnets-$J-offsim-$n-s$s.log | tail -1)"
    grep -E '^(pub2|priv2): document weights|^   talt vs t50' logs/nano4/newnets-$J-offsim-$n-s$s.log | head -4 | cut -c1-200 | sed 's/^/        /'
  done
  echo "      $(grep -E '^both: document weights' logs/nano4/newnets-$J-offsim-$n-s0_1_2_3.log | tail -1 | cut -c1-260)"
done
bash $R2/scripts/nano4/pid_exp_oldsim.sh "$grp" "$ref ${names[*]}" "$out" || echo "old-pool simulation failed"
pids=()
for n in "${names[@]}"; do
  ( bash $R2/scripts/nano4/pid_v2_make_file2.sh "$n" "pid_${n}s.csv" 1 && bash $R2/scripts/nano4/pid_v2_make_file2.sh "$n" "pid_${n}_a2k.csv" 2 0.5 \
      && python -I $R2/scripts/pid_diff_files.py $R2/upload/pid_v2o13c_9_a2k.csv $R2/upload/pid_${n}_a2k.csv \
      && bash $R2/scripts/nano4/pid_v2_make_file4.sh "$n" "pid_${n}_a2k_p3.csv" 2 0.5 0.05 1.6 pid_v2o16c_b_a2k_p3.csv ) > logs/nano4/newnets-$J-file-$n.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "a file was not built (logs/nano4/newnets-$J-file-*.log)"; done
for n in "${names[@]}"; do
  echo "== file pid_${n}s.csv (lam 1) vs pid_v2o13c_9s.csv: $(grep -E 'questions with a different Top-1' logs/nano4/newnets-$J-file-$n.log | sed -n 1p) | md5 $(md5sum $R2/upload/pid_${n}s.csv 2>/dev/null | cut -c1-32)"
  echo "== file pid_${n}_a2k.csv (lam 2, tau 0.5) vs pid_v2o13c_9_a2k.csv: $(grep -E 'questions with a different Top-1' logs/nano4/newnets-$J-file-$n.log | sed -n 5p) | md5 $(md5sum $R2/upload/pid_${n}_a2k.csv 2>/dev/null | cut -c1-32)"
  echo "== file pid_${n}_a2k_p3.csv (+ prior of the 1-game questions, w 0.05) vs pid_v2o16c_b_a2k_p3.csv: $(grep -E 'questions with a different Top-1' logs/nano4/newnets-$J-file-$n.log | tail -1) | md5 $(md5sum $R2/upload/pid_${n}_a2k_p3.csv 2>/dev/null | cut -c1-32)"
done
}
main "$@"; exit $?
