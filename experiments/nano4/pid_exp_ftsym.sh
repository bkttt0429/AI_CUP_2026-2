#!/usr/bin/env bash
# METHOD_NOTES 2.69: the pool fine-tune of the symmetry-augmented networks never used the augmentation (the trainer applies
# one random board symmetry per episode; scripts/pid_v2_seq_finetune.py applied none). Does it help there, and does a longer
# fine-tune pay once the network cannot lean on absolute coordinates?
#   networks: the two held-out symmetry networks of the live composition (yse of folds 1 and 2, weight 8 in v2w_n8)
#   variants: P = the usual fine-tune, F = one random board symmetry per fine-tune step (--sym-aug); the same episodes in both;
#             vectors after 50 / 100 / 150 steps
#   pools:    the real added pools, the official pseudo-pools (3 games held out, splits 0 and 1), the old-pool simulation
#   documents v2f_<variant><steps>: v2w_n8 with its yse vectors replaced; reference v2w_n8 (the live file's composition)
# One job with up to 8 GPUs, from the AI_CUP_2026-1 checkout. Nothing here chooses: the rule is in the notes.
set -uo pipefail
R2=/home/u2507115/BK_AICUP_2026-2
S=/work/u2507115/pid_v2/seqnet
REAL=$R2/data/seq_ft
OFFS=/work/u2507115/pid_v2/seq_ft_offsim
OLDS=/work/u2507115/pid_v2/seq_ft_oldsim
J=${SLURM_JOB_ID:-local}
IFS=, read -r -a G <<< "$CUDA_VISIBLE_DEVICES"
tasks=()
for v in 1 2; do
  for var in P F; do
    for pool in pub2 priv2; do
      tasks+=("$var --mode official --pool $pool --net $v --out $REAL")
      for s in 0 1; do tasks+=("$var --mode official --pool $pool --net $v --holdout 3 --holdout-seed $s --out $OFFS"); done
    done
    for side in pub priv; do tasks+=("$var --mode oldsim --side $side --holdout-seed 0 --net $v --out $OLDS"); done
  done
done
echo "${#tasks[@]} fine-tunes on ${#G[@]} GPU(s)"
pids=()
for i in "${!G[@]}"; do
  (
    for j in "${!tasks[@]}"; do
      [ $(( j % ${#G[@]} )) -eq "$i" ] || continue
      read -r var opts <<< "${tasks[$j]}"
      aug=(); [ "$var" = F ] && aug=(--sym-aug)
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=${G[$i]} OMP_NUM_THREADS=8 python -I $R2/scripts/pid_v2_seq_finetune.py $opts --steps 0 50 100 150 --horizon 1000 \
        --ckpt-dir $S --ckpt-suffix _yse --tag "_yse$var" "${aug[@]}" || echo "FAILED: $var $opts"
    done
  ) > logs/nano4/ftsym-$J-$i.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
grep -h "FAILED\|Traceback" logs/nano4/ftsym-$J-*.log | head -5
# a variant at 100 / 150 steps as a network tag of its own: <tag><steps>_t50 -> <tag>_t<steps> (the documents and the checks read *_t50)
for d in $REAL $OFFS $OLDS; do
  for f in "$d"/*_yse[PF]_t50.npz; do
    [ -e "$f" ] || continue
    b=${f%_t50.npz}
    ln -sf "$(basename "$b")_t50.npz" "${b}50_t50.npz"
    for t in 100 150; do [ -s "${b}_t$t.npz" ] && ln -sf "$(basename "$b")_t$t.npz" "${b}${t}_t50.npz"; done
  done
done
echo "fine-tunes done: $(ls $REAL | grep -c '_yse[PF][0-9]*_t50') real, $(ls $OFFS | grep -c '_yse[PF][0-9]*_t50') pseudo-pool, $(ls $OLDS | grep -c '_yse[PF][0-9]*_t50') old-pool files"
names=()
for var in P50 P100 P150 F50 F100 F150; do
  n=v2f_$(echo "$var" | tr 'PF' 'pf')
  python3 - "$n" "$var" <<'EOF'
import json, sys
R = "/home/u2507115/BK_AICUP_2026-2"
name, var = sys.argv[1], sys.argv[2]
d = json.load(open(f"{R}/results/pid_fusion_v2w_n8.json"))
d["seq_official"] = [[f.replace("_yse_t50.npz", f"_yse{var}_t50.npz") for f in pair] for pair in d["seq_official"]]
d["composition"] = d["composition"] + [f"py fine-tuned as {var}"]
d["seq_model"] = f"v2w_n8 with the yse networks' pool fine-tune replaced by variant {var} (METHOD_NOTES 2.69)"
json.dump(d, open(f"{R}/results/pid_fusion_{name}.json", "w"), indent=1)
print(name, sum(f"_yse{var}_" in f for pair in d["seq_official"] for f in pair), "vector files replaced")
EOF
  names+=("$n")
done
pids=()
for n in "${names[@]}"; do
  python -I $R2/scripts/pid_v2_offsim.py --fusion $R2/results/pid_fusion_v2w_n8.json --alt $R2/results/pid_fusion_$n.json --steps 50 --workers 8 \
    --out $R2/results/final/v2_offsim_${n}_vs_v2w_n8.json > logs/nano4/ftsym-$J-offsim-$n.log 2>&1 &
  pids+=($!)
done
python -I $R2/scripts/pid_v2_oldsim.py --docs v2w_n8 "${names[@]}" --step 50 --out $R2/results/final/v2_oldsim_ftsym.json > logs/nano4/ftsym-$J-oldsim.log 2>&1 &
pids+=($!)
for p in "${pids[@]}"; do wait "$p" || echo "a check failed (logs/nano4/ftsym-$J-*.log)"; done
for n in "${names[@]}"; do
  echo "== pseudo-pools: $n vs v2w_n8  $(grep -E '^   talt vs t50' logs/nano4/ftsym-$J-offsim-$n.log | tail -1)"
done
echo "== old-pool simulation"
grep -E "^both:|^   v2f_.* vs v2w_n8" logs/nano4/ftsym-$J-oldsim.log | tail -7 | cut -c1-400
pids=()
for n in "${names[@]}"; do
  [ "$n" = v2f_p50 ] && continue
  bash $R2/scripts/nano4/pid_v2_make_file.sh "$n" "pid_${n}s.csv" > logs/nano4/ftsym-$J-file-$n.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "a file was not built"; done
for n in "${names[@]}"; do [ "$n" = v2f_p50 ] || echo "== file pid_${n}s.csv: $(grep -E 'questions with a different Top-1' logs/nano4/ftsym-$J-file-$n.log | head -1) | md5 $(md5sum $R2/upload/pid_${n}s.csv 2>/dev/null | cut -c1-12)"; done
