#!/usr/bin/env bash
# METHOD_NOTES 2.191: a candidate composition on the hold-out seeds 4 and 5 (never used to choose anything; the final's own
# replication of notes 2.155 used them the same way), without writing vectors to /home:
#   1. the networks of the given NEW families are fine-tuned 50 steps on the pseudo-pools of seeds 4 and 5 - vectors go to
#      /dev/shm of the node, next to links to the seed-4/5 vectors that exist already (the reference's families);
#   2. the candidate is compared with the reference on those two seeds (scripts/pid_v2_offsim_par.py, --ft-dir = that folder).
#   pid_exp_fresh45.sh <reference document> <candidate document> <new family> [<new family> ...]
# Result: results/final/v2_offsim_<candidate>_vs_<reference>_s4_5.json. From the AI_CUP_2026-1 checkout, 8 GPUs.
# Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 SN=/work/u2507115/pid_v2/seqnet J=${SLURM_JOB_ID:-local} ref=$1 alt=$2
  local O=$R2/data/pid_v2_seq_ft_offsim
  shift 2
  local D=/dev/shm/ft45_$J G i j k f fam pool s t tasks=() pids=() need miss rc=0
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  mkdir -p $D/ft $D/dump || return 1
  for f in $O/offsim_*_h3s[45]_net*_t50.npz; do ln -s "$f" $D/ft/; done
  echo "linked $(ls $D/ft | wc -l) existing seed-4/5 vectors"
  for fam in "$@"; do
    for k in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
      [ -s $SN/ckpt_-1_${fam}s${k}e.pt ] || continue
      for pool in pub2 priv2; do for s in 4 5; do
        [ -e $D/ft/offsim_${pool}_h3s${s}_net-1_${fam}s${k}e_t50.npz ] || tasks+=("$pool $s _${fam}s${k}e")
      done; done
    done
  done
  echo "${#tasks[@]} fine-tunes on ${#G[@]} GPUs, start $(date +%T)"
  for i in "${!G[@]}"; do
    (
      for j in "${!tasks[@]}"; do
        [ $(( j % ${#G[@]} )) -eq "$i" ] || continue
        read -r pool s t <<< "${tasks[$j]}"
        CUDA_VISIBLE_DEVICES=${G[$i]} python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --pool "$pool" --net -1 --horizon 1000 --ckpt-dir $SN --ckpt-suffix "$t" --tag "$t" \
          --steps 50 --holdout 3 --holdout-seed "$s" --out $D/ft || echo "FAILED: $t $pool seed $s"
      done
    ) > logs/nano4/fresh45-$J-ft-$i.log 2>&1 &
    pids+=($!)
  done
  for i in "${pids[@]}"; do wait "$i" || rc=1; done
  echo "fine-tunes done at $(date +%T); failures: $(cat logs/nano4/fresh45-$J-ft-*.log | grep -c FAILED)"
  miss=$(python3 -c "
import json, re, os, sys
n = 0
for d in sys.argv[2:]:
    for x in json.load(open('$R2/results/pid_fusion_' + d + '.json'))['seq_official']:
        v, tg = re.fullmatch(r'official_pub2_net(-?\d+)(.*)_t\d+\.npz', os.path.basename(x[0])).groups()
        for pool in ('pub2', 'priv2'):
            for s in (4, 5):
                n += not os.path.exists(f'{sys.argv[1]}/offsim_{pool}_h3s{s}_net{v}{tg}_t50.npz')
print(n)" $D/ft "$ref" "$alt")
  echo "vector files missing for the two documents on seeds 4 and 5: $miss"
  [ "$miss" = 0 ] || { find $D -delete 2> /dev/null; return 1; }
  pids=(); i=0
  for pool in pub2 priv2; do for s in 4 5; do
    CUDA_VISIBLE_DEVICES=${G[$(( i % ${#G[@]} ))]} OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 python -I $R2/scripts/pid_v2_offsim_par.py --fusion $R2/results/pid_fusion_$ref.json \
      --alt $R2/results/pid_fusion_$alt.json --ft-dir $D/ft --steps 50 --seeds 4 5 --workers 6 --only $pool:$s --dump $D/dump --out /dev/null > logs/nano4/fresh45-$J-split-$pool-$s.log 2>&1 &
    pids+=($!); i=$((i + 1))
  done; done
  for i in "${pids[@]}"; do wait "$i" || { echo "a split failed (logs/nano4/fresh45-$J-split-*.log)"; rc=1; }; done
  echo "== pseudo-pools, FRESH hold-out seeds 4 5: $alt vs $ref"
  python -I $R2/scripts/pid_v2_offsim_par.py --fusion $R2/results/pid_fusion_$ref.json --alt $R2/results/pid_fusion_$alt.json --ft-dir $D/ft --steps 50 --seeds 4 5 --from-dumps --dump $D/dump \
    --out $R2/results/final/v2_offsim_${alt}_vs_${ref}_s4_5.json > logs/nano4/fresh45-$J-stat.log 2>&1 || rc=1
  grep -E '^(pub2|priv2|both): document weights|^   talt vs t50' logs/nano4/fresh45-$J-stat.log | cut -c1-240
  find $D -delete 2> /dev/null
  echo "done at $(date +%T), rc $rc"
  return $rc
}
main "$@"; exit $?
