#!/usr/bin/env bash
# METHOD_NOTES 2.193: the family-level reading of a NEW family against its twin (the same recipe and seeds without the change)
# on the pseudo-pools of the four hold-out seeds 0-3, WITHOUT writing vectors to /home (it is nearly full):
#   1. the new family's networks are fine-tuned 50 steps on the 8 pseudo-pools (pub2 / priv2 x seeds 0-3) - the vectors go
#      to /dev/shm of the node, next to links to the seed-0..3 vectors that exist already;
#   2. <tag>_only (new family x18) is compared with <tag>_ref1 (twin x18)                          <- the reading of the rule
#      and, when a composition document that holds the twin is given, <tag>_x (that composition with the twin replaced by
#      the new family) with that document                                                          <- recorded only
#      (scripts/pid_v2_offsim_par.py: the splits at the same time, then the statistics from the stored splits).
# The documents <tag>_only / <tag>_x name real-pool vector files that do not exist yet (only their names are used here).
#   [LINKS="<family>:<seed>=<checkpoint file> ..."] pid_exp_shmfam.sh <new family, e.g. o40F> <twin, e.g. o38F> <tag, e.g. v2o40m> [<composition document>]
# Results: results/final/v2_offsim_<tag>_only_vs_<tag>_ref1_s0_1_2_3.json (and ..._<tag>_x_vs_<document>_...).
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 SN=/work/u2507115/pid_v2/seqnet J=${SLURM_JOB_ID:-local} new=$1 twin=$2 tag=$3 comp=${4:-}
  local O=$R2/data/pid_v2_seq_ft_offsim
  local D=/dev/shm/shmfam_$J G i j k f l fam seed src pool s t tasks=() pids=() miss rc=0 pair ref alt pairs=() n
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  for l in ${LINKS:-}; do
    fam=${l%%:*}; seed=${l#*:}; src=${seed#*=}; seed=${seed%%=*}
    if [ -s "$SN/$src" ]; then ln -sf "$src" "$SN/ckpt_-1_${fam}s${seed}e.pt"; echo "link ckpt_-1_${fam}s${seed}e.pt -> $src"; else echo "missing $src (no link)"; fi
  done
  n=0; for k in 0 1 2 3 4 5 6 7; do [ -s $SN/ckpt_-1_${new}s${k}e.pt ] && n=$((n + 1)); done
  [ "$n" -ge 7 ] || { echo "family $new has only $n networks"; return 1; }
  # the documents
  python -I $R2/data/day_scripts/build_v2h.py ${tag}_ref1 "${twin}*18" || return 1
  python3 -I - "$R2/results" "$tag" "$twin" "$new" "$comp" <<'PY' || return 1
import json, sys
res, tag, twin, new, comp = sys.argv[1:6]
def swap(src, dst, note):
    d = json.load(open(f"{res}/pid_fusion_{src}.json"))
    old = json.dumps(d["seq_official"])
    k = old.count(f"_{twin}s")
    assert k > 0, f"{src} holds no network of {twin}"
    d["seq_official"] = json.loads(old.replace(f"_{twin}s", f"_{new}s"))
    d["composition"] = [c.replace(twin + "*", new + "*") for c in d.get("composition", [])]
    d["note"] = note
    d["seq_model"] = f"{src} with the family {twin} replaced by {new} (pid_exp_shmfam.sh; pseudo-pool reading only)"
    json.dump(d, open(f"{res}/pid_fusion_{dst}.json", "w"), indent=1)
    print(f"{dst}: {src} with {k} file names of {twin} -> {new}; composition {'+'.join(d['composition'])}")
swap(tag + "_ref1", tag + "_only", "pseudo-pool reading only: the real-pool vector files named here may not exist")
if comp:
    swap(comp, tag + "_x", "pseudo-pool reading only: the real-pool vector files named here may not exist")
PY
  pairs=("${tag}_ref1:${tag}_only")
  [ -n "$comp" ] && pairs+=("$comp:${tag}_x")
  mkdir -p $D/ft || return 1
  for f in $O/offsim_*_h3s[0-3]_net*_t50.npz; do ln -s "$f" $D/ft/ 2> /dev/null; done
  echo "linked $(ls $D/ft | wc -l) existing seed-0..3 vectors"
  for k in 0 1 2 3 4 5 6 7; do
    [ -s $SN/ckpt_-1_${new}s${k}e.pt ] || continue
    for pool in pub2 priv2; do for s in 0 1 2 3; do
      [ -e $D/ft/offsim_${pool}_h3s${s}_net-1_${new}s${k}e_t50.npz ] || tasks+=("$pool $s _${new}s${k}e")
    done; done
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
    ) > logs/nano4/shmfam-$J-ft-$i.log 2>&1 &
    pids+=($!)
  done
  for i in "${pids[@]}"; do wait "$i" || rc=1; done
  echo "fine-tunes done at $(date +%T); failures: $(cat logs/nano4/shmfam-$J-ft-*.log | grep -c FAILED)"
  for pair in "${pairs[@]}"; do
    ref=${pair%%:*}; alt=${pair#*:}
    miss=$(python3 -c "
import json, re, os, sys
n = 0
for d in sys.argv[2:]:
    for x in json.load(open('$R2/results/pid_fusion_' + d + '.json'))['seq_official']:
        v, tg = re.fullmatch(r'official_pub2_net(-?\d+)(.*)_t\d+\.npz', os.path.basename(x[0])).groups()
        for pool in ('pub2', 'priv2'):
            for s in (0, 1, 2, 3):
                n += not os.path.exists(f'{sys.argv[1]}/offsim_{pool}_h3s{s}_net{v}{tg}_t50.npz')
print(n)" $D/ft "$ref" "$alt")
    echo "vector files missing for $ref / $alt on seeds 0-3: $miss"
    [ "$miss" = 0 ] || { find $D -delete 2> /dev/null; return 1; }
  done
  pids=(); i=0
  for pair in "${pairs[@]}"; do
    ref=${pair%%:*}; alt=${pair#*:}
    mkdir -p $D/dump_$alt
    for pool in pub2 priv2; do for s in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=${G[$(( i % ${#G[@]} ))]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_v2_offsim_par.py --fusion $R2/results/pid_fusion_$ref.json \
        --alt $R2/results/pid_fusion_$alt.json --ft-dir $D/ft --steps 50 --seeds 0 1 2 3 --workers 6 --only $pool:$s --dump $D/dump_$alt --out /dev/null > logs/nano4/shmfam-$J-$alt-$pool-$s.log 2>&1 &
      pids+=($!); i=$((i + 1))
    done; done
  done
  for i in "${pids[@]}"; do wait "$i" || { echo "a split failed (logs/nano4/shmfam-$J-*.log)"; rc=1; }; done
  echo "splits done at $(date +%T)"
  for pair in "${pairs[@]}"; do
    ref=${pair%%:*}; alt=${pair#*:}
    echo "== pseudo-pools, four seeds: $alt vs $ref"
    python -I $R2/scripts/pid_v2_offsim_par.py --fusion $R2/results/pid_fusion_$ref.json --alt $R2/results/pid_fusion_$alt.json --ft-dir $D/ft --steps 50 --seeds 0 1 2 3 --from-dumps --dump $D/dump_$alt \
      --out $R2/results/final/v2_offsim_${alt}_vs_${ref}_s0_1_2_3.json > logs/nano4/shmfam-$J-$alt-stat.log 2>&1 || rc=1
    grep -E '^(pub2|priv2|both): document weights|^   talt vs t50' logs/nano4/shmfam-$J-$alt-stat.log | cut -c1-240
  done
  find $D -delete 2> /dev/null
  echo "done at $(date +%T), rc $rc"
  return $rc
}
main "$@"; exit $?
