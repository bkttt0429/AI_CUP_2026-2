#!/usr/bin/env bash
# METHOD_NOTES 2.201: a verdict chain that writes NO vector to /home (it has no room for further families): every vector
# that is missing goes to /dev/shm of the node, next to links to the vectors that exist. One task on one node (8 GPUs,
# single-GPU processes only: also runs on a node whose multi-GPU check fails).
#   pid_exp_f0_chain3.sh <reference document> "<new families>" "<family>:<twin> ..." <name>=<composition> ...
#     <new families>   families without real-pool vectors on /home (their members = the linked checkpoints); "-" for none.
#                      o44F-o47F get their member links here (the tags of scripts/nano4/pid_exp_f0_train.sh).
#     <family>:<twin>  recorded only: the family alone (x18) against its twin alone, pseudo-pools of seeds 0-3 and old pools
#   1. documents (scripts/pid_build_doc_names.py): the candidates; f0p_<family> for every family of a pair
#   2. links to the existing vectors; fine-tunes of what the documents need and is missing (PER_GPU processes per GPU,
#      default ONE: a fine-tune holds up to ~100 GB of a card; three per card ran out of memory in the trial of
#      pid_exp_f0_chain.sh on node 172, 2026-11-18 01:10, and the node's GPU driver hung afterwards):
#      pseudo-pools of the seeds 0-3 and 4 5 (candidates and reference; pairs: seeds 0-3), old-pool simulation
#   3. pseudo-pools, seeds 0-3: every candidate against the reference, every pair
#   4. old-pool simulation (50 steps): reference + candidates; every pair
#   5. pseudo-pools, FRESH seeds 4 5: every candidate against the reference      (FRESH=0 leaves the seeds 4 5 out)
# Results: results/final/v2_offsim_<alt>_vs_<ref>_s0_1_2_3.json, ..._s4_5.json, v2_oldsim_f0_<alt>.json. Nothing here
# chooses a file or writes one: the rule is in the notes. SHM_NAME=<name>: the folder /dev/shm/f0c_<name> (default: the
# step's id); KEEP=1 leaves it for a later step on the same node.
# pid_exp_f0_chain3.sh = pid_exp_f0_chain2.sh (one fine-tune per GPU, stop after failed fine-tunes) with:
#   VEC_DIR=<folder on a shared file system>  the vectors go there instead of /dev/shm, so that several tasks on
#       DIFFERENT nodes add to the same store (one task per family as soon as it is trained, then the verdict with
#       nothing left to fine-tune). The folder is never deleted by this script; two tasks must not fine-tune the same
#       network at the same time (give every task its own families). Scratch files: <folder>/run_<step id>, removed.
#   PREP=1   the families of the PAIRS also get the vectors of the fresh seeds 4 5 (they are used by the later verdict).
#   no <name>=<composition> at all: only the vectors and the pairs.
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 SN=/work/u2507115/pid_v2/seqnet J=${SLURM_JOB_ID:-local}
  local O=$R2/data/pid_v2_seq_ft_offsim OLD=$R2/data/pid_v2_seq_ft_oldsim ref=$1 newfams=$2 pairs=$3
  shift 3
  local D=${VEC_DIR:-/dev/shm/f0c_${SHM_NAME:-$J}} T PG=${PER_GPU:-1} prep=${PREP:-0} G W w j i k f n c p a b rc=0 names=() pdocs=() tasks=() pids=() fresh=${FRESH:-1} t0=$SECONDS
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ "$newfams" = "-" ] && newfams=""
  [ -s $R2/results/pid_fusion_$ref.json ] || { echo "no document $ref"; return 1; }
  for f in $newfams; do  # member links of the fold-0 families
    case "$f" in
      o44F) for i in 0 1 2 3; do [ -s $SN/ckpt_-1_o44Ss${i}e.pt ] && ln -sf ckpt_-1_o44Ss${i}e.pt $SN/ckpt_-1_o44Fs${i}e.pt
                                 [ -s $SN/ckpt_-1_o44SMs${i}e.pt ] && ln -sf ckpt_-1_o44SMs${i}e.pt $SN/ckpt_-1_o44Fs$((i + 4))e.pt; done ;;
      o45F|o46F|o47F) for i in 0 1 2 3 4 5; do [ -s $SN/ckpt_-1_${f%F}As${i}e.pt ] && ln -sf ckpt_-1_${f%F}As${i}e.pt $SN/ckpt_-1_${f}s${i}e.pt; done
                      for i in 0 1; do [ -s $SN/ckpt_-1_${f%F}Cs${i}e.pt ] && ln -sf ckpt_-1_${f%F}Cs${i}e.pt $SN/ckpt_-1_${f}s$((i + 6))e.pt; done ;;
    esac
    n=0; for k in 0 1 2 3 4 5 6 7; do [ -s $SN/ckpt_-1_${f}s${k}e.pt ] && n=$((n + 1)); done
    echo "family $f: $n linked checkpoints"
    [ "$n" -ge 7 ] || { echo "family $f has only $n networks: not started"; return 1; }
  done
  # 1. documents
  for c in "$@"; do
    # shellcheck disable=SC2086
    python3 -I $R2/scripts/pid_build_doc_names.py "${c%%=*}" "${c#*=}" $newfams || return 1
    names+=("${c%%=*}")
  done
  for p in $pairs; do
    for f in ${p%%:*} ${p#*:}; do
      # shellcheck disable=SC2086
      python3 -I $R2/scripts/pid_build_doc_names.py "f0p_$f" "$f*18" $newfams || return 1
      pdocs+=("f0p_$f")
    done
  done
  # 2. vectors
  T=$D/run_$J
  mkdir -p $D/ft $D/old $T || return 1
  # links only for the names that are missing (another task may be reading the folder: nothing is replaced)
  find $O -maxdepth 1 -name 'offsim_*_h3s[0-5]_net*_t50.npz' -exec sh -c 'd=$0; for f; do [ -e "$d/${f##*/}" ] || ln -s "$f" "$d/"; done' $D/ft {} +
  find $OLD -maxdepth 1 -name 'oldsim_*_s0_net*_t50.npz' -exec sh -c 'd=$0; for f; do [ -e "$d/${f##*/}" ] || ln -s "$f" "$d/"; done' $D/old {} +
  echo "folder $D on $(hostname): $(ls $D/ft | wc -l) pseudo-pool vectors, $(ls $D/old | wc -l) old-pool vectors linked or kept; $(date +%T)"
  python3 -I - "$R2" "$D" "$ref ${names[*]-}" "${pdocs[*]-}" "$fresh" "$prep" > $T/tasks.txt <<'PY' || return 1
import json, os, re, sys
R, D, full, pair, fresh, prep = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4].split(), sys.argv[5] == "1", sys.argv[6] == "1"
def nets(d):
    return {re.fullmatch(r"official_pub2_net(-?\d+)(.*)_t\d+\.npz", os.path.basename(x[0])).groups() for x in json.load(open(f"{R}/results/pid_fusion_{d}.json"))["seq_official"]}
off, old = set(), set()
for docs, seeds in ((full, range(6) if fresh else range(4)), (pair, range(6) if prep else range(4))):
    for d in docs:
        for v, tg in nets(d):
            off |= {(pool, s, v, tg) for pool in ("pub2", "priv2") for s in seeds if not os.path.exists(f"{D}/ft/offsim_{pool}_h3s{s}_net{v}{tg}_t50.npz")}
            old |= {(side, v, tg) for side in ("pub", "priv") if not os.path.exists(f"{D}/old/oldsim_{side}_s0_net{v}{tg}_t50.npz")}
for x in sorted(off, key=lambda x: (x[3], x[1], x[0])):
    print("off", *x)
for x in sorted(old, key=lambda x: (x[2], x[0])):
    print("old", *x)
PY
  mapfile -t tasks < $T/tasks.txt
  echo "${#tasks[@]} fine-tunes to do ($(grep -c '^off' $T/tasks.txt) pseudo-pool, $(grep -c '^old' $T/tasks.txt) old-pool) on ${#G[@]} GPUs x $PG processes, start $(date +%T)"
  while read -r k a b c f; do  # every network to fine-tune must have its checkpoint
    [ "$k" = off ] && { [ -s "$SN/ckpt_${c}${f}.pt" ] || { echo "no checkpoint ckpt_${c}${f}.pt for a missing vector"; return 1; }; }
    [ "$k" = old ] && { [ -s "$SN/ckpt_${b}${c}.pt" ] || { echo "no checkpoint ckpt_${b}${c}.pt for a missing vector"; return 1; }; }
  done < $T/tasks.txt
  W=$(( ${#G[@]} * PG ))
  for w in $(seq 0 $((W - 1))); do
    (
      for j in "${!tasks[@]}"; do
        [ $(( j % W )) -eq "$w" ] || continue
        read -r k a b c f <<< "${tasks[$j]}"
        if [ "$k" = off ]; then
          CUDA_VISIBLE_DEVICES=${G[$(( w % ${#G[@]} ))]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --pool "$a" --net "$c" --horizon 1000 --ckpt-dir $SN --ckpt-suffix "$f" --tag "$f" \
            --steps 50 --holdout 3 --holdout-seed "$b" --out $D/ft < /dev/null || echo "FAILED: ${tasks[$j]}"
        else
          CUDA_VISIBLE_DEVICES=${G[$(( w % ${#G[@]} ))]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -I $R2/scripts/pid_v2_seq_finetune.py --mode oldsim --side "$a" --holdout-seed 0 --net "$b" --steps 0 50 --horizon 1000 \
            --out $D/old --ckpt-dir $SN --ckpt-suffix "$c" --tag "$c" < /dev/null || echo "FAILED: ${tasks[$j]}"
        fi
      done
    ) > logs/nano4/f0c-$J-ft-$w.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  n=$(cat logs/nano4/f0c-$J-ft-*.log | grep -c FAILED)
  echo "fine-tunes done at $(date +%T) ($(( (SECONDS - t0) / 60 )) min); failures: $n"
  [ "$n" = 0 ] || { grep -h FAILED logs/nano4/f0c-$J-ft-*.log | head -5; echo "fine-tunes failed: the comparisons are not started (folder $D kept)"; find $T -delete 2> /dev/null; return 1; }
  cmp2() {  # <reference> <alternative> "<seeds>" <first GPU>: the splits at the same time, then the statistics
    local r=$1 al=$2 sd=$3 g0=$4 sfx=${3// /_} pool s q=() x=0 e=0
    mkdir -p $T/dump_${al}_$sfx
    for pool in pub2 priv2; do for s in $sd; do
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=${G[$(( (g0 + x) % ${#G[@]} ))]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -I $R2/scripts/pid_v2_offsim_par.py --fusion $R2/results/pid_fusion_$r.json \
        --alt $R2/results/pid_fusion_$al.json --ft-dir $D/ft --steps 50 --seeds $sd --workers 6 --only $pool:$s --dump $T/dump_${al}_$sfx --out /dev/null > logs/nano4/f0c-$J-split-$al-$sfx-$pool-$s.log 2>&1 &
      q+=($!); x=$((x + 1))
    done; done
    for x in "${q[@]}"; do wait "$x" || e=1; done
    # shellcheck disable=SC2086
    python -I $R2/scripts/pid_v2_offsim_par.py --fusion $R2/results/pid_fusion_$r.json --alt $R2/results/pid_fusion_$al.json --ft-dir $D/ft --steps 50 --seeds $sd --from-dumps --dump $T/dump_${al}_$sfx \
      --out $R2/results/final/v2_offsim_${al}_vs_${r}_s$sfx.json > logs/nano4/f0c-$J-stat-$al-$sfx.log 2>&1 || e=1
    echo "== pseudo-pools, seeds $sd: $al vs $r ($(date +%T))$([ $e = 0 ] || echo '  FAILED (logs/nano4/f0c-'$J'-s*-'$al'-*.log)')"
    grep -E '^(pub2|priv2|both): document weights|^   talt vs t50' logs/nano4/f0c-$J-stat-$al-$sfx.log | cut -c1-240
    find $T/dump_${al}_$sfx -delete 2> /dev/null
    return $e
  }
  oldsim() {  # <result name> <reference> <alternatives...>
    local nm=$1
    shift
    echo "== old-pool simulation, 50 steps: $* ($(date +%T))"
    python -I $R2/scripts/pid_v2_oldsim.py --docs "$@" --step 50 --ft-dir $D/old --out $R2/results/final/v2_oldsim_f0_$nm.json 2>&1 | grep -vE '^\s*$' | tail -n 30 | cut -c1-240
    return "${PIPESTATUS[0]}"
  }
  # 3. + 4. seeds 0-3 and the old pools, two comparisons at a time
  pids=(); i=0
  for n in ${names[@]+"${names[@]}"}; do
    cmp2 "$ref" "$n" "0 1 2 3" $(( 4 * i )) > $T/out_cand_$n.txt 2>&1 &
    pids+=($!); i=$((i + 1))
    if [ $(( i % 2 )) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p" || rc=1; done; pids=(); fi
  done
  for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || rc=1; done
  for n in ${names[@]+"${names[@]}"}; do cat $T/out_cand_$n.txt; done
  if [ "${#names[@]}" -gt 0 ]; then oldsim "${names[0]}" "$ref" "${names[@]}" || { echo "old-pool simulation failed"; rc=1; }; fi
  pids=(); i=0
  for p in $pairs; do
    cmp2 "f0p_${p#*:}" "f0p_${p%%:*}" "0 1 2 3" $(( 4 * i )) > $T/out_pair_${p%%:*}.txt 2>&1 &
    pids+=($!); i=$((i + 1))
    if [ $(( i % 2 )) -eq 0 ]; then for w in "${pids[@]}"; do wait "$w" || rc=1; done; pids=(); fi
  done
  for w in ${pids[@]+"${pids[@]}"}; do wait "$w" || rc=1; done
  for p in $pairs; do
    cat $T/out_pair_${p%%:*}.txt
    oldsim "pair_${p%%:*}" "f0p_${p#*:}" "f0p_${p%%:*}" || { echo "old-pool simulation of the pair $p failed"; rc=1; }
  done
  # 5. fresh seeds
  if [ "$fresh" = 1 ] && [ "${#names[@]}" -gt 0 ]; then
    pids=(); i=0
    for n in "${names[@]}"; do
      cmp2 "$ref" "$n" "4 5" $(( 4 * i )) > $T/out_fresh_$n.txt 2>&1 &
      pids+=($!); i=$((i + 1))
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    echo "===== FRESH hold-out seeds 4 and 5"
    for n in "${names[@]}"; do cat $T/out_fresh_$n.txt; done
  fi
  find $T -delete 2> /dev/null
  if [ -n "${VEC_DIR:-}" ] || [ "${KEEP:-0}" = 1 ]; then echo "folder $D kept ($(du -sh $D | cut -f1); own files $(find $D/ft $D/old -type f | wc -l))"; else find $D -delete 2> /dev/null; fi
  echo "done at $(date +%T) ($(( (SECONDS - t0) / 60 )) min), rc $rc"
  return $rc
}
main "$@"; exit $?
