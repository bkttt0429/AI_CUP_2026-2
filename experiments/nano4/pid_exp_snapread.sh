#!/usr/bin/env bash
# METHOD_NOTES 2.177: read checkpoints - snapshots of a running training or finished networks - on the official candidates
# of the two added pools (1-game and 3-game questions, leave-games-out, vectors BEFORE the pool fine-tune; the
# single-network instrument scripts/pid_v2_offcand2.py), without writing anything into the vector stores.
#   pid_exp_snapread.sh <label> <group>=<tag pattern> [<group>=<tag pattern> ...]
#     tag pattern: the checkpoint's tag, alternatives in braces - o34As{0|1}k10e reads ckpt_-1_o34As0k10e.pt and
#                  ckpt_-1_o34As1k10e.pt of /work/u2507115/pid_v2/seqnet (a single network: no braces)
#     the first group is the reference of the two-network comparison; a group with a missing checkpoint is left out
# Vectors go to /dev/shm of the node and are removed at the end; the reading is results/final/v2_snapread_<label>.json.
# From the AI_CUP_2026-1 checkout, one GPU is enough (more are used if given). Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 S=/work/u2507115/pid_v2/seqnet J=${SLURM_JOB_ID:-local} label=${1:?label} rc=0
  shift
  local D=/dev/shm/pid_snap_$J cards a name pat tags t p g=0 ok pids=() groups=() comps=() names=() todo=()
  IFS=, read -r -a cards <<< "${CUDA_VISIBLE_DEVICES:?}"
  python -I -m py_compile $R2/scripts/pid_v2_seq_finetune.py $R2/scripts/pid_v2_offcand2.py $R2/scripts/pid_win.py || return 1
  mkdir -p $D || return 1
  for a in "$@"; do
    name=${a%%=*}; pat=${a#*=}
    case "$pat" in *[!A-Za-z0-9_{}\|]*) echo "group $name: unexpected characters in '$pat'"; return 2 ;; esac
    tags=$(eval echo "$(printf '%s' "$pat" | tr '|' ',')")
    ok=1
    for t in $tags; do [ -s "$S/ckpt_-1_$t.pt" ] || { echo "group $name: no checkpoint ckpt_-1_$t.pt - group left out"; ok=0; }; done
    [ "$ok" = 1 ] || continue
    names+=("$name"); groups+=("$name=$D/official_{pool}_net-1_${pat}_t0.npz"); comps+=("$name=$name")
    for t in $tags; do todo+=("$t"); done
  done
  [ "${#names[@]}" -ge 1 ] || { echo "no group has its checkpoints"; rmdir $D; return 1; }
  for t in $(printf '%s\n' "${todo[@]}" | sort -u); do
    ( for p in pub2 priv2; do
        CUDA_VISIBLE_DEVICES=${cards[$g]} python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --pool $p --net -1 --horizon 1000 --ckpt-dir $S --ckpt-suffix "_$t" --tag "_$t" --steps 0 --out $D \
          || echo "FAILED vectors of $t $p"
      done ) > logs/nano4/snapread-$J-$t.log 2>&1 &
    pids+=($!)
    g=$(( (g + 1) % ${#cards[@]} ))
    if [ "${#pids[@]}" -ge "${#cards[@]}" ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "vector files: $(ls $D | grep -c '_t0\.npz$'); failures: $(cat logs/nano4/snapread-$J-*.log 2> /dev/null | grep -c FAILED)"
  echo "===== $label: single network by group (mean of its networks), and each group against ${names[0]}"
  python -I $R2/scripts/pid_v2_offcand2.py --groups "${groups[@]}" --comps "${comps[@]}" --ref "${names[0]}" --out $R2/results/final/v2_snapread_$label.json 2>&1 | grep -vE "^\s*$|^group " | cut -c1-600 || rc=1
  find "$D" -delete
  return $rc
}
main "$@"; exit $?
