#!/usr/bin/env bash
# METHOD_NOTES 2.171 / 2.173: SHORT RUNS read on players the network has never seen, for ANY recipe (pid_exp_early.sh was
# fixed to the plain-input recipe; this one takes the base recipe, the arms and the schedule from the environment).
#   BASE="<options of every arm>"                      e.g. the o19 recipe (second feature set, by-games sampling)
#   ARMS="<name>=<options>;<name>=<options>;..."       the FIRST arm is the reference (its options may be empty: "JB=")
#   SUPJ=1                                             pass the stores of the second supervised feature set ("js")
#   SCHED="--steps 10000 --horizon 60000"              default: the first 10,000 steps of the 60,000-step cosine
#   SNAP="2500 5000 7500"                              snapshots (names k<step/1000>); "" = none
#   SEEDS="140 141"                                    two seeds per arm
#                                                      (an arm may carry its own pair: "<name>=<options> @S=182,183")
#   REF2=<arm>                                         also print every snapshot against this arm (paired)
#   TRAINER=pid_seq_train_wx.py                        another trainer of scripts/ (default pid_seq_train_w.py)
# Every snapshot is read with the decision instrument of the single-network level (official candidates of the two added
# pools, leave-games-out, vectors before the pool fine-tune; scripts/pid_v2_offcand2.py), paired with the reference arm.
# Checkpoints and vectors live in /dev/shm of the node and are removed at the end; only the readings are kept
# (results/final/v2_early_<job>_<snapshot>.json). From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj J=${SLURM_JOB_ID:-local}
  local cards i a name opt k st f p rc=0 seeds snaps sched
  local D=/dev/shm/pid_early_$J
  local TR=${TRAINER:-pid_seq_train_w.py}
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local SUP="--sup-ply $P/ply_js $O/comp $O/ext --sup-stats $P/stats_js.npz"
  sched=${SCHED:---steps 10000 --horizon 60000}
  read -r -a seeds <<< "${SEEDS:-140 141}"
  read -r -a snaps <<< "${SNAP-2500 5000 7500}"
  [ "${#seeds[@]}" -eq 2 ] || { echo "SEEDS needs two seeds"; return 2; }
  local C="$sched ${BASE:?BASE=\"<options of every arm>\"} --ema-only --out $D"
  [ "${#snaps[@]}" -gt 0 ] && C="$C --snap ${snaps[*]}"
  local arms=() lines=() names=() IFS_=$IFS
  IFS=';' read -r -a arms <<< "${ARMS:?ARMS=\"name=options;name=options\"}"; IFS=$IFS_
  IFS=, read -r -a cards <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ $((2 * ${#arms[@]})) -le "${#cards[@]}" ] || { echo "${#arms[@]} arms x 2 seeds need more than ${#cards[@]} GPUs"; return 1; }
  python -I -m py_compile $R2/scripts/$TR $R2/scripts/pid_win.py $R2/scripts/pid_winx.py $R2/scripts/pid_v2_seq_finetune.py $R2/scripts/pid_v2_offcand2.py || return 1
  mkdir -p $D/vec || return 1
  [ "$(df -BM --output=avail /dev/shm | tail -1 | tr -dc '0-9')" -ge 8000 ] || { echo "/dev/shm of $(hostname) has less than 8 GB free"; return 1; }
  k=0
  for a in "${arms[@]}"; do
    name=${a%%=*}; opt=${a#*=}; names+=("$name")
    local sd=("${seeds[@]}") w kept=""
    for w in $opt; do  # "@S=182,183" among an arm's options: its own two seeds
      case "$w" in @S=*,*) w=${w#@S=}; sd=("${w%%,*}" "${w##*,}") ;; *) kept="$kept $w" ;; esac
    done
    opt=$kept
    for i in 0 1; do
      lines+=("${cards[$k]} logs/nano4/pidseq_-1_e${name}s$i-$J.log --net -1 --tag e${name}s$i $C $opt --seed ${sd[$i]}")
      k=$((k + 1))
    done
  done
  local spec data="$OFF"
  [ "${SUPJ:-0}" = 1 ] && data="$OFF $SUP"
  spec=$(mktemp "${TMPDIR:-/tmp}/early_spec_XXXXXX")
  printf '%s\n' "${lines[@]}" > "$spec"
  cat "$spec"
  echo "node $(hostname); reference arm ${names[0]}; schedule: $sched; snapshots: ${snaps[*]:-none}"
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/$TR --fork "$spec" $data || { echo "a trainer FAILED"; rc=1; }
  rm -f "$spec"
  grep -hE "training on|supervised per-ply" logs/nano4/pidseq_-1_e*-$J.log | sort -u | cut -c1-200
  echo "checkpoints: $(ls $D | grep -c '\.pt$') ($(du -sh $D | cut -f1)) at $(date +%H:%M:%S)"
  # vectors of every checkpoint on the two added pools, before any fine-tune
  local pids=() g=0
  for f in "$D"/ckpt_-1_e*e.pt; do
    [ -e "$f" ] || continue
    st=$(basename "$f" .pt); st=${st#ckpt_-1}
    ( for p in pub2 priv2; do
        CUDA_VISIBLE_DEVICES=${cards[$g]} python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --pool $p --net -1 --horizon 1000 --ckpt-dir $D --ckpt-suffix "$st" --tag "$st" --steps 0 --out $D/vec \
          || echo "FAILED vectors of $st $p"
      done ) > logs/nano4/early-$J-vec$st.log 2>&1 &
    pids+=($!)
    g=$(( (g + 1) % ${#cards[@]} ))
    if [ "${#pids[@]}" -ge "${#cards[@]}" ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "vector files: $(ls $D/vec | grep -c '_t0\.npz$'); failures: $(cat logs/nano4/early-$J-vec*.log 2> /dev/null | grep -c FAILED)"
  local labels=() sfxs=() s
  for s in "${snaps[@]}"; do labels+=("k$((s / 1000))"); sfxs+=("k$((s / 1000))"); done
  labels+=(end); sfxs+=("")
  for i in "${!labels[@]}"; do
    local groups=() comps=()
    for name in "${names[@]}"; do
      # an arm whose trainer failed is left out instead of failing the whole reading
      k=0; for f in "$D"/vec/official_{pub2,priv2}_net-1_e${name}s{0,1}${sfxs[$i]}e_t0.npz; do [ -s "$f" ] && k=$((k + 1)); done
      [ "$k" -eq 4 ] || { echo "arm $name: only $k of 4 vector files at snapshot ${labels[$i]} - left out"; rc=1; continue; }
      groups+=("$name=$D/vec/official_{pool}_net-1_e${name}s{0|1}${sfxs[$i]}e_t0.npz"); comps+=("$name=$name")
    done
    f=${groups[0]-}; [ "${f%%=*}" = "${names[0]}" ] || { echo "the reference arm has no vectors at snapshot ${labels[$i]}"; continue; }
    echo "===== snapshot ${labels[$i]} (k<n> = step n,000; end = the last step): single network by arm, and the two-network arm against ${names[0]}"
    python -I $R2/scripts/pid_v2_offcand2.py --groups "${groups[@]}" --comps "${comps[@]}" --ref "${names[0]}" --out $R2/results/final/v2_early_${J}_${labels[$i]}.json 2>&1 | grep -vE "^\s*$" | cut -c1-400 || rc=1
    if [ -n "${REF2:-}" ] && printf '%s\n' "${comps[@]}" | grep -qx "$REF2=$REF2"; then
      echo "-- the same snapshot against $REF2"
      python -I $R2/scripts/pid_v2_offcand2.py --groups "${groups[@]}" --comps "${comps[@]}" --ref "$REF2" --out $R2/results/final/v2_early_${J}_${labels[$i]}_ref2.json 2>&1 | grep -E "^both pools" | cut -c1-200
    fi
  done
  for name in "${names[@]}"; do
    echo "== training log of arm $name (first seed): $(grep -E 'step +[0-9]+/' logs/nano4/pidseq_-1_e${name}s0-$J.log | tail -1 | cut -c1-120)"
  done
  find "$D" -delete
  return $rc
}
main "$@"; exit $?
