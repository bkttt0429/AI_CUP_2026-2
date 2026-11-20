#!/usr/bin/env bash
# METHOD_NOTES 2.171: SHORT RUNS read on players the network has never seen - can a quarter of a training tell what the whole
# training will say? Eight single-GPU runs of 10,000 steps (the first quarter of the 40,000-step schedule: same horizon, so a
# run with the seed of a full training repeats that training's first quarter), snapshots at 2,500 / 5,000 / 7,500 steps; every
# snapshot is read with the decision instrument of the single-network level (official candidates, leave-games-out, vectors
# before the pool fine-tune; scripts/pid_v2_offcand2.py).
#   pid_exp_early.sh calib     two seeds (180, 181) of: B = o22S recipe (baseline), W = + window input (o31S; full run ends
#                              about 07:00), L = 512x10 (final known: o24 vs o22 -0.005), F = one competition fold fewer
#                              (final known: o26 vs o22 -0.018)
#   ARMS="<name>=<options> ..." pid_exp_early.sh arms     any other arms against B (B is always run); names one letter/word
# Checkpoints and vectors live in /dev/shm of the node and are removed at the end; only the readings are kept
# (results/final/v2_early_<job>_<step>.json). From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox mode=${1:-calib} J=${SLURM_JOB_ID:-local} cards i a name opt k st f p rc=0
  local D=/dev/shm/pid_early_$J
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv"
  local C="--steps 10000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --ema-only --own-by-games linear --snap 2500 5000 7500 --out $D"
  local arms=("B=") lines=() names=()
  case "$mode" in
    calib) arms+=("W=--shape-win 9" "L=--arch 512,10,8,2048" "F=--drop-fold 4") ;;
    arms) local IFS_=$IFS; IFS=';' read -r -a extra_arms <<< "${ARMS:?ARMS=\"name=options;name=options\"}"; IFS=$IFS_; arms+=("${extra_arms[@]}") ;;
    *) echo "usage: pid_exp_early.sh calib | arms"; return 2 ;;
  esac
  IFS=, read -r -a cards <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ $((2 * ${#arms[@]})) -le "${#cards[@]}" ] || { echo "${#arms[@]} arms x 2 seeds need more than ${#cards[@]} GPUs"; return 1; }
  python -I -m py_compile $R2/scripts/pid_seq_train_w.py $R2/scripts/pid_win.py $R2/scripts/pid_v2_seq_finetune.py $R2/scripts/pid_v2_offcand2.py || return 1
  mkdir -p $D/vec || return 1
  [ "$(df -BM --output=avail /dev/shm | tail -1 | tr -dc '0-9')" -ge 8000 ] || { echo "/dev/shm of $(hostname) has less than 8 GB free"; return 1; }
  k=0
  for a in "${arms[@]}"; do
    name=${a%%=*}; opt=${a#*=}; names+=("$name")
    for i in 0 1; do
      lines+=("${cards[$k]} logs/nano4/pidseq_-1_e${name}s$i-$J.log --net -1 --tag e${name}s$i $C $opt --seed $((180 + i))")
      k=$((k + 1))
    done
  done
  local spec
  spec=$(mktemp "${TMPDIR:-/tmp}/early_spec_XXXXXX")
  printf '%s\n' "${lines[@]}" > "$spec"
  cat "$spec"
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_seq_train_w.py --fork "$spec" $OFF || { echo "a trainer FAILED"; rc=1; }
  rm -f "$spec"
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
  for st in k2 k5 k7 end; do
    local sfx=$st groups=() comps=()
    [ "$st" = end ] && sfx=""
    for name in "${names[@]}"; do
      groups+=("$name=$D/vec/official_{pool}_net-1_e${name}s{0|1}${sfx}e_t0.npz"); comps+=("$name=$name")
    done
    echo "===== snapshot $st (steps: k2 = 2,500, k5 = 5,000, k7 = 7,500, end = 10,000): single network by arm, and the two-network arm against B"
    python -I $R2/scripts/pid_v2_offcand2.py --groups "${groups[@]}" --comps "${comps[@]}" --ref B --out $R2/results/final/v2_early_${J}_$st.json 2>&1 | grep -vE "^\s*$" | cut -c1-400 || rc=1
  done
  for name in "${names[@]}"; do
    echo "== training log of arm $name (seed 180): $(grep -E 'step +10000/' logs/nano4/pidseq_-1_e${name}s0-$J.log | cut -c1-120)"
    grep -E 'step (5000|10000) \[ema\]' logs/nano4/pidseq_-1_e${name}s0-$J.log | cut -c1-160
  done
  find "$D" -delete
  return $rc
}
main "$@"; exit $?
