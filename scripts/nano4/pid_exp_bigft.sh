#!/usr/bin/env bash
# METHOD_NOTES 2.204: the pool fine-tunes that do not fit ONE 140 GB card. The 512x10 networks with the second window
# (families o43F, o46F) need more than 139 GB for an episode of 256 players x 8 games: the real pools and the old-pool
# simulation fail with "CUDA out of memory" (vec_o43b of job 793567, 2026-11-18 03:33), while the pseudo-pools (7 games per
# candidate) just fit. Here every such fine-tune gets TWO cards (pid_v2_seq_finetune.py --dp: the training forward is split
# over them, the loss stays on the first), four fine-tunes at a time. Options as the standard chains
# (pid_v2_ft_many2.sh real / pid_exp_oldsim.sh); files that exist are skipped.
#   pid_exp_bigft.sh <family> <real|old> [<out dir>]
#     real  official_<pool>_net-1_<family>s<k>e_t{0,50}.npz     default out dir: data/seq_ft
#     old   oldsim_<side>_s0_net-1_<family>s<k>e_t{0,50}.npz    default out dir: data/pid_v2_seq_ft_oldsim
# The member links of o44F-o47F are made here when they are missing (the tags of scripts/nano4/pid_exp_f0_train.sh).
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 SN=/work/u2507115/pid_v2/seqnet fam=$1 what=$2 J=${SLURM_JOB_ID:-local}
  local out G i k w n t a f free rc=0 tags=() tasks=() pids=() t0=$SECONDS
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ "${#G[@]}" -eq 8 ] || { echo "expected eight GPUs"; return 1; }
  case "$what" in
    real) out=${3:-$R2/data/seq_ft} ;;
    old)  out=${3:-$R2/data/pid_v2_seq_ft_oldsim} ;;
    *) echo "usage: pid_exp_bigft.sh <family> <real|old> [<out dir>]"; return 2 ;;
  esac
  case "$fam" in
    o44F) for i in 0 1 2 3; do [ -s $SN/ckpt_-1_o44Ss${i}e.pt ] && ln -sf ckpt_-1_o44Ss${i}e.pt $SN/ckpt_-1_o44Fs${i}e.pt
                               [ -s $SN/ckpt_-1_o44SMs${i}e.pt ] && ln -sf ckpt_-1_o44SMs${i}e.pt $SN/ckpt_-1_o44Fs$((i + 4))e.pt; done ;;
    o45F|o46F|o47F) for i in 0 1 2 3 4 5; do [ -s $SN/ckpt_-1_${fam%F}As${i}e.pt ] && ln -sf ckpt_-1_${fam%F}As${i}e.pt $SN/ckpt_-1_${fam}s${i}e.pt; done
                    for i in 0 1; do [ -s $SN/ckpt_-1_${fam%F}Cs${i}e.pt ] && ln -sf ckpt_-1_${fam%F}Cs${i}e.pt $SN/ckpt_-1_${fam}s$((i + 6))e.pt; done ;;
  esac
  for k in $(seq 0 23); do [ -s $SN/ckpt_-1_${fam}s${k}e.pt ] && tags+=("_${fam}s${k}e"); done
  [ "${#tags[@]}" -ge 7 ] || { echo "family $fam: only ${#tags[@]} linked checkpoints"; return 1; }
  mkdir -p "$out" || return 1
  free=$(df -BM --output=avail "$out" | tail -1 | tr -dc '0-9')
  [ "$free" -ge 2500 ] || { echo "$out has only $free MB free: not started"; return 1; }
  for t in "${tags[@]}"; do
    if [ "$what" = real ]; then
      for a in pub2 priv2; do [ -s "$out/official_${a}_net-1${t}_t50.npz" ] || tasks+=("$t $a"); done
    else
      for a in pub priv; do [ -s "$out/oldsim_${a}_s0_net-1${t}_t50.npz" ] || tasks+=("$t $a"); done
    fi
  done
  echo "family $fam ($what): ${#tags[@]} networks, ${#tasks[@]} fine-tunes to do on 4 x 2 GPUs -> $out; node $(hostname); $(date '+%F %T')"
  [ "${#tasks[@]}" -gt 0 ] || { echo "nothing to do"; return 0; }
  python -I -m py_compile $R2/scripts/pid_v2_seq_finetune.py || return 1
  for w in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${G[$((2 * w))]},${G[$((2 * w + 1))]}" timeout ${DP_TIMEOUT:-240} python -u -c 'import torch; m=torch.nn.Linear(4,4).cuda(); y=torch.nn.DataParallel(m)(torch.ones(64,4,device="cuda")); y.sum().backward(); torch.cuda.synchronize(); print("cards ok:", torch.cuda.device_count())' \
      || { echo "GPU pair ${G[$((2 * w))]},${G[$((2 * w + 1))]} on $(hostname) failed the two-card check"; return 1; }
  done
  for w in 0 1 2 3; do
    (
      for i in "${!tasks[@]}"; do
        [ $(( i % 4 )) -eq "$w" ] || continue
        read -r t a <<< "${tasks[$i]}"
        if [ "$what" = real ]; then
          CUDA_VISIBLE_DEVICES="${G[$((2 * w))]},${G[$((2 * w + 1))]}" OMP_NUM_THREADS=8 python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --pool "$a" --net -1 --horizon 1000 \
            --ckpt-dir $SN --ckpt-suffix "$t" --tag "$t" --steps 0 50 --out "$out" --dp < /dev/null || echo "FAILED: real $t $a"
        else
          CUDA_VISIBLE_DEVICES="${G[$((2 * w))]},${G[$((2 * w + 1))]}" OMP_NUM_THREADS=8 python -I $R2/scripts/pid_v2_seq_finetune.py --mode oldsim --side "$a" --holdout-seed 0 --net -1 --steps 0 50 --horizon 1000 \
            --out "$out" --ckpt-dir $SN --ckpt-suffix "$t" --tag "$t" --dp < /dev/null || echo "FAILED: old $t $a"
        fi
      done
    ) > logs/nano4/bigft-$J-$fam-$what-$w.log 2>&1 &
    pids+=($!)
  done
  for f in "${pids[@]}"; do wait "$f" || rc=1; done
  n=$(cat logs/nano4/bigft-$J-$fam-$what-*.log | grep -c FAILED)
  grep -hE 'training forward across|step 50/50' logs/nano4/bigft-$J-$fam-$what-0.log | head -4 | cut -c1-160
  if [ "$what" = real ]; then
    echo "files: $(ls "$out" | grep -c "^official_.*_net-1_${fam}s[0-9]*e_t50\.npz$") of $(( 2 * ${#tags[@]} ))"
  else
    echo "files: $(ls "$out" | grep -c "^oldsim_.*_s0_net-1_${fam}s[0-9]*e_t50\.npz$") of $(( 2 * ${#tags[@]} ))"
  fi
  echo "done at $(date +%T) ($(( (SECONDS - t0) / 60 )) min); failures: $n"
  [ "$n" = 0 ] || { grep -h FAILED logs/nano4/bigft-$J-$fam-$what-*.log | head -5; rc=1; }
  return $rc
}
main "$@"; exit $?
