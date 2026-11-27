#!/usr/bin/env bash
# METHOD_NOTES 2.229: early check that the POOL FINE-TUNES work on a network whose windows carry stone ages (--shape-age 1;
# until now such networks were only read before the fine-tune, in the short runs of METHOD_NOTES 2.193). Run on the
# 1,000-step snapshot of one member (SNAP1K of pid_exp_ns2_train.sh) while the family trains:
#   pid_wait_files.sh 1 '^ckpt_-1_o68As0k1e[.]pt$' 90 bash pid_exp_ap_smoke.sh _o68As0k1e
# Three fine-tunes as the chains run them: a pseudo-pool (one GPU), the old-pool simulation (one GPU), a real pool with
# two GPUs (--dp). Vectors go to /dev/shm and are removed; the snapshot checkpoint is removed at the end.
# From the AI_CUP_2026-1 checkout, 8 GPUs. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 SN=/work/u2507115/pid_v2/seqnet t=$1 J=${SLURM_JOB_ID:-local} G D n rc=0 p pids=()
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  [ -s "$SN/ckpt_-1${t}.pt" ] || { echo "no checkpoint ckpt_-1${t}.pt"; return 1; }
  D=/dev/shm/apsmoke_$J
  mkdir -p $D || return 1
  ( CUDA_VISIBLE_DEVICES=${G[0]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --pool pub2 --net -1 --horizon 1000 \
      --ckpt-dir $SN --ckpt-suffix "$t" --tag "$t" --steps 50 --holdout 3 --holdout-seed 0 --out $D < /dev/null || echo "FAILED: pseudo-pool" ) > logs/nano4/apsmoke-$J-a.log 2>&1 &
  pids+=($!)
  ( CUDA_VISIBLE_DEVICES=${G[1]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -I $R2/scripts/pid_v2_seq_finetune.py --mode oldsim --side pub --holdout-seed 0 --net -1 --steps 0 50 --horizon 1000 \
      --out $D --ckpt-dir $SN --ckpt-suffix "$t" --tag "$t" < /dev/null || echo "FAILED: old-pool simulation" ) > logs/nano4/apsmoke-$J-b.log 2>&1 &
  pids+=($!)
  ( CUDA_VISIBLE_DEVICES=${G[2]},${G[3]} OMP_NUM_THREADS=8 python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --pool priv2 --net -1 --horizon 1000 \
      --ckpt-dir $SN --ckpt-suffix "$t" --tag "$t" --steps 0 50 --out $D --dp < /dev/null || echo "FAILED: real pool" ) > logs/nano4/apsmoke-$J-c.log 2>&1 &
  pids+=($!)
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  grep -hE "window|step 50/50|FAILED|Error|out of memory" logs/nano4/apsmoke-$J-?.log | cut -c1-200 | sort | uniq -c | head -20
  n=$(cat logs/nano4/apsmoke-$J-?.log | grep -c FAILED)
  ls -la $D | cut -c1-160
  # expected: one pseudo-pool file (t50), two old-pool files (t0, t50), two real-pool files (t0, t50)
  [ "$(ls $D | grep -c '_t50\.npz$')" -ge 3 ] || { echo "vectors are missing"; rc=1; }
  [ "$n" = 0 ] || rc=1
  find $D -delete 2> /dev/null
  rm -f "$SN/ckpt_-1${t}.pt"
  if [ "$rc" = 0 ]; then echo "AP SMOKE PASSED ($t) at $(date +%T)"; else echo "AP SMOKE FAILED ($t) at $(date +%T)"; fi
  return $rc
}
main "$@"; exit $?
