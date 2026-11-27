#!/usr/bin/env bash
# METHOD_NOTES 2.240: cross-check of the ORIGINAL pools (the half of the Private score whose rows are frozen since 11-10)
# with the final's plain-input sequence networks (families o32F, o44F, o58F, o62F: 32 networks, no supervised per-ply features
# needed). One node, 8 GPUs:
#   1. per-ply vectors of the original pools' QUESTION games -> /work/u2507115/pid_v2/ply_origq  (pid_v2_ply_extract.py
#      --original-questions, 8 shards; skipped when the parts exist)
#   2. per network and side: 50-step fine-tune on the side's candidates (their stored games, 30 per candidate), vectors of
#      candidates' and question games at 0 and 50 steps -> /work/u2507115/pid_v2/orig_ft  (pid_v2_seq_finetune.py --mode orig)
#   3. scripts/pid_orig_check.py for both sides -> results/final/v2_orig_check_{pub,priv}.json
#   pid_exp_origq.sh [<families, default "o32F o44F o58F o62F">]
# Nothing here writes a submission. From the AI_CUP_2026-1 checkout. Do not edit while a job runs it.
set -uo pipefail
main() {
  local R2=/home/u2507115/BK_AICUP_2026-2 SN=/work/u2507115/pid_v2/seqnet Q=/work/u2507115/pid_v2/ply_origq O=/work/u2507115/pid_v2/orig_ft J=${SLURM_JOB_ID:-local}
  local fams=${1:-o32F o44F o58F o62F} G W w j f k s n free rc=0 pids=() tasks=() t0=$SECONDS a b
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"
  W=${#G[@]}
  free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')
  [ "$free" -ge 6500 ] || { echo "/work has only $free MB free: not started (this step writes about 1.5 GB)"; return 1; }
  mkdir -p $Q $O || return 1
  # 1. per-ply vectors of the question games
  if [ "$(ls $Q | grep -c '^ply_s.*\.npz$')" -lt 8 ]; then
    for w in $(seq 0 $((W - 1))); do
      CUDA_VISIBLE_DEVICES=${G[$w]} python -I $R2/scripts/pid_v2_ply_extract.py --tests-dir $R2/data/tests_1111 --out $Q --original-questions --shard "$w" --nshards "$W" --workers 8 > logs/nano4/origq-$J-ply-$w.log 2>&1 &
      pids+=($!)
    done
    for k in "${pids[@]}"; do wait "$k" || rc=1; done
    pids=()
    tail -n 1 logs/nano4/origq-$J-ply-0.log | cut -c1-160
    [ "$rc" = 0 ] || { echo "the per-ply extraction failed (logs/nano4/origq-$J-ply-*.log)"; grep -h -E 'Error|Traceback' logs/nano4/origq-$J-ply-*.log | head -5; return 1; }
  fi
  echo "question games: $(ls $Q | grep -c '^ply_s.*\.npz$') parts, $(du -sh $Q | cut -f1); $(date +%T)"
  # 2. vectors
  for f in $fams; do
    n=$(bash $R2/scripts/nano4/pid_fam_links.sh "$f" | awk '{print $2}')
    [ "${n:-0}" -ge 7 ] || { echo "family $f has only ${n:-0} networks"; return 1; }
    for k in $(seq 0 23); do
      [ -s $SN/ckpt_-1_${f}s${k}e.pt ] || continue
      for s in pub priv; do [ -s "$O/orig_${s}_net-1_${f}s${k}e_t50.npz" ] || tasks+=("$s _${f}s${k}e"); done
    done
  done
  echo "${#tasks[@]} fine-tunes to do on $W GPUs ($fams), start $(date +%T)"
  for w in $(seq 0 $((W - 1))); do
    (
      for j in "${!tasks[@]}"; do
        [ $(( j % W )) -eq "$w" ] || continue
        read -r a b <<< "${tasks[$j]}"
        CUDA_VISIBLE_DEVICES=${G[$w]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -I $R2/scripts/pid_v2_seq_finetune.py --mode orig --side "$a" --net -1 --horizon 1000 --ckpt-dir $SN --ckpt-suffix "$b" --tag "$b" \
          --steps 0 50 --out $O < /dev/null || echo "FAILED: ${tasks[$j]}"
      done
    ) > logs/nano4/origq-$J-ft-$w.log 2>&1 &
    pids+=($!)
  done
  for k in "${pids[@]}"; do wait "$k" || rc=1; done
  n=$(cat logs/nano4/origq-$J-ft-*.log 2> /dev/null | grep -c FAILED)
  grep -h -E 'orig_(pub|priv)_net.*games \(|step 50/50' logs/nano4/origq-$J-ft-0.log | head -4 | cut -c1-220
  echo "fine-tunes done at $(date +%T) ($(( (SECONDS - t0) / 60 )) min); failures: $n; files: $(ls $O | grep -c '_t50\.npz$') ($(du -sh $O | cut -f1))"
  [ "$n" = 0 ] || { grep -h -B2 FAILED logs/nano4/origq-$J-ft-*.log | head -12 | cut -c1-240; return 1; }
  # 3. the check, the Public side (answers known) first
  for s in pub priv; do
    python -I $R2/scripts/pid_orig_check.py --side $s --vec-dir $O --out $R2/results/final/v2_orig_check_$s.json 2>&1 | cut -c1-420 || rc=1
  done
  echo "done at $(date +%T), rc $rc"
  return $rc
}
main "$@"; exit $?
