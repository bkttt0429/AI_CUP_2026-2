#!/usr/bin/env bash
# As AI_CUP_2026-1 scripts/nano4/shards.sh, for a sharded command split over SEVERAL jobs that write to one directory:
#   bash shards_offset.sh <copies in this job> <first shard of this job> <shards in total> <command> [args...]
# Copy i runs on the job's i-th GPU with "--shard <first + i> --nshards <total>" appended; logs logs/nano4/shard<k>-<jobid>.log.
set -uo pipefail
N="$1"; OFF="$2"; TOTAL="$3"
shift 3
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:?}"
if [ "${#GPUS[@]}" -lt "$N" ]; then echo "need $N GPUs but the job has ${#GPUS[@]}" >&2; exit 1; fi
mkdir -p logs/nano4
pids=()
for ((i = 0; i < N; i++)); do
  k=$((OFF + i))
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$@" --shard "$k" --nshards "$TOTAL" > "logs/nano4/shard$k-${SLURM_JOB_ID:-local}.log" 2>&1 &
  pids+=($!)
  echo "shard $k of $TOTAL on GPU ${GPUS[$i]}: pid ${pids[$i]}"
done
failed=0
for i in "${!pids[@]}"; do if wait "${pids[$i]}"; then echo "shard $((OFF + i)) done"; else echo "shard $((OFF + i)) FAILED"; failed=1; fi; done
exit "$failed"
