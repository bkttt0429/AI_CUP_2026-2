#!/usr/bin/env bash
# One sequence network per GPU of the job (scripts/pid_seq_train.py; METHOD_NOTES 2.39). Run from the AI_CUP_2026-1 checkout:
#   sbatch --gres=gpu:4 -c 48 -t 08:00:00 scripts/nano4/run.sbatch bash <this> "1 b --steps 20000 --horizon 60000" "1 r1 --steps 25000 --drop 0.2 --mask 0.3" ...
# Each argument is "<net> <tag> [options of pid_seq_train.py]". GPUs that CUDA cannot use are skipped (networks then share cards).
# Training resumes from its state file when the job is run again.
# STAGGER=4: networks start in waves of four, each wave after the previous one has read its games (eight readers at once on
# one node starved each other for an hour, 2026-11-14).
set -uo pipefail
R2="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IFS=, read -r -a ALL <<< "${CUDA_VISIBLE_DEVICES}"
GPUS=()
for g in "${ALL[@]}"; do
  if CUDA_VISIBLE_DEVICES="$g" python -I -c "import torch; torch.zeros(1).cuda()" > /dev/null 2>&1; then GPUS+=("$g"); else echo "GPU $g on $(hostname) is not usable, skipped"; fi
done
if [ "${#GPUS[@]}" -lt 1 ] || [ $(( ${#GPUS[@]} * 2 )) -lt "$#" ]; then echo "only ${#GPUS[@]} usable GPUs of ${#ALL[@]} for $# networks" >&2; exit 1; fi
i=0; pids=(); logs=()
for spec in "$@"; do
  read -r -a W <<< "$spec"; v="${W[0]}"; tag="${W[1]}"; g="${GPUS[$((i % ${#GPUS[@]}))]}"; log="logs/nano4/pidseq_${v}_${tag}-${SLURM_JOB_ID:-local}.log"
  CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I "$R2/scripts/pid_seq_train.py" --net "$v" --tag "$tag" "${W[@]:2}" > "$log" 2>&1 &
  pids+=($!); logs+=("$log"); echo "net $v tag $tag on GPU $g: pid ${pids[$i]}, log $log"; i=$((i + 1))
  if [ "${STAGGER:-0}" -gt 0 ] && [ $(( i % STAGGER )) -eq 0 ] && [ "$i" -lt "$#" ]; then  # the next wave starts when this one has read its games
    while :; do
      ok=1
      for k in "${!pids[@]}"; do grep -q "^games loaded\|^games attached" "${logs[$k]}" 2>/dev/null || { kill -0 "${pids[$k]}" 2>/dev/null && ok=0; }; done
      [ "$ok" = 1 ] && break
      sleep 15
    done
    echo "wave of $STAGGER loaded at $(date +%H:%M:%S); starting the next"
  fi
done
failed=0
for k in "${!pids[@]}"; do if wait "${pids[$k]}"; then echo "process $k done"; else echo "process $k FAILED"; failed=1; fi; done
exit "$failed"
