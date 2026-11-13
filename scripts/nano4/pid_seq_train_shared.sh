#!/usr/bin/env bash
# As pid_seq_train_multi.sh (one sequence network per GPU of the job), but the games are read ONCE into shared memory and
# mapped by every network: eight networks of one node took over 20 minutes to read eight private copies (2026-11-14), and
# held eight times the memory.
#   bash <this> "<data options: --ext-ply ... --ext-select ...>" "<net> <tag> [options]" ...
# The shared arrays live in /dev/shm/pidseq_<job id> and are removed when the job ends. Run from the AI_CUP_2026-1 checkout.
set -uo pipefail
R2="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D="/dev/shm/pidseq_${SLURM_JOB_ID:-$$}"
trap 'rm -rf "$D"' EXIT
data=$1; shift
IFS=, read -r -a ALL <<< "${CUDA_VISIBLE_DEVICES}"
GPUS=()
for g in "${ALL[@]}"; do
  if CUDA_VISIBLE_DEVICES="$g" python -I -c "import torch; torch.zeros(1).cuda()" > /dev/null 2>&1; then GPUS+=("$g"); else echo "GPU $g on $(hostname) is not usable, skipped"; fi
done
if [ "${#GPUS[@]}" -lt 1 ] || [ $(( ${#GPUS[@]} * 2 )) -lt "$#" ]; then echo "only ${#GPUS[@]} usable GPUs of ${#ALL[@]} for $# networks" >&2; exit 1; fi
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="${GPUS[0]}" python -I "$R2/scripts/pid_seq_train.py" --net -1 --tag prepare --prepare-only --share-dir "$D" $data > "logs/nano4/pidseq_prepare-${SLURM_JOB_ID:-local}.log" 2>&1 \
  || { echo "reading the games failed; see logs/nano4/pidseq_prepare-${SLURM_JOB_ID:-local}.log" >&2; exit 1; }
echo "games shared in $D: $(du -sh "$D" | cut -f1)"
i=0; pids=()
for spec in "$@"; do
  read -r -a W <<< "$spec"; v="${W[0]}"; tag="${W[1]}"; g="${GPUS[$((i % ${#GPUS[@]}))]}"; log="logs/nano4/pidseq_${v}_${tag}-${SLURM_JOB_ID:-local}.log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$g" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I "$R2/scripts/pid_seq_train.py" --net "$v" --tag "$tag" --share-dir "$D" $data "${W[@]:2}" > "$log" 2>&1 &
  pids+=($!); echo "net $v tag $tag on GPU $g: pid ${pids[$i]}, log $log"; i=$((i + 1))
done
failed=0
for k in "${!pids[@]}"; do if wait "${pids[$k]}"; then echo "process $k done"; else echo "process $k FAILED"; failed=1; fi; done
exit "$failed"
