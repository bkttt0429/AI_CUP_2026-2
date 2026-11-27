#!/usr/bin/env bash
# One sequence network per GPU of the job, all on ONE copy of the games (scripts/pid_seq_train.py --fork; METHOD_NOTES 2.57):
# the games are read once into shared memory, then the trainers are forked. Use this instead of pid_seq_train_multi.sh.
#   bash <this> "<data options: [--smoke] --ext-ply ... --ext-select ...>" "<net> <tag> [options]" ...
# Logs: the reader's in the job's own output, each network's in logs/nano4/pidseq_<net>_<tag>-<job>.log.
set -uo pipefail
R2="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data=$1; shift
IFS=, read -r -a ALL <<< "${CUDA_VISIBLE_DEVICES}"
GPUS=()
for g in "${ALL[@]}"; do
  if CUDA_VISIBLE_DEVICES="$g" python -I -c "import torch; torch.zeros(1).cuda()" > /dev/null 2>&1; then GPUS+=("$g"); else echo "GPU $g on $(hostname) is not usable, skipped"; fi
done
if [ "${#GPUS[@]}" -lt 1 ] || [ $(( ${#GPUS[@]} * 2 )) -lt "$#" ]; then echo "only ${#GPUS[@]} usable GPUs of ${#ALL[@]} for $# networks" >&2; exit 1; fi
free_kb=$(awk '/CommitLimit/{l=$2} /Committed_AS/{c=$2} END{print l-c}' /proc/meminfo)
echo "node $(hostname): commit room $((free_kb / 1048576)) GB; memory $(free -g | awk 'NR==2{print $4" GB free, "$6" GB cache"}')"
spec=$(mktemp "${TMPDIR:-/tmp}/pidseq_spec_XXXXXX")
i=0
for s in "$@"; do
  read -r -a W <<< "$s"
  echo "${GPUS[$((i % ${#GPUS[@]}))]} logs/nano4/pidseq_${W[0]}_${W[1]}-${SLURM_JOB_ID:-local}.log --net ${W[0]} --tag ${W[1]} ${W[*]:2}" >> "$spec"
  i=$((i + 1))
done
# shellcheck disable=SC2086
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I "$R2/scripts/pid_seq_train.py" --fork "$spec" $data
rc=$?
rm -f "$spec"
exit $rc
