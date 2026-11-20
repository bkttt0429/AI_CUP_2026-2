#!/usr/bin/env bash
#SBATCH --job-name=multi
#SBATCH --account=gov113074
#SBATCH --partition=128gpus
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=10:00:00
#SBATCH --no-kill
#SBATCH --output=logs/nano4/%x-%j.out
# A multi-node job whose command (a driver) runs on the first node and starts single-node steps on the job's nodes with
#   srun --nodes=1 --ntasks=1 --cpus-per-task=96 --gres=gpu:8 --nodelist=<node> ...
# (METHOD_NOTES 2.188: since 2026-11-17 the small partitions take no new jobs - group cap of 64 GPUs for all users - and
# the larger ones need more than one node per job). Submit from the AI_CUP_2026-1 checkout:
#   sbatch -p 128gpus -N 13 <this file> python3 -u <driver> ...
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
source "$ROOT/scripts/nano4/common.sh"
echo "nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
echo "+ $*"
"$@"
