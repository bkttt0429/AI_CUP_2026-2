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
# 多節點的工作：它的指令（驅動程式）在第一個節點上執行，並以下列方式在本工作的各節點上啟動單節點的工作步驟（step）
#   srun --nodes=1 --ntasks=1 --cpus-per-task=96 --gres=gpu:8 --nodelist=<節點> ...
# （METHOD_NOTES 2.188：自 2026-11-17 起小分割區不再收新工作——所有使用者合計 64 張 GPU 的群組上限——
# 而較大的分割區每個工作要不只一個節點）。從 AI_CUP_2026-1 的工作目錄送出：
#   sbatch -p 128gpus -N 13 <本檔> python3 -u <驅動程式> ...
set -euo pipefail  # 任何錯誤即中止，未定義變數即報錯，管線中任一步失敗即算失敗
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"  # 送出工作時的目錄（AI_CUP_2026-1 的工作目錄）
source "$ROOT/scripts/nano4/common.sh"  # 進入專案、啟用 .venv 並印出工作標頭
echo "nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"  # 列出本工作分到的節點
echo "+ $*"  # 印出要執行的驅動程式指令
"$@"  # 在第一個節點上執行驅動程式
