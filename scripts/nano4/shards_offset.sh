#!/usr/bin/env bash
# 同 AI_CUP_2026-1 的 scripts/nano4/shards.sh，但用於分散到「多個」工作、寫進同一個資料夾的分片指令：
#   bash shards_offset.sh <本工作的份數> <本工作的第一個分片> <分片總數> <指令> [參數...]
# 第 i 份在本工作的第 i 張 GPU 上執行，並附加 "--shard <first + i> --nshards <total>"；紀錄寫到 logs/nano4/shard<k>-<jobid>.log。
set -uo pipefail  # 未定義變數即報錯，管線中任一步失敗即算失敗
N="$1"; OFF="$2"; TOTAL="$3"  # 本工作的份數、起始分片編號、分片總數
shift 3  # 其餘參數就是要執行的指令
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:?}"  # 本工作分到的 GPU 清單（未設定就報錯）
if [ "${#GPUS[@]}" -lt "$N" ]; then echo "need $N GPUs but the job has ${#GPUS[@]}" >&2; exit 1; fi  # GPU 不夠就不啟動
mkdir -p logs/nano4  # 紀錄資料夾
pids=()  # 各份的行程編號
for ((i = 0; i < N; i++)); do  # 逐份啟動
  k=$((OFF + i))  # 這一份在全部分片中的編號
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$@" --shard "$k" --nshards "$TOTAL" > "logs/nano4/shard$k-${SLURM_JOB_ID:-local}.log" 2>&1 &  # 每份占一張 GPU、限制 CPU 執行緒數，背景執行
  pids+=($!)  # 記下行程編號
  echo "shard $k of $TOTAL on GPU ${GPUS[$i]}: pid ${pids[$i]}"  # 回報已啟動
done  # 結束啟動迴圈
failed=0  # 是否有任何一份失敗
for i in "${!pids[@]}"; do if wait "${pids[$i]}"; then echo "shard $((OFF + i)) done"; else echo "shard $((OFF + i)) FAILED"; failed=1; fi; done  # 等每一份結束並記下失敗
exit "$failed"  # 任何一份失敗就回傳非零
