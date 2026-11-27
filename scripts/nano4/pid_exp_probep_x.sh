#!/usr/bin/env bash
# METHOD_NOTES 2.102，探針 P，第 1 步：探針 B 棋局（第 1、2 折的模擬題池，以及第 3、4 折每位玩家
# 最多 32 局；從不用第 0 折）相對於 policy 的每局統計列與逐手特徵（scripts/pid_pol_extract.py）。
#   pid_exp_probep_x.sh [起始分片，預設 0] [分片總數，預設 8]
# 從 AI_CUP_2026-1 的工作目錄執行，8 張 GPU。有工作在執行它時不要修改；主體是一個函式，在最後一行呼叫。
set -uo pipefail  # 未定義變數即報錯，管線中任一步失敗即算失敗
main() {  # 主體包成函式，bash 會先讀完整個函式再執行
  local R2=/home/u2507115/BK_AICUP_2026-2 B=/work/u2507115/pid_v2/probeb D=/work/u2507115/pid_v2/probep  # 儲存庫、探針 B 與輸出的位置
  mkdir -p $D  # 建立輸出資料夾
  bash $R2/scripts/nano4/shards_offset.sh 8 "${1:-0}" "${2:-8}" python -I $R2/scripts/pid_pol_extract.py --keys $B/keys.txt --proj $B/proj.npz --out $D --workers 10  # 8 張 GPU 抽每局統計列（rows）與三份逐手特徵
  local rc=$?  # 抽取的結果碼
  echo "parts: rows $(ls $D/rows | grep -c npz), per-ply $(ls $D/ply_r | grep -c npz) / $(ls $D/ply_fc | grep -c npz) / $(ls $D/ply_s | grep -c npz); $(du -sh $D | cut -f1)"  # 回報各輸出的分段檔數與大小
  return $rc  # 回傳結果
}  # 結束 main
main "$@"; exit $?  # 執行主流程並以它的結果結束
