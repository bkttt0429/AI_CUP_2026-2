#!/usr/bin/env bash
# METHOD_NOTES 2.107，探針 P6，第 1 步：在 [全盤平均、落點、policy 預期落點] 的平均值上「聯合」擬合的有監督投影
# （scripts/pid_pol_extract.py 的每局統計列；只用第 3、4 折的玩家，從不用第 0 折），再以它抽出
# 探針 B 棋局的逐手特徵，有與沒有四個 policy 純量兩種（一次產生兩個特徵庫）：
#   /work/u2507115/pid_v2/probep/ply_j   64 維        /work/u2507115/pid_v2/probep/ply_js   64 維 + 4 個純量
# 從 AI_CUP_2026-1 的工作目錄執行，8 張 GPU。有工作在執行它時不要修改；主體是一個函式，在最後一行呼叫。
set -uo pipefail  # 未定義變數即報錯，管線中任一步失敗即算失敗
main() {  # 主體包成函式，bash 會先讀完整個函式再執行
  local R2=/home/u2507115/BK_AICUP_2026-2 B=/work/u2507115/pid_v2/probeb D=/work/u2507115/pid_v2/probep rc  # 儲存庫、探針 B 與探針 P 的位置
  if [ ! -s $D/proj3.npz ]; then  # 還沒有聯合投影時才擬合
    python -I $R2/scripts/pid_sup_fit.py --rows $D/rows --blocks g x e --fit-folds 3 4 --k 64 --out $D/proj3.npz || { echo "projection fit FAILED"; return 1; }  # 三個區塊聯合擬合 64 維投影，只用第 3、4 折
  fi  # 結束投影擬合
  bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_sup_ply_extract.py --keys $B/keys.txt --proj $D/proj3.npz --scalars --out $D/ply_js --out2 $D/ply_j --workers 10  # 8 張 GPU 一次寫出有純量與沒有純量的兩個特徵庫
  rc=$?  # 抽取的結果碼
  echo "parts: $(ls $D/ply_j | grep -c 'npz$') / $(ls $D/ply_js | grep -c 'npz$'); $(du -sh $D | cut -f1)"  # 回報兩個特徵庫的分段檔數與大小
  return $rc  # 回傳結果
}  # 結束 main
main "$@"; exit $?  # 執行主流程並以它的結果結束
