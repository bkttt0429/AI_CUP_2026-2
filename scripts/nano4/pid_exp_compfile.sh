#!/usr/bin/env bash
# METHOD_NOTES 2.194：一份已存在的組成文件的提交檔（步驟與設定同判定鏈
# pid_v2_newnets3f.sh：a2k 配對、1 局題先驗 0.05 / 1.6）。原題池 800 列來自 make_file 腳本的 --old-submission；pid_v2o16c_b_a2k_p3.csv 只是 make_file4 的比較檔。
# 檔案由 make_file 腳本在 upload/ 建好後「立刻」搬到 results/final/not_for_upload/——一個檔
# 只有在它的候選組成通過規則之後，才以人工搬回 upload/。
#   pid_exp_compfile.sh <文件> [<要比較的檔，在 upload/ 內> ...]
# 從 AI_CUP_2026-1 的工作目錄執行。有工作在執行它時不要修改。
set -uo pipefail  # 未定義變數即報錯，管線中任一步失敗即算失敗
main() {  # 主體包成函式，bash 會先讀完整個函式再執行
  local R2=/home/u2507115/BK_AICUP_2026-2 n=$1 f rc=0  # 儲存庫位置、融合文件名稱與結果碼
  shift  # 其餘參數是要比較的檔
  local N=$R2/results/final/not_for_upload  # 通過規則之前存放檔案的資料夾
  mkdir -p $N  # 確保資料夾存在
  for f in pid_${n}s.csv pid_${n}_a2k.csv pid_${n}_a2k_p3.csv; do [ -e $R2/upload/$f ] && { echo "upload/$f exists already"; return 1; }; done  # upload/ 已有同名檔就不覆蓋
  bash $R2/scripts/nano4/pid_v2_make_file2.sh "$n" "pid_${n}s.csv" 1 && bash $R2/scripts/nano4/pid_v2_make_file2.sh "$n" "pid_${n}_a2k.csv" 2 0.5 && bash $R2/scripts/nano4/pid_v2_make_file4.sh "$n" "pid_${n}_a2k_p3.csv" 2 0.5 0.05 1.6 pid_v2o16c_b_a2k_p3.csv || rc=1  # 依序建軟性配對 lam=1 的 s 檔、a2k 配對檔（lam 2、tau 0.5）與再加 1 局題先驗的 a2k_p3 檔；pid_v2o16c_b_a2k_p3.csv 只拿來和最後一檔比較
  for f in pid_${n}s.csv pid_${n}_a2k.csv pid_${n}_a2k_p3.csv; do [ -e $R2/upload/$f ] && mv -f $R2/upload/$f $N/; done  # 建好的檔立刻移出 upload/
  [ "$rc" = 0 ] || { echo "the file was not built"; return 1; }  # 建檔失敗
  echo "== $N/pid_${n}_a2k_p3.csv: md5 $(md5sum $N/pid_${n}_a2k_p3.csv | cut -c1-32), $(wc -l < $N/pid_${n}_a2k_p3.csv) lines"  # 回報最終檔的 md5 與行數
  for f in "$@"; do  # 與指定的檔逐一比較
    echo "== against upload/$f"  # 標示比較對象
    python -I $R2/scripts/pid_diff_files.py $R2/upload/$f $N/pid_${n}_a2k_p3.csv || rc=1  # 逐池計數第一名不同、五個答案有任何不同的題目數，並印出兩檔的 md5
  done  # 結束比較
  return $rc  # 回傳結果
}  # 結束 main
main "$@"; exit $?  # 執行主流程並以它的結果結束
