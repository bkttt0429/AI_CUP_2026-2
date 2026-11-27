#!/usr/bin/env bash
# 我的正式網路中，成員以配方標籤訓練的那些家族的成員連結 ckpt_-1_<family>s<k>e.pt
# （METHOD_NOTES 2.201、2.204、2.207、2.211）。判定鏈與微調輔助腳本共用的唯一一處。
#   pid_fam_links.sh <家族> [<家族> ...]      每個家族印出 "<family> <linked checkpoints>"；有任一家族不到 7 個時 rc 為 1
#     o44F o50F o58F             成員 0-3 = <base>Ss0-3，4-7 = <base>SMs0-3
#     o45F-o49F o51F-o53F o56F o57F   成員 0-5 = <base>As0-5，6 / 7 = <base>Cs0 / Cs1
#     o62F o66F o70F 同 o44F；o60F o61F o63F o64F o65F o67F o68F o69F o71F 同 o45F（METHOD_NOTES 2.220、2.229）
#     o53KF / o56KF              o53 在 50,000 步／o56 在 45,000 步的「快照」（ckpt_-1_<tag>k50e.pt / k45e.pt）
# 其他家族：不建任何連結，只清點現有的連結。只建連結（同一資料夾內名稱的 ln -sf）。
set -uo pipefail  # 未定義變數即報錯，管線中任一步失敗即算失敗
main() {  # 主體包成函式，bash 會先讀完整個函式再執行
  local SN=/work/u2507115/pid_v2/seqnet f b sfx i k n rc=0  # 檢查點資料夾與迴圈變數
  for f in "$@"; do  # 逐一處理每個家族
    b=${f%F}; sfx=""  # 基底標籤（去掉結尾的 F）；快照後綴預設為空
    case "$f" in o53KF) b=o53; sfx=k50 ;; o56KF) b=o56; sfx=k45 ;; esac  # 兩個快照家族改指向快照檢查點
    case "$f" in  # 依家族的命名方式建連結
      o44F|o50F|o58F|o62F|o66F|o70F)  # 一般輸入的配方：Ss 與遮開局的 SMs 各 4 個
        for i in 0 1 2 3; do  # 四組成員
          [ -s $SN/ckpt_-1_${b}Ss${i}e.pt ] && ln -sf ckpt_-1_${b}Ss${i}e.pt $SN/ckpt_-1_${f}s${i}e.pt  # Ss<i> -> 成員 i
          [ -s $SN/ckpt_-1_${b}SMs${i}e.pt ] && ln -sf ckpt_-1_${b}SMs${i}e.pt $SN/ckpt_-1_${f}s$((i + 4))e.pt  # SMs<i> -> 成員 i+4
        done ;;  # 結束這類家族
      o45F|o46F|o47F|o48F|o49F|o51F|o52F|o53F|o56F|o57F|o59F|o60F|o61F|o63F|o64F|o65F|o67F|o68F|o69F|o71F|o53KF|o56KF)  # 第二套特徵的配方：As 6 個、只用有監督特徵的 Cs 2 個
        for i in 0 1 2 3 4 5; do [ -s $SN/ckpt_-1_${b}As${i}${sfx}e.pt ] && ln -sf ckpt_-1_${b}As${i}${sfx}e.pt $SN/ckpt_-1_${f}s${i}e.pt; done  # As<i> -> 成員 0-5
        for i in 0 1; do [ -s $SN/ckpt_-1_${b}Cs${i}${sfx}e.pt ] && ln -sf ckpt_-1_${b}Cs${i}${sfx}e.pt $SN/ckpt_-1_${f}s$((i + 6))e.pt; done ;;  # Cs0／Cs1 -> 成員 6／7
    esac  # 其他家族不建連結
    n=0; for k in $(seq 0 23); do [ -s $SN/ckpt_-1_${f}s${k}e.pt ] && n=$((n + 1)); done  # 清點已連結的成員檢查點
    echo "$f $n"  # 印出家族與成員數
    [ "$n" -ge 7 ] || rc=1  # 少於 7 個成員視為失敗
  done  # 結束家族迴圈
  return $rc  # 有家族成員不足時回傳 1
}  # 結束 main
main "$@"; exit $?  # 執行主流程並以它的結果結束
