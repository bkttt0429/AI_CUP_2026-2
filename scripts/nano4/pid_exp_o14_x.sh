#!/usr/bin/env bash
# METHOD_NOTES 2.110：正式網路用到的每一局棋的「第二套」有監督特徵（"js"：在全盤平均／落點／
# policy 預期落點向量上聯合擬合的投影，64 維，+ 4 個 policy 純量；探針 P6）。int8 的分段檔，尺度取自
# 探針特徵庫 /work/u2507115/pid_v2/probep/ply_js（float，探針 B 的 115,011 局，維持原樣）。
#   pid_exp_o14_x.sh comp <起始分片> <分片總數>   第 1-4 折其餘的 679,039 局競賽棋局 -> supj/comp
#   pid_exp_o14_x.sh pools                                   官方題池的棋局                       -> supj/pools
#   pid_exp_o14_x.sh ext <起始分片> <分片總數>    116,420 個外部帳號的 1,261,070 局 -> supj/ext
# 每個工作 8 個分片。從 AI_CUP_2026-1 的工作目錄執行，8 張 GPU。有工作在執行它時不要修改；主體是一個函式。
set -uo pipefail  # 未定義變數即報錯，管線中任一步失敗即算失敗
main() {  # 主體包成函式，bash 會先讀完整個函式再執行
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox B=/work/u2507115/pid_v2/probeb P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj rc  # 儲存庫、外部資料、探針 B、探針 P 與輸出的位置
  mkdir -p $O  # 建立輸出資料夾
  if [ ! -s $P/stats_js.npz ]; then  # 整套共用一個尺度：探針特徵庫的尺度（可能有多個工作寫同一個檔）
    python -I -c "import sys; sys.path.insert(0, '$R2/scripts'); import pid_sup; print('stats over', pid_sup.write_stats(['$P/ply_js'], '$P/stats_js.$$.npz'), 'plies')" && mv $P/stats_js.$$.npz $P/stats_js.npz  # 先寫到帶行程編號的暫存檔再改名，避免同時寫入時檔案不完整
  fi  # 結束尺度檔準備
  local common="--proj $P/proj3.npz --scalars --stats $P/stats_js.npz --workers 10"  # 各子命令共用的抽取選項：聯合投影、純量、尺度
  case $1 in  # 依子命令分派
    comp)  # shellcheck disable=SC2086 # 競賽棋局（下一行的 $common 不加引號，拆成多個參數）
      bash $R2/scripts/nano4/shards_offset.sh 8 "$2" "$3" python -I $R2/scripts/pid_sup_ply_extract.py --keys /work/u2507115/pid_v2/fullrows_train/keys.txt --skip-keys $B/keys.txt $common --out $O/comp; rc=$? ;;  # 第 1-4 折的棋局，略過探針 B 已抽過的
    pools)  # shellcheck disable=SC2086 # 官方題池（$common 不加引號）
      bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_sup_ply_extract.py --pools $R2/data/tests_1111 $common --out $O/pools; rc=$? ;;  # 一個工作抽完全部題池棋局
    ext)  # shellcheck disable=SC2086 # 外部帳號（$common 不加引號）
      bash $R2/scripts/nano4/shards_offset.sh 8 "$2" "$3" python -I $R2/scripts/pid_sup_ply_extract.py --select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv --max-plies 256 $common --out $O/ext; rc=$? ;;  # 四個選取檔的外部棋局，只取前 256 手
    *) echo "unknown case $1"; return 2 ;;  # 不認得的子命令
  esac  # 結束分派
  echo "parts: comp $(ls $O/comp 2>/dev/null | grep -c 'npz$'), pools $(ls $O/pools 2>/dev/null | grep -c 'npz$'), ext $(ls $O/ext 2>/dev/null | grep -c 'npz$'); $(du -sh $O | cut -f1)"  # 回報三個資料夾的分段檔數與總大小
  return $rc  # 回傳結果
}  # 結束 main
main "$@"; exit $?  # 執行主流程並以它的結果結束
