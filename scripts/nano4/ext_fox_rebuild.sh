#!/usr/bin/env bash
# METHOD_NOTES 2.148：重建 2026-11-16 04:17 被刪掉的外部逐手分段檔（/work/u2507115/ext_fox/ext_ply_home：
# scale1_c、scale2_top、scale2_5d，是唯一的一份；~/BK_AICUP_2026-2/data/ext_ply 是指向它的連結）。不必下載任何東西：
# 選取檔裡就有棋譜，指令也和原本的相同（工作 769445 與 771730）。
#   ext_fox_rebuild.sh c      scale1 第 700,000-977,750 列 -> scale1_c   （1,392 個分段檔；第一次在 8 張 GPU 上花 2 小時 35 分）
#   ext_fox_rebuild.sh d      scale2_top（208 個分段檔）與 scale2_5d（248 個分段檔）   （51 分）
# 最後對照選取檔清點分段檔數與局數。從 AI_CUP_2026-1 的工作目錄執行，8 張 GPU。
# 有工作在執行它時不要修改。
set -uo pipefail  # 未定義變數即報錯，管線中任一步失敗即算失敗
main() {  # 主體包成函式，bash 會先讀完整個函式再執行
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox which=$1 rc=0 spec name sel rng  # 儲存庫與外部資料的位置、要重建哪一批
  local H=$X/ext_ply_home  # 重建的目的地（data/ext_ply 連結指向這裡）
  mkdir -p "$H"  # 確保目的地存在
  [ "$(readlink -f $R2/data/ext_ply)" = "$H" ] || { echo "$R2/data/ext_ply does not point to $H"; return 1; }  # 確認訓練讀的連結確實指向這裡
  df -h /work/u2507115 | tail -1  # 先記下 /work 的剩餘空間
  local jobs=()  # 這次要抽取的各批：名稱、選取檔、列範圍
  case "$which" in  # 依參數決定重建哪一批
    c) jobs=("scale1_c $X/scale1/select.csv --start 700000 --stop 977750") ;;  # scale1 的第三段（原工作 769445）
    d) jobs=("scale2_top $X/scale2_top/select.csv" "scale2_5d $X/scale2_5d/select.csv") ;;  # 高段的兩批（原工作 771730）
    *) echo "usage: ext_fox_rebuild.sh c|d"; return 2 ;;  # 參數錯誤
  esac  # 結束批次選擇
  for spec in "${jobs[@]}"; do  # 逐批重建
    read -r name sel rng <<< "$spec"  # 拆出名稱、選取檔與列範圍參數
    # shellcheck disable=SC2086 # 下一行的 $rng 刻意不加引號，讓它拆成多個參數
    bash scripts/nano4/shards.sh 8 python -I $R2/scripts/pid_ext_ply_extract.py --workers 10 --select "$sel" $rng --out "$H/$name" || { echo "extraction of $name FAILED"; rc=1; }  # 8 張 GPU 分片抽逐手特徵，失敗則記下
    # shellcheck disable=SC2086 # 同上，$rng 要拆成多個參數
    python -I - "$H/$name" "$sel" $rng <<'EOF' || rc=1  # 核對：要有分段檔、鍵不重複，且缺列的選取局少於 1%
import csv, glob, sys
import numpy as np
out, sel, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
start = int(rest[rest.index("--start") + 1]) if "--start" in rest else 0
stop = int(rest[rest.index("--stop") + 1]) if "--stop" in rest else 0
csv.field_size_limit(10 ** 9)
with open(sel, newline="") as fh:
    want = [r["game_key"] for r in csv.DictReader(fh)]
want = want[start:(stop or len(want))]
files = sorted(f for f in glob.glob(out + "/ply_x_*.npz") if not f.endswith(".tmp.npz"))
got = np.concatenate([np.load(f)["key"].astype(str) for f in files]) if files else np.array([], str)
miss = len(set(want) - set(got.tolist()))
print(f"{out}: {len(files)} parts, {len(got)} games ({len(set(got.tolist()))} distinct) of {len(want)} selected; {miss} selected games without a part row (records the replay cannot follow are left out)")
sys.exit(0 if files and len(got) == len(set(got.tolist())) and miss < 0.01 * len(want) else 1)
EOF
  done  # 結束各批
  du -sh "$H"/* 2> /dev/null; df -h /work/u2507115 | tail -1  # 回報各資料夾大小與剩餘空間
  return $rc  # 任何一批失敗就回傳非零
}  # 結束 main
main "$@"; exit $?  # 執行主流程並以它的結果結束
