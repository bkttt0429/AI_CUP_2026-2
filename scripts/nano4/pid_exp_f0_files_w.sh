#!/usr/bin/env bash
# METHOD_NOTES 2.233：一個組成的提交檔，但它的「融合權重」取自另一份文件（以題池自己的候選人
# 重新擬合的權重），而不是 v2ge 的。= experiments/nano4/pid_exp_f0_files.sh，中間多一步：
#   1. data/day_scripts/build_v2h.py <名稱> <組成>      （正式題池的文件，用 v2ge 的權重）
#   2. 9 個權重換成 results/pid_fusion_<權重文件>.json 的權重（成分相同，會核對）
#   3. scripts/nano4/pid_exp_compfile.sh <名稱> <要比較的檔>   -> results/final/not_for_upload/pid_<name>_a2k_p3.csv
#   pid_exp_f0_files_w.sh <名稱>=<組成> <權重文件> [<upload/ 內要比較的檔>]
# 一個檔只有在它的候選組成通過規則之後，才以人工放進 upload/。從 AI_CUP_2026-1 的工作目錄執行。
set -uo pipefail  # 未定義變數即報錯，管線中任一步失敗即算失敗
main() {  # 主體包成函式，bash 會先讀完整個函式再執行
  local R2=/home/u2507115/BK_AICUP_2026-2 c=$1 w=$2 cmp=${3:-pid_v2fin9_w_a2k_p3.csv} n  # 名稱=組成、權重文件、比較檔（預設為當時的定案檔）
  n=${c%%=*}  # 取出等號前的文件名稱
  [ -s $R2/results/pid_fusion_$w.json ] || { echo "no weights document $w"; return 1; }  # 權重文件必須存在
  [ -e $R2/results/pid_fusion_$n.json ] && { echo "document $n exists already"; return 1; }  # 不覆蓋已存在的融合文件
  python -I $R2/data/day_scripts/build_v2h.py "$n" "${c#*=}" || { echo "document $n was not built"; return 1; }  # 依組成建融合文件（權重先用 v2ge 的）
  python3 -I - "$R2/results/pid_fusion_$n.json" "$R2/results/pid_fusion_$w.json" <<'PY' || { echo "the weights were not replaced"; return 1; }  # 核對成分相同後，把 9 個融合權重換成權重文件的值
import json, os, sys
f, g = sys.argv[1], sys.argv[2]
d, s = json.load(open(f)), json.load(open(g))
assert list(d["components"]) == list(s["components"]) and len(s["weights"]) == len(d["components"])
old = d["weights"]
d["weights"] = [float(x) for x in s["weights"]]
d["note"] = (d.get("note") or "") + f" | fusion weights of {os.path.basename(g)} (METHOD_NOTES 2.233) in place of v2ge's"
json.dump(d, open(f + ".tmp", "w"), indent=1)
os.replace(f + ".tmp", f)
print("weights", dict(zip(d["components"], [round(x, 4) for x in old])), "->", dict(zip(d["components"], [round(x, 4) for x in d["weights"]])))
PY
  bash $R2/scripts/nano4/pid_exp_compfile.sh "$n" "$cmp"  # 照一般流程出檔，並與比較檔對照
  local rc=$?  # 出檔的結果碼
  ls -la $R2/results/final/not_for_upload/ | grep -E "pid_${n}_a2k_p3\.csv"  # 列出建好的檔
  echo "done at $(date +%T), rc $rc"  # 回報完成時間與結果
  return $rc  # 回傳結果
}  # 結束 main
main "$@"; exit $?  # 執行主流程並以它的結果結束
