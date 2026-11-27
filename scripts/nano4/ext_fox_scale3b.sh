#!/usr/bin/env bash
# METHOD_NOTES 2.165：更多外部帳號（「scale3b」）——Fox Go Dataset 段位資料夾 3d 1d 1k 3k 4k 中，
# 至少有 12 局合格對局、且沒有被先前任何選取包含的帳號，每人 8 局（scale3 的選取規則，
# notes 2.52；那一批的特徵在 11-14 為了空間刪掉了）。各階段，每個階段一個工作：
#   ingest <資料夾>       下載的壓縮檔 /work/u2507115/ext_fox/raw_<資料夾> -> 精簡分段檔 parts_<資料夾>；之後刪除壓縮檔
#   select                對競賽訓練棋局的重疊閘門（只計數；不含第 0 折、不碰測試集、不讀任何
#                         帳號；命中而且是同一份棋譜就停止工作）-> 選取檔 scale3b/select.csv -> 刪除精簡分段檔
#   ply <起始分片> work|home    逐手主幹向量（一般輸入），48 個分片中的 8 個，壓縮後寫到
#                         /work/u2507115/ext_fox/scale3b/ply  或  ~/BK_AICUP_2026-2/data/ext_ply2/scale3b
#   sup <起始分片>     第二套特徵（js），32 個分片中的 8 個 -> ~/BK_AICUP_2026-2/data/supj_ext/scale3b
#   check                 三個特徵庫的分段檔數與局數對照選取檔
# 從 AI_CUP_2026-1 的工作目錄執行。有工作在執行它時不要修改。
set -uo pipefail  # 未定義變數即報錯，管線中任一步失敗即算失敗
main() {  # 主體包成函式，bash 會先讀完整個函式再執行
  local X=/work/u2507115/ext_fox R2=/home/u2507115/BK_AICUP_2026-2 P=/work/u2507115/pid_v2/probep mode=$1 r s NAME=scale3b  # 外部資料、儲存庫與投影檔的位置；階段；這批的名稱
  local PY="$X/tools_venv/bin/python -I" FOLDERS="3d 1d 1k 3k 4k"  # 匯入與選取用的獨立工具環境；這批的五個段位資料夾
  local PLY_W=$X/$NAME/ply PLY_H=$R2/data/ext_ply2/$NAME SUP_H=$R2/data/supj_ext/$NAME SEL=$X/$NAME/select.csv  # 兩處逐手特徵庫、第二套特徵庫與選取檔
  case "$mode" in  # 依階段分派
    ingest)  # 匯入一個段位資料夾
      r=$2  # 資料夾名稱
      [ -d "$X/raw_$r" ] || { echo "no $X/raw_$r"; return 1; }  # 壓縮檔必須已經下載
      $PY $R2/scripts/ext_fox_ingest.py --stage ingest --archives $X/raw_$r --out $X/parts_$r || { echo "ingest of $r FAILED (archives kept)"; return 1; }  # 壓縮檔 -> 精簡分段檔；失敗時保留壓縮檔
      rm -rf "$X/raw_$r"  # 匯入成功後刪除壓縮檔以省空間
      echo "$r: parts $(du -sh $X/parts_$r | cut -f1); work free $(df -BM --output=avail /work/u2507115 | tail -1)" ;;  # 回報分段檔資料夾的大小與 /work 剩餘空間
    select)  # 重疊閘門通過後選取帳號與棋局
      local parts=() named=() gate=$R2/results/final/ext_fox_overlap_$NAME.json probe=$R2/results/final/ext_fox_match_$NAME.json hits same  # 分段檔資料夾清單、閘門與命中描述的輸出檔
      for r in $FOLDERS; do [ -d "$X/parts_$r" ] || { echo "no parts of $r"; return 1; }; parts+=("$X/parts_$r"); named+=("$r=$X/parts_$r"); done  # 五個資料夾都必須已匯入
      $PY $R2/scripts/ext_fox_ingest.py --stage overlap --parts "${parts[@]}" --out $gate || return 1  # 只計數的重疊閘門
      hits=$(python3 -c "import json; print(json.load(open('$gate'))['matches'])")  # 讀出命中的局數
      echo "overlap gate: $hits competition game(s) share their first 60 moves with an external record"  # 回報命中數
      if [ "$hits" != 0 ]; then  # 有命中：檢視是不是同一局
        $PY $R2/scripts/ext_fox_match_probe.py --parts "${named[@]}" --out $probe || return 1  # 不讀帳號，描述每一對命中
        same=$(python3 -c "
import json
d = json.load(open('$probe'))
print(sum(1 for x in d if x['whole_record_equal'] or x['moves_equal_from_start'] >= min(x['competition_moves'], x['external_moves'])))")  # 整局相同或一局是另一局開頭的配對數
        echo "pairs that are the same record or one the start of the other: $same"  # 回報真正重疊的配對數
        if [ "$same" != 0 ]; then echo "OVERLAP: stopped before the selection; tell the user"; return 1; fi  # 確有重疊：停在選取之前，回報使用者
      fi  # 結束命中檢視
      $PY $R2/scripts/ext_fox_select.py --joint --by-account --parts "${named[@]}" --skip $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv --accounts "${ACCOUNTS:-90000}" --games 8 --need 12 --seed 5 --out $X/$NAME || return 1  # 跨資料夾合併選帳號：排除先前四次選取，每人 8 局
      rm -rf "${parts[@]}"  # 選完刪除精簡分段檔
      echo "selection: $(($(wc -l < $SEL) - 1)) games, $(du -sh $SEL | cut -f1); $(head -c 600 $X/$NAME/select.json | tr '\n' ' ')"  # 回報選取的局數、檔案大小與摘要開頭
      df -h /work/u2507115 /home/u2507115 | tail -2 ;;  # 回報兩處的剩餘空間
    ply)  # 抽逐手主幹向量
      local out=$PLY_W; [ "${3:-work}" = home ] && out=$PLY_H  # 依第三個參數寫到 /work 或 /home
      [ -s "$SEL" ] || { echo "no selection"; return 1; }  # 需要選取檔
      mkdir -p "$out"  # 建立輸出資料夾
      bash $R2/scripts/nano4/shards_offset.sh 8 "$2" 48 python -I $R2/scripts/pid_ext_ply_extract.py --workers 10 --compress --select $SEL --out "$out" || { echo "a shard failed"; return 1; }  # 本工作跑 48 個分片中的 8 個，壓縮寫出
      echo "parts in $out: $(ls $out | grep -c '[0-9]\.npz$'), $(du -sh $out | cut -f1)"; df -h /work/u2507115 /home/u2507115 | tail -2 ;;  # 回報分段檔數、大小與剩餘空間
    sup)  # 抽第二套特徵
      [ -s "$SEL" ] || { echo "no selection"; return 1; }  # 需要選取檔
      mkdir -p "$SUP_H"  # 建立輸出資料夾
      bash $R2/scripts/nano4/shards_offset.sh 8 "$2" 32 python -I $R2/scripts/pid_sup_ply_extract.py --select $SEL --max-plies 256 --proj $P/proj3.npz --scalars --stats $P/stats_js.npz --workers 10 --compress --out $SUP_H || { echo "a shard failed"; return 1; }  # 32 個分片中的 8 個：聯合投影＋純量，沿用探針特徵庫的尺度
      echo "parts in $SUP_H: $(ls $SUP_H | grep -c '[0-9]\.npz$'), $(du -sh $SUP_H | cut -f1)"; df -h /work/u2507115 /home/u2507115 | tail -2 ;;  # 回報分段檔數、大小與剩餘空間
    check)  # 核對三個特徵庫
      python -I - "$SEL" "$PLY_W" "$PLY_H" "$SUP_H" <<'EOF'  # 核對：缺逐手特徵或缺第二套特徵列的選取局都要少於 0.1%
import csv, glob, sys
import numpy as np
sel, ply_w, ply_h, sup = sys.argv[1:5]
csv.field_size_limit(10 ** 9)
with open(sel, newline="") as fh:
    rows = [(r["game_key"], r["account"]) for r in csv.DictReader(fh)]
want = [k for k, _ in rows]
print(f"selection: {len(want)} games ({len(set(want))} distinct keys), {len(set(a for _, a in rows))} accounts")
got = []
for d in (ply_w, ply_h):
    fs = sorted(f for f in glob.glob(d + "/ply_x_*.npz") if not f.endswith(".tmp.npz"))
    ks = np.concatenate([np.load(f)["key"].astype(str) for f in fs]) if fs else np.array([], str)
    got.append(ks)
    print(f"{d}: {len(fs)} parts, {len(ks)} games")
allk = np.concatenate(got)
print(f"ply: {len(allk)} games, {len(set(allk.tolist()))} distinct, {len(set(want) - set(allk.tolist()))} selected games without a part row")
fs = sorted(f for f in glob.glob(sup + "/sup_*.npz") if not f.endswith(".tmp.npz"))
n = 0; with_plies = 0; keys = set()
for f in fs:
    z = np.load(f)
    k = z["keys"].astype(str); keys.update(k.tolist()); n += len(k); with_plies += int((z["n"] > 0).sum())
print(f"{sup}: {len(fs)} parts, {n} games ({with_plies} with plies), {len(set(want) - keys)} selected games without a row")
ok = len(set(want) - set(allk.tolist())) < 0.001 * len(want) and len(set(want) - keys) < 0.001 * len(want)
print("CHECK", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
EOF
      ;;  # 結束 check 階段
    *) echo "usage: ext_fox_scale3b.sh ingest <folder> | select | ply <first shard> work|home | sup <first shard> | check"; return 2 ;;  # 不認得的階段
  esac  # 結束階段分派
}  # 結束 main
main "$@"; exit $?  # 執行主流程並以它的結果結束
