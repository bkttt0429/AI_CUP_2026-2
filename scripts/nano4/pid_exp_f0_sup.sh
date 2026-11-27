#!/usr/bin/env bash
# METHOD_NOTES 2.201（第 0 折；使用者 2026-11-18 00:29 的同意）：第 0 折玩家每一局訓練棋局的
# 「第二套」有監督特徵（"js"，同 scripts/nano4/pid_exp_o14_x.sh comp：相同的投影、純量與尺度）
# -> /work/u2507115/pid_v2/supj/comp0。第二套特徵的網路以 --sup-require 訓練，所以沒有這些特徵的
# 第 0 折棋局會被靜靜地排除在訓練之外。
#   pid_exp_f0_sup.sh ext <起始分片> <分片總數>    在節點的 8 張 GPU 上跑 8 個分片（每個行程「一張」GPU：
#                                                            多卡檢查失敗的節點也能跑）；以分段檔為單位可以續跑
#   pid_exp_f0_sup.sh check                                  清單裡每個鍵在分段檔中都有一列 -> comp0.CHECK_PASSED
# 從 AI_CUP_2026-1 的工作目錄執行，8 張 GPU。有工作在執行它時不要修改；主體是一個函式。
set -uo pipefail  # 未定義變數即報錯，管線中任一步失敗即算失敗
main() {  # 主體包成函式，bash 會先讀完整個函式再執行
  local R2=/home/u2507115/BK_AICUP_2026-2 P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj rc=0 free  # 儲存庫、投影與第二套特徵的位置
  local K=$O/fold0_keys.txt  # 第 0 折全部訓練棋局的鍵清單
  [ -s $P/stats_js.npz ] && [ -s $P/proj3.npz ] || { echo "the projection or the scale of the second feature set is missing"; return 1; }  # 需要與 comp 相同的投影與尺度檔
  free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')  # /work 的剩餘空間（MB）
  [ "$free" -ge 7000 ] || { echo "/work has only $free MB free: not started (the parts need about 1.2 GB, the floor is 4 GB)"; return 1; }  # 空間不夠就不啟動
  [ -s $K ] || python -I $R2/scripts/pid_fold0_keys.py $K || return 1  # 沒有鍵清單就先產生
  echo "key list: $(wc -l < $K) games of fold 0; node $(hostname); $(date '+%F %T')"  # 回報局數與節點
  case $1 in  # 依子命令分派
    ext)  # shellcheck disable=SC2086 # 抽特徵（下一行的路徑變數不加引號）
      bash $R2/scripts/nano4/shards_offset.sh 8 "$2" "$3" python -I $R2/scripts/pid_sup_ply_extract.py --keys $K --proj $P/proj3.npz --scalars --stats $P/stats_js.npz --workers 10 --out $O/comp0; rc=$? ;;  # 本工作跑 8 個分片，每個一張 GPU
    check)  # 核對每個鍵都有特徵列
      python3 -I - $K $O/comp0 <<'PY' || return 1  # 核對分段檔：68 維、鍵不重複、鍵集合與清單完全相同、沒有手數的局不超過 0.5%
import glob, sys
import numpy as np
keys = set(open(sys.argv[1]).read().split())
have, plies, empty = set(), 0, 0
for f in sorted(glob.glob(sys.argv[2] + "/sup_s*_p*.npz")):
    z = np.load(f)
    assert "zq" in z.files and z["zq"].shape[1] == 68 and z["zq"].shape[0] == int(z["n"].sum()) == len(z["t"]), f
    k = z["keys"].astype(str).tolist()
    assert not (set(k) & have), f"a key twice ({f})"
    have.update(k); plies += int(z["n"].sum()); empty += int((z["n"] == 0).sum())
print(f"{len(keys)} keys in the list, {len(have)} in the parts, {len(keys - have)} missing, {len(have - keys)} foreign; {plies} plies, {empty} games without plies")
assert have == keys and empty <= len(keys) // 200
PY
      echo "check passed $(date '+%F %T')" > $O/comp0.CHECK_PASSED ;;  # 通過才寫標記檔，訓練的啟動腳本會檢查它
    *) echo "unknown case $1"; return 2 ;;  # 不認得的子命令
  esac  # 結束分派
  echo "parts: comp0 $(ls $O/comp0 2>/dev/null | grep -c 'npz$'); $(du -sh $O/comp0 | cut -f1); /work free $(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9') MB; $(date '+%F %T')"  # 回報分段檔數、大小與 /work 剩餘空間
  return $rc  # 回傳結果
}  # 結束 main
main "$@"; exit $?  # 執行主流程並以它的結果結束
