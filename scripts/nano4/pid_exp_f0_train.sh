#!/usr/bin/env bash
# METHOD_NOTES 2.201：四個照局數抽樣的家族「納入第 0 折」來訓練（使用者 2026-11-18 00:29 的同意，「同意用 fold 0」；
# 訓練程式選項 --all-folds：第 0 折的 1,605 位玩家和第 1-4 折的玩家一樣參加訓練）。配方、步數、抽樣
# 與外部帳號（scale3b，不含 scale4）完全照第二視窗那一輪的家族（experiments/nano4/pid_exp_w2fam_train2.sh；
# 9x9 視窗 + 第二個視窗，scripts/pid_win2.py），種子則是新的：這些家族是要「加入」組成的。
#   o44 = o41 的配方 + 第 0 折（一般輸入，40,000 步）
#   o45 = o42 的配方 + 第 0 折（第二套特徵，40,000 步）
#   o46 = o43 的配方 + 第 0 折（第二套特徵，512x10，30,000 步）
#   o47 = o34 的配方 + 第二個視窗 + 第 0 折（第二套特徵，30,000 步；三個成員照 linear 抽樣，三個照 sqrt）
# 第 0 折棋局的第二套特徵在 /work/u2507115/pid_v2/supj/comp0（scripts/nano4/pid_exp_f0_sup.sh）；這套特徵的
# 網路以 --sup-require 訓練，所以特徵庫通過核對之前，啟動腳本拒絕啟動它們。
# 節點的 8 張 GPU 平均分給指定的成員（1 個成員 = 8 張 GPU，2 = 各 4 張，4 = 各 2 張，
# 8 = 各「一張」：沒有多卡前向，給多卡檢查失敗的節點用）。
#   pid_exp_f0_train.sh <o44|o45|o46|o47> "<成員> [<成員> ...]"        成員 0-7
# 訓練結束後，啟動腳本在每個成員的紀錄裡確認第 0 折真的有參加訓練。
# 只有 EMA，沒有狀態檔、沒有快照。從 AI_CUP_2026-1 的工作目錄執行，8 張 GPU。有工作在執行它時不要修改。
set -uo pipefail  # 未定義變數即報錯，管線中任一步失敗即算失敗
main() {  # 主體包成函式，bash 會先讀完整個函式再執行
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj  # 儲存庫、外部資料、投影與第二套特徵的位置
  local fam=$1 members=$2 J=${SLURM_JOB_ID:-local} WIN=${WIN:-9}  # 家族、成員清單、工作編號、視窗大小（預設 9）
  local PLY_W=$X/scale3b/ply PLY_H=$R2/data/ext_ply2/scale3b SUP_H=$R2/data/supj_ext/scale3b  # scale3b 的兩處逐手特徵庫與第二套特徵庫
  local d i k n per slot set free cards lines=() tags=() C tag extra seed mem rc spec data  # 迴圈與設定用的區域變數
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d $PLY_W $PLY_H --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv $X/scale3b/select.csv"  # 全部外部帳號：八個逐手特徵資料夾與五個選取檔
  local SUP="--sup-ply $P/ply_js $O/comp $O/comp0 $O/ext $SUP_H --sup-stats $P/stats_js.npz"  # 第二套特徵：探針、競賽棋局（含第 0 折的 comp0）與外部帳號的特徵庫，加上尺度檔
  local JS="--ext-share 0.5 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --sup-name js --sup-dims 0 68"  # 第二套特徵配方的共同選項
  for d in $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d $PLY_W $PLY_H; do  # 逐一檢查外部逐手特徵資料夾
    [ "$(ls "$d" 2> /dev/null | grep -c 'ply_x_.*[0-9]\.npz$')" -ge 200 ] || { echo "external ply folder $d is missing or incomplete"; return 1; }  # 缺少或不完整（不到 200 個分段檔）就不啟動；讀檔端只擋完全沒有分段檔的資料夾
  done  # 結束外部資料檢查
  [ -e "$X/scale3b/CHECK_PASSED" ] || { echo "the scale3b stores have not passed their check"; return 1; }  # scale3b 特徵庫必須已通過核對
  for d in results/trunk_vec results/trunk_vec_x40 results/trunk_vec_x80; do  # 檢查三個主幹向量資料夾裡第 0 折的分段檔
    [ "$(ls "$d/fold0" 2> /dev/null | grep -c 'games_s.*\.csv$')" -ge 200 ] || { echo "the stored parts of fold 0 in $d are missing"; return 1; }  # 不到 200 個就不啟動
  done  # 結束分段檔檢查
  case "$fam" in o44|o45|o46|o47) ;; *) echo "unknown family $fam"; return 2 ;; esac  # 只接受這四個家族
  if [ "$fam" != o44 ] && [ ! -e $O/comp0.CHECK_PASSED ]; then  # 第二套特徵的家族需要 comp0 已通過核對
    bash $R2/scripts/nano4/pid_exp_f0_sup.sh check || { echo "the second feature set of fold 0 has not passed its check: not started"; return 1; }  # 還沒有標記檔時先核對一次
  fi  # 結束 comp0 檢查
  mem=$(free -g | awk 'NR==2{print $4}')  # 節點的空閒記憶體（GB）
  [ "$mem" -ge "${MIN_FREE_GB:-200}" ] || { echo "node $(hostname) has only $mem GB of free memory (page cache full): not started"; return 1; }  # 快取佔滿的節點載入很慢，不啟動
  free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')  # /work 的剩餘空間（MB）
  [ "$free" -ge 5000 ] || { echo "/work has only $free MB free: not started (the rank session's training needs 4 GB)"; return 1; }  # 替棋力那邊的訓練保留空間
  IFS=, read -r -a cards <<< "${CUDA_VISIBLE_DEVICES:?}"  # 本工作的 GPU 清單
  [ "${#cards[@]}" -eq 8 ] || { echo "expected eight GPUs"; return 1; }  # 需要整個節點的 8 張卡
  python -I -m py_compile $R2/scripts/pid_win.py $R2/scripts/pid_win2.py $R2/scripts/pid_seq_train_w2.py || { echo "a script does not compile"; return 1; }  # 先確認訓練程式與視窗模組可以編譯
  # shellcheck disable=SC2086 # $members 刻意不加引號，拆成多個成員編號
  set -- $members  # 把成員清單放進位置參數
  n=$#  # 成員數
  case "$n" in 1|2|4|8) ;; *) echo "1, 2, 4 or 8 members per node"; return 2 ;; esac  # 成員數必須能平分 8 張卡
  per=$((8 / n))  # 每個成員分到的 GPU 數
  slot=0  # 目前分配到第幾組 GPU
  for i in "$@"; do  # 逐一成員設定
    case "$i" in 0|1|2|3|4|5|6|7) ;; *) echo "member $i: 0-7"; return 2 ;; esac  # 成員編號只能是 0-7
    case "$fam" in  # 依家族設定配方、標籤與種子
      o44) C="--steps 40000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --ema-only --own-by-games linear"  # 一般輸入：4 萬步、對稱、遮手 0.3、照局數 linear
           if [ "$i" -lt 4 ]; then tag=o44Ss$i; extra=""; else tag=o44SMs$((i - 4)); extra="--drop-open 0.3"; fi  # 0-3 為 Ss，4-7 為遮開局的 SMs
           seed=$((300 + i)) ;;  # 種子 300-307
      o45) C="--steps 40000 --horizon 60000 $JS --own-by-games linear"  # 第二套特徵：4 萬步、照局數 linear
           if [ "$i" -lt 6 ]; then tag=o45As$i; extra=""; else tag=o45Cs$((i - 6)); extra="--sup-only"; fi  # 0-5 為 As，6、7 為只用有監督特徵的 Cs
           seed=$((310 + i)) ;;  # 種子 310-317
      o46) C="--steps 30000 --horizon 60000 $JS --arch 512,10,8,2048 --own-by-games linear"  # 第二套特徵、512x10：3 萬步
           if [ "$i" -lt 6 ]; then tag=o46As$i; extra=""; else tag=o46Cs$((i - 6)); extra="--sup-only"; fi  # 0-5 為 As，6、7 為 Cs
           seed=$((320 + i)) ;;  # 種子 320-327
      o47) C="--steps 30000 --horizon 60000 $JS"  # 第二套特徵：3 萬步，抽樣方式依成員而定
           if [ "$i" -lt 3 ]; then tag=o47As$i; extra="--own-by-games linear"  # 成員 0-2：照局數 linear
           elif [ "$i" -lt 6 ]; then tag=o47As$i; extra="--own-by-games sqrt"  # 成員 3-5：照局數 sqrt
           elif [ "$i" -eq 6 ]; then tag=o47Cs0; extra="--sup-only --own-by-games linear"  # 成員 6：Cs0、linear
           else tag=o47Cs1; extra="--sup-only --own-by-games sqrt"; fi  # 成員 7：Cs1、sqrt
           seed=$((330 + i)) ;;  # 種子 330-337
    esac  # 結束家族設定
    extra="$extra --shape-prev 1"  # 加上第二個視窗（前一手落點周圍，有自己的編碼器）
    [ -e /work/u2507115/pid_v2/seqnet/ckpt_-1_${tag}e.pt ] && { echo "ckpt of $tag exists already"; return 1; }  # 不覆蓋已訓練好的檢查點
    set=""  # 這個成員的 GPU 清單
    for k in $(seq 0 $((per - 1))); do set="$set${set:+,}${cards[$((per * slot + k))]}"; done  # 依序取 per 張卡，以逗號串起
    if [ "$per" -gt 1 ]; then  # 多卡的成員
      CUDA_VISIBLE_DEVICES="$set" timeout ${DP_TIMEOUT:-240} python -u -c 'import torch; m=torch.nn.Linear(4,4).cuda(); y=torch.nn.DataParallel(m)(torch.ones(64,4,device="cuda")); y.sum().backward(); torch.cuda.synchronize(); print("cards ok:", torch.cuda.device_count())' || { echo "GPU set $set on $(hostname) failed the multi-card check"; return 1; }  # 多卡 DataParallel 試跑，壞節點及早停下
    else  # 單卡的成員
      CUDA_VISIBLE_DEVICES="$set" timeout ${DP_TIMEOUT:-240} python -u -c 'import torch; m=torch.nn.Linear(4,4).cuda(); y=m(torch.ones(64,4,device="cuda")); y.sum().backward(); torch.cuda.synchronize(); print("card ok:", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory >> 30, "GB")' || { echo "GPU $set on $(hostname) failed the single-card check"; return 1; }  # 單卡試跑並印出顯示卡記憶體
    fi  # 結束 GPU 檢查
    lines+=("$set logs/nano4/pidseq_-1_${tag}-$J.log --net -1 --tag $tag $C $extra --shape-win $WIN --seed $seed")  # 規格檔的一列：GPU、紀錄檔與訓練參數
    tags+=("$tag")  # 記下成員標籤
    slot=$((slot + 1))  # 下一組 GPU
  done  # 結束成員迴圈
  spec=$(mktemp "${TMPDIR:-/tmp}/f0_spec_XXXXXX")  # 暫存的規格檔
  printf '%s\n' "${lines[@]}" > "$spec"  # 每個成員一列
  cat "$spec"  # 印出規格以便追查
  echo "node $(hostname): family $fam members $members, $n model(s) x $per GPU(s), FOLD 0 TRAINED ON; memory $(free -g | awk 'NR==2{print $4" GB free, "$6" GB cache"}'); /work free $free MB; $(date '+%F %T')"  # 回報節點、成員、記憶體與空間
  data="$OFF"  # 先只放外部帳號的逐手特徵與選取檔（一般輸入的 o44 到此為止）
  [ "$fam" = o44 ] || data="$OFF $SUP"  # 其餘家族再加第二套特徵
  # shellcheck disable=SC2086 # $data 刻意不加引號，拆成多個參數
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_seq_train_w2.py --fork "$spec" --all-folds $data  # 含第 0 折；資料只載入一次，為每個成員分叉一個訓練行程
  rc=$?  # 訓練的結束碼
  rm -f "$spec"  # 刪除暫存的規格檔
  grep -hE "training on|supervised per-ply|training forward" logs/nano4/pidseq_-1_${fam}*-$J.log | sort -u | cut -c1-260  # 摘錄各成員的資料與設定紀錄
  for tag in "${tags[@]}"; do  # 訓練的折必須包含第 0 折（缺少特徵庫時會靜靜地把它排除）
    grep -qE "training on .*folds \[-9, 0, 1, 2, 3, 4\]" logs/nano4/pidseq_-1_${tag}-$J.log || { echo "ALERT: the log of $tag does not show fold 0 among the training folds"; rc=1; }  # 紀錄裡沒有第 0 折就警示
  done  # 結束紀錄核對
  echo "checkpoints of $fam so far: $(ls /work/u2507115/pid_v2/seqnet | grep -c "ckpt_-1_${fam}.*e\.pt"); $(date '+%F %T')"  # 回報這個家族目前的檢查點數
  return $rc  # 回傳結果
}  # 結束 main
main "$@"; exit $?  # 執行主流程並以它的結果結束
