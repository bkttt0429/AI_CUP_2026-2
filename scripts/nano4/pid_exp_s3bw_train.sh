#!/usr/bin/env bash
# METHOD_NOTES 2.177（合併重訓）：定案組成裡四個照局數抽樣的家族，同時帶上各自單獨對家族有幫助的
# 兩個小改動重訓——更多外部帳號（scale3b，同 o27-o30）「以及」o31 的視窗輸入
# （每一手落點周圍 WIN x WIN 視窗內的棋子；scripts/pid_win.py，訓練程式 scripts/pid_seq_train_w.py）。
# 配方、種子、步數與抽樣同 o27-o30（= 原版），所以每個網路都有一個沒有視窗的孿生。
#   o32 = o27 + 視窗（一般輸入，40,000 步）            o33 = o28 + 視窗（第二套特徵，40,000 步）
#   o34 = o29 + 視窗（第二套特徵，30,000 步）      o35 = o30 + 視窗（第二套特徵，512x10，30,000 步）
# 一個工作 = 一個家族的四個模型，每個模型兩張 GPU。
# 每個家族的前兩個成員另外寫出快照（ckpt_-1_<tag>k<step/1000>e.pt），用來看沿著訓練步數的
# 軌跡；以 pid_exp_snapread.sh 讀取後刪除。
#   [WIN=9] pid_exp_s3bw_train.sh <o32|o33|o34|o35> <組 0|1>       第 0 組 = 成員 0-3，第 1 組 = 成員 4-7
# 只有 EMA，沒有狀態檔。從不讀第 0 折。從 AI_CUP_2026-1 的工作目錄執行，8 張 GPU。有工作在執行它時不要修改。
set -uo pipefail  # 未定義變數即報錯，管線中任一步失敗即算失敗
main() {  # 主體包成函式，bash 會先讀完整個函式再執行
  local R2=/home/u2507115/BK_AICUP_2026-2 X=/work/u2507115/ext_fox P=/work/u2507115/pid_v2/probep O=/work/u2507115/pid_v2/supj fam=$1 grp=$2 J=${SLURM_JOB_ID:-local} WIN=${WIN:-9}  # 各資料位置、家族、成員組、工作編號、視窗大小（預設 9）
  local PLY_W=$X/scale3b/ply PLY_H=$R2/data/ext_ply2/scale3b SUP_H=$R2/data/supj_ext/scale3b d i k slot pair free cards lines=() C tag extra seed  # scale3b 的特徵庫與迴圈、設定用的區域變數
  local OFF="--ext-ply $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d $PLY_W $PLY_H --ext-select $X/pilot/select.csv $X/scale1/select.csv $X/scale2_top/select.csv $X/scale2_5d/select.csv $X/scale3b/select.csv"  # 全部外部帳號：八個逐手特徵資料夾與五個選取檔
  local SUP="--sup-ply $P/ply_js $O/comp $O/ext $SUP_H --sup-stats $P/stats_js.npz"  # 第二套特徵：探針、競賽棋局與外部帳號的特徵庫及尺度檔；沒有第 0 折的 comp0
  local JS="--ext-share 0.5 --no-state --sym-aug --mask 0.3 --sup-require --ema-only --sup-name js --sup-dims 0 68"  # 第二套特徵配方的共同選項
  for d in $X/pilot/ply $X/scale1/ply_a $X/scale1/ply_b $R2/data/ext_ply/scale1_c $R2/data/ext_ply/scale2_top $R2/data/ext_ply/scale2_5d $PLY_W $PLY_H; do  # 逐一檢查外部逐手特徵資料夾
    [ "$(ls "$d" 2> /dev/null | grep -c 'ply_x_.*[0-9]\.npz$')" -ge 200 ] || { echo "external ply folder $d is missing or incomplete"; return 1; }  # 缺少或不完整（不到 200 個分段檔）就不啟動；讀檔端只擋完全沒有分段檔的資料夾
  done  # 結束外部資料檢查
  [ -e "$X/scale3b/CHECK_PASSED" ] || { echo "the scale3b stores have not passed their check (ext_fox_scale3b.sh check)"; return 1; }  # scale3b 特徵庫必須已通過核對
  free=$(df -BM --output=avail /work/u2507115 | tail -1 | tr -dc '0-9')  # /work 的剩餘空間（MB）
  [ "$free" -ge 5000 ] || { echo "/work has only $free MB free: not started (the rank session's training needs 4 GB)"; return 1; }  # 替棋力那邊的訓練保留空間
  IFS=, read -r -a cards <<< "${CUDA_VISIBLE_DEVICES:?}"  # 本工作的 GPU 清單
  python -I -m py_compile $R2/scripts/pid_win.py $R2/scripts/pid_seq_train_w.py || { echo "a script does not compile"; return 1; }  # 先確認訓練程式與視窗模組可以編譯
  case "$WIN" in 5|7|9|11|13) ;; *) echo "WIN=$WIN is not an odd size between 5 and 13"; return 2 ;; esac  # 視窗大小必須是 5 到 13 的奇數
  [ "${#cards[@]}" -eq 8 ] || { echo "expected eight GPUs"; return 1; }  # 需要整個節點的 8 張卡
  for slot in 0 1 2 3; do  # 逐一檢查四組雙卡
    pair=${cards[$((2 * slot))]},${cards[$((2 * slot + 1))]}  # 這一組的兩張卡
    CUDA_VISIBLE_DEVICES="$pair" timeout 600 python -u -c 'import torch; m=torch.nn.Linear(4,4).cuda(); y=torch.nn.DataParallel(m)(torch.ones(8,4,device="cuda")); y.sum().backward(); torch.cuda.synchronize()' || { echo "GPU pair $pair on $(hostname) failed the two-card check"; return 1; }  # 雙卡 DataParallel 試跑，壞節點及早停下
  done  # 結束雙卡檢查
  slot=0  # 從第一組雙卡開始分配
  for i in $((4 * grp)) $((4 * grp + 1)) $((4 * grp + 2)) $((4 * grp + 3)); do  # 這一組的四個成員
    case "$fam" in  # 依家族設定配方、標籤與種子
      o32) C="--steps 40000 --horizon 60000 --ext-share 0.5 --no-state --sym-aug --mask 0.3 --ema-only --own-by-games linear"  # o27 的配方：一般輸入、4 萬步
           if [ "$i" -lt 4 ]; then tag=o32Ss$i; extra=""; seed=$((180 + i)); else tag=o32SMs$((i - 4)); extra="--drop-open 0.3"; seed=$((180 + i)); fi ;;  # 0-3 為 Ss，4-7 為遮開局的 SMs；種子 180-187
      o33) C="--steps 40000 --horizon 60000 $JS --own-by-games linear"  # o28 的配方：第二套特徵、4 萬步
           if [ "$i" -lt 6 ]; then tag=o33As$i; extra=""; seed=$((200 + i)); else tag=o33Cs$((i - 6)); extra="--sup-only"; seed=$((210 + i - 6)); fi ;;  # As 種子 200-205，只用有監督特徵的 Cs 種子 210-211
      o34) C="--steps 30000 --horizon 60000 $JS"  # o29 的配方：第二套特徵、3 萬步，抽樣方式依成員而定
           if [ "$i" -lt 3 ]; then tag=o34As$i; extra="--own-by-games linear"; seed=$((140 + i))  # 成員 0-2：照局數 linear
           elif [ "$i" -lt 6 ]; then tag=o34As$i; extra="--own-by-games sqrt"; seed=$((140 + i))  # 成員 3-5：照局數 sqrt
           elif [ "$i" -eq 6 ]; then tag=o34Cs0; extra="--sup-only --own-by-games linear"; seed=150  # 成員 6：Cs0、linear
           else tag=o34Cs1; extra="--sup-only --own-by-games sqrt"; seed=151; fi ;;  # 成員 7：Cs1、sqrt
      o35) C="--steps 30000 --horizon 60000 $JS --arch 512,10,8,2048 --own-by-games linear"  # o30 的配方：第二套特徵、512x10、3 萬步
           if [ "$i" -lt 6 ]; then tag=o35As$i; extra=""; seed=$((160 + i)); else tag=o35Cs$((i - 6)); extra="--sup-only"; seed=$((170 + i - 6)); fi ;;  # As 種子 160-165，Cs 種子 170-171
      *) echo "unknown family $fam"; return 2 ;;  # 不認得的家族
    esac  # 結束家族設定
    [ -e /work/u2507115/pid_v2/seqnet/ckpt_-1_${tag}e.pt ] && { echo "ckpt of $tag exists already"; return 1; }  # 不覆蓋已訓練好的檢查點
    if [ "$i" -lt 2 ]; then  # 前兩個成員另外存快照
      case "$fam" in o32) extra="$extra --snap 10000 20000 30000" ;; o33) extra="$extra --snap 30000 35000" ;;  # o32 與 o33 的快照步數
                     o34) extra="$extra --snap 10000 15000 20000 25000" ;; o35) extra="$extra --snap 5000 10000 15000 20000" ;; esac  # o34 與 o35 的快照步數
    fi  # 結束快照設定
    pair=${cards[$((2 * slot))]},${cards[$((2 * slot + 1))]}  # 這個成員用的兩張卡
    lines+=("$pair logs/nano4/pidseq_-1_${tag}-$J.log --net -1 --tag $tag $C $extra --shape-win $WIN --seed $seed")  # 規格檔的一列：GPU、紀錄檔與訓練參數
    slot=$((slot + 1))  # 下一組雙卡
  done  # 結束成員迴圈
  local spec  # 規格檔路徑
  spec=$(mktemp "${TMPDIR:-/tmp}/s3b_spec_XXXXXX")  # 暫存的規格檔
  printf '%s\n' "${lines[@]}" > "$spec"  # 每個成員一列
  cat "$spec"  # 印出規格以便追查
  echo "node $(hostname): family $fam group $grp, four models x two GPUs; memory $(free -g | awk 'NR==2{print $4" GB free, "$6" GB cache"}'); /work free $free MB"  # 回報節點、記憶體與空間
  local data="$OFF"  # 先只放外部帳號的逐手特徵與選取檔（一般輸入的 o32 到此為止）
  [ "$fam" = o32 ] || data="$OFF $SUP"  # 其餘家族再加第二套特徵
  # shellcheck disable=SC2086 # $data 刻意不加引號，拆成多個參數
  # 帶視窗輸入的雙卡訓練程式（scripts/pid_seq_train_w.py = pid_seq_train_dp.py + --shape-win）
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -I $R2/scripts/pid_seq_train_w.py --fork "$spec" $data  # 資料只載入一次，為每個成員分叉一個訓練行程（不含第 0 折）
  local rc=$?  # 訓練的結束碼
  rm -f "$spec"  # 刪除暫存的規格檔
  grep -hE "training on|supervised per-ply" logs/nano4/pidseq_-1_${fam}*-$J.log | sort -u | cut -c1-260  # 摘錄各成員的資料紀錄
  ls -la /work/u2507115/pid_v2/seqnet | grep -c "ckpt_-1_${fam}.*e\.pt"  # 清點這個家族已寫出的檢查點
  return $rc  # 回傳結果
}  # 結束 main
main "$@"; exit $?  # 執行主流程並以它的結果結束
