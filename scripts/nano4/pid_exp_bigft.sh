#!/usr/bin/env bash
# METHOD_NOTES 2.204：單張 140 GB 的卡放不下的題池微調。帶第二個視窗的 512x10 網路
# （家族 o43F、o46F）在 256 位玩家 x 8 局的一個情節需要超過 139 GB：正式題池與原題池模擬
# 都以 "CUDA out of memory" 失敗（工作 793567 的 vec_o43b，2026-11-18 03:33），而擬題池（每位
# 候選人 7 局）剛好放得下。這裡每個這類微調分配「兩張」卡（pid_v2_seq_finetune.py --dp：訓練的前向
# 分到兩張卡上，損失留在第一張），一次跑四個微調。選項與標準鏈相同
# （pid_v2_ft_many2.sh real / pid_exp_oldsim.sh）；已存在的檔案略過。
#   pid_exp_bigft.sh <家族> <real|old> [<輸出資料夾>]
#     real  official_<pool>_net-1_<family>s<k>e_t{0,50}.npz     預設輸出資料夾：data/seq_ft
#     old   oldsim_<side>_s0_net-1_<family>s<k>e_t{0,50}.npz    預設輸出資料夾：data/pid_v2_seq_ft_oldsim
# o44F-o47F 的成員連結缺少時在這裡建立（scripts/nano4/pid_exp_f0_train.sh 的標籤）。
# 從 AI_CUP_2026-1 的工作目錄執行，8 張 GPU。有工作在執行它時不要修改。
set -uo pipefail  # 未定義變數即報錯，管線中任一步失敗即算失敗
main() {  # 主體包成函式，bash 會先讀完整個函式再執行
  local R2=/home/u2507115/BK_AICUP_2026-2 SN=/work/u2507115/pid_v2/seqnet fam=$1 what=$2 J=${SLURM_JOB_ID:-local}  # 儲存庫、檢查點資料夾、家族、模式與工作編號
  local out G i k w n t a f free rc=0 tags=() tasks=() pids=() t0=$SECONDS  # 迴圈變數、成員標籤、待做的微調與計時
  IFS=, read -r -a G <<< "${CUDA_VISIBLE_DEVICES:?}"  # 本工作的 GPU 清單
  [ "${#G[@]}" -eq 8 ] || { echo "expected eight GPUs"; return 1; }  # 需要整個節點的 8 張卡
  case "$what" in  # 依模式決定預設輸出資料夾
    real) out=${3:-$R2/data/seq_ft} ;;  # 正式題池（新增池）的向量
    old)  out=${3:-$R2/data/pid_v2_seq_ft_oldsim} ;;  # 原題池模擬的向量
    *) echo "usage: pid_exp_bigft.sh <family> <real|old> [<out dir>]"; return 2 ;;  # 模式錯誤
  esac  # 結束模式選擇
  case "$fam" in  # 含第 0 折的家族 o44F-o47F 的成員連結（缺少時補建）
    o44F) for i in 0 1 2 3; do [ -s $SN/ckpt_-1_o44Ss${i}e.pt ] && ln -sf ckpt_-1_o44Ss${i}e.pt $SN/ckpt_-1_o44Fs${i}e.pt  # o44 的 Ss<i> -> 成員 i
                               [ -s $SN/ckpt_-1_o44SMs${i}e.pt ] && ln -sf ckpt_-1_o44SMs${i}e.pt $SN/ckpt_-1_o44Fs$((i + 4))e.pt; done ;;  # SMs<i> -> 成員 i+4
    o45F|o46F|o47F) for i in 0 1 2 3 4 5; do [ -s $SN/ckpt_-1_${fam%F}As${i}e.pt ] && ln -sf ckpt_-1_${fam%F}As${i}e.pt $SN/ckpt_-1_${fam}s${i}e.pt; done  # As<i> -> 成員 0-5
                    for i in 0 1; do [ -s $SN/ckpt_-1_${fam%F}Cs${i}e.pt ] && ln -sf ckpt_-1_${fam%F}Cs${i}e.pt $SN/ckpt_-1_${fam}s$((i + 6))e.pt; done ;;  # Cs0／Cs1 -> 成員 6／7
  esac  # 其他家族不建連結
  for k in $(seq 0 23); do [ -s $SN/ckpt_-1_${fam}s${k}e.pt ] && tags+=("_${fam}s${k}e"); done  # 收集存在的成員檢查點標籤
  [ "${#tags[@]}" -ge 7 ] || { echo "family $fam: only ${#tags[@]} linked checkpoints"; return 1; }  # 成員不足 7 個就不啟動
  mkdir -p "$out" || return 1  # 建立輸出資料夾
  free=$(df -BM --output=avail "$out" | tail -1 | tr -dc '0-9')  # 輸出位置的剩餘空間（MB）
  [ "$free" -ge 2500 ] || { echo "$out has only $free MB free: not started"; return 1; }  # 不足 2.5 GB 就不啟動
  for t in "${tags[@]}"; do  # 逐一成員列出待做的微調
    if [ "$what" = real ]; then  # 正式題池
      for a in pub2 priv2; do [ -s "$out/official_${a}_net-1${t}_t50.npz" ] || tasks+=("$t $a"); done  # 兩個新增池中還沒有 50 步向量的列入待做
    else  # 原題池模擬
      for a in pub priv; do [ -s "$out/oldsim_${a}_s0_net-1${t}_t50.npz" ] || tasks+=("$t $a"); done  # 兩邊原題池模擬中還沒有 50 步向量的列入待做
    fi  # 結束模式判斷
  done  # 結束成員迴圈
  echo "family $fam ($what): ${#tags[@]} networks, ${#tasks[@]} fine-tunes to do on 4 x 2 GPUs -> $out; node $(hostname); $(date '+%F %T')"  # 回報成員數、待做數與節點
  [ "${#tasks[@]}" -gt 0 ] || { echo "nothing to do"; return 0; }  # 全部已存在就結束
  python -I -m py_compile $R2/scripts/pid_v2_seq_finetune.py || return 1  # 先確認微調程式可以編譯
  for w in 0 1 2 3; do  # 逐一檢查四組雙卡
    CUDA_VISIBLE_DEVICES="${G[$((2 * w))]},${G[$((2 * w + 1))]}" timeout ${DP_TIMEOUT:-240} python -u -c 'import torch; m=torch.nn.Linear(4,4).cuda(); y=torch.nn.DataParallel(m)(torch.ones(64,4,device="cuda")); y.sum().backward(); torch.cuda.synchronize(); print("cards ok:", torch.cuda.device_count())' || { echo "GPU pair ${G[$((2 * w))]},${G[$((2 * w + 1))]} on $(hostname) failed the two-card check"; return 1; }  # 雙卡 DataParallel 試跑，壞節點及早停下
  done  # 結束雙卡檢查
  for w in 0 1 2 3; do  # 四個背景工作者，各用一組雙卡
    (  # 子殼層：依序跑分到的微調
      for i in "${!tasks[@]}"; do  # 走過全部待做的微調
        [ $(( i % 4 )) -eq "$w" ] || continue  # 依編號輪流分給四個工作者
        read -r t a <<< "${tasks[$i]}"  # 拆出成員標籤與題池
        if [ "$what" = real ]; then  # 正式題池的微調
          CUDA_VISIBLE_DEVICES="${G[$((2 * w))]},${G[$((2 * w + 1))]}" OMP_NUM_THREADS=8 python -I $R2/scripts/pid_v2_seq_finetune.py --mode official --pool "$a" --net -1 --horizon 1000 --ckpt-dir $SN --ckpt-suffix "$t" --tag "$t" --steps 0 50 --out "$out" --dp < /dev/null || echo "FAILED: real $t $a"  # 雙卡微調，存 0／50 步的向量
        else  # 原題池模擬的微調
          CUDA_VISIBLE_DEVICES="${G[$((2 * w))]},${G[$((2 * w + 1))]}" OMP_NUM_THREADS=8 python -I $R2/scripts/pid_v2_seq_finetune.py --mode oldsim --side "$a" --holdout-seed 0 --net -1 --steps 0 50 --horizon 1000 --out "$out" --ckpt-dir $SN --ckpt-suffix "$t" --tag "$t" --dp < /dev/null || echo "FAILED: old $t $a"  # 雙卡微調原題池模擬，存 0／50 步的向量
        fi  # 結束模式判斷
      done  # 結束這個工作者的微調
    ) > logs/nano4/bigft-$J-$fam-$what-$w.log 2>&1 &  # 每個工作者一份紀錄，背景執行
    pids+=($!)  # 記下行程編號
  done  # 結束啟動
  for f in "${pids[@]}"; do wait "$f" || rc=1; done  # 等四個工作者結束
  n=$(cat logs/nano4/bigft-$J-$fam-$what-*.log | grep -c FAILED)  # 失敗的微調數
  grep -hE 'training forward across|step 50/50' logs/nano4/bigft-$J-$fam-$what-0.log | head -4 | cut -c1-160  # 摘錄雙卡分配與完成的紀錄
  if [ "$what" = real ]; then  # 依模式清點輸出檔
    echo "files: $(ls "$out" | grep -c "^official_.*_net-1_${fam}s[0-9]*e_t50\.npz$") of $(( 2 * ${#tags[@]} ))"  # 正式題池的 50 步向量檔數
  else  # 原題池模擬
    echo "files: $(ls "$out" | grep -c "^oldsim_.*_s0_net-1_${fam}s[0-9]*e_t50\.npz$") of $(( 2 * ${#tags[@]} ))"  # 原題池模擬的 50 步向量檔數
  fi  # 結束清點
  echo "done at $(date +%T) ($(( (SECONDS - t0) / 60 )) min); failures: $n"  # 回報耗時與失敗數
  [ "$n" = 0 ] || { grep -h FAILED logs/nano4/bigft-$J-$fam-$what-*.log | head -5; rc=1; }  # 有失敗就列出並回傳非零
  return $rc  # 回傳結果
}  # 結束 main
main "$@"; exit $?  # 執行主流程並以它的結果結束
