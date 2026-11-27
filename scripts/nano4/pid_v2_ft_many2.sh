#!/usr/bin/env bash
# 在一個工作內對我自己的多個序列網路做題池微調（METHOD_NOTES 2.32、2.42），已存在的略過。
#   pid_v2_ft_many.sh real   "<網路> <後綴>" ...   實際的新增池：0 / 50 步後的向量（程式只跑 --steps 0 50）-> data/seq_ft
#   pid_v2_ft_many.sh offsim "<網路> <後綴>" ...   官方擬題池（每位候選人留出 3 局，種子 0 與 1）：
#                                                   0 / 50 步後的向量 -> /work/u2507115/pid_v2/seq_ft_offsim
# 檢查點為 /work/u2507115/pid_v2/seqnet/ckpt_<net><suffix>.pt。透過 scripts/nano4/run.sbatch 以一張 GPU 執行。
set -euo pipefail  # 任何錯誤即中止，未定義變數即報錯，管線中任一步失敗即算失敗
R2=/home/u2507115/BK_AICUP_2026-2  # 儲存庫位置
mode=$1  # real 或 offsim
shift  # 其餘參數為各網路的規格
for spec in "$@"; do  # 逐一網路
  read -r v sfx <<< "$spec"  # 拆出網路編號與檢查點後綴
  for pool in pub2 priv2; do  # 兩個新增池分開微調，不混用
    common=(--mode official --pool "$pool" --net "$v" --horizon 1000 --ckpt-dir /work/u2507115/pid_v2/seqnet --ckpt-suffix "$sfx" --tag "$sfx")  # 共同選項：官方題池、1000 步的餘弦排程、檢查點位置
    if [ "$mode" = real ]; then  # 正式的新增池
      out=$R2/data/seq_ft  # 正式向量的資料夾
      [ -s "$out/official_${pool}_net${v}${sfx}_t50.npz" ] || python -I $R2/scripts/pid_v2_seq_finetune.py "${common[@]}" --steps 0 50 --out "$out"  # 還沒有 50 步向量才微調，存 0／50 步的向量
    else  # 擬題池
      out=/work/u2507115/pid_v2/seq_ft_offsim  # 擬題池向量的資料夾
      for s in 0 1; do  # 兩個留局種子
        [ -s "$out/offsim_${pool}_h3s${s}_net${v}${sfx}_t50.npz" ] || python -I $R2/scripts/pid_v2_seq_finetune.py "${common[@]}" --steps 0 50 --holdout 3 --holdout-seed "$s" --out "$out"  # 每位候選人留出 3 局當擬題，其餘拿來微調
      done  # 結束種子迴圈
    fi  # 結束模式判斷
  done  # 結束題池迴圈
done  # 結束網路迴圈
