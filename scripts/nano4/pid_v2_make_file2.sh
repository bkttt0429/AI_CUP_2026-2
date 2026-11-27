#!/usr/bin/env bash
# 同 pid_v2_make_file.sh，但多局題的配對方式由參數指定（METHOD_NOTES 2.122）：
#   pid_v2_make_file2.sh <文件名稱> <上傳檔名> <lam | hard | none> [tau]
#     lam      以這個懲罰值做軟性配對（到目前為止的提交檔：1）       hard   硬性的一對一配對
#     tau      以這個溫度計算 Sinkhorn 價格；給 lam 時第一名取軟性配對的結果，給 "none" 時只用 Sinkhorn
# 印出這個檔和定案檔（pid_v2o13c_9s.csv，自 2026-11-15 04:32 起）以及再前一個檔的差異。
set -euo pipefail  # 任何錯誤即中止，未定義變數即報錯，管線中任一步失敗即算失敗
R2=/home/u2507115/BK_AICUP_2026-2  # 儲存庫位置
doc=$1  # 融合文件名稱（results/pid_fusion_<doc>.json）
name=$2  # 上傳檔名
lam=$3  # 軟性配對的懲罰值，或 hard／none
tau=${4:-}  # Sinkhorn 的溫度（可省略）
out=$R2/results/submission_${name%.csv}/submission_player.csv  # 預測程式的輸出位置
opts=(--assign big)  # 多局題的配對
case $lam in hard|none) ;; *) opts+=(--big-lam "$lam") ;; esac  # 給數值時改用軟性配對
[ -n "$tau" ] && opts+=(--big-tau "$tau")  # 有 tau 時加上 Sinkhorn 價格
[ "$lam" = none ] && [ -z "$tau" ] && { echo "none needs a tau"; exit 2; }  # none 必須搭配 tau
[ "$lam" = hard ] && [ -n "$tau" ] && { echo "hard takes no tau (use a large lam with a tau)"; exit 2; }  # hard 不接受 tau
python -I $R2/scripts/predict_pid_v2.py --tests-dir $R2/data/tests_1111 --fusion $R2/results/pid_fusion_$doc.json --trunk-store $R2/data/trunk/tests_1111_v2 --old-submission $R2/results/submission_old_embt_assign/submission_player.csv "${opts[@]}" --out $out  # 新增池由融合文件預測，原題池 800 列沿用舊提交檔
cp $out $R2/upload/$name  # 複製到 upload/
python -I $R2/scripts/pid_diff_files.py $R2/upload/pid_v2o13c_9s.csv $R2/upload/$name  # 與當時的定案檔比較
python -I $R2/scripts/pid_diff_files.py $R2/upload/pid_v2c2_bs.csv $R2/upload/$name  # 與再前一個檔比較
echo "md5 $(md5sum $R2/upload/$name | cut -c1-32) $name"  # 印出新檔的 md5
