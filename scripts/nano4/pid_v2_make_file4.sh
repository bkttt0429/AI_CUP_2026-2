#!/usr/bin/env bash
# 同 pid_v2_make_file2.sh，再多一個參數：METHOD_NOTES 2.123（探針 S3）的 1 局題先驗：
#   pid_v2_make_file4.sh <文件名稱> <上傳檔名> <lam> <tau> <w3 | none> [T，預設 1.6] [要比較的檔]
#     lam, tau  多局題的配對（2 0.5 = 「a2k」檔）
#     w3        1 局題的擁有者是被配給 3 局題的候選人之一的機率
#               （獨立抽樣：1/6；估計 0.014 [0.001, 0.062]；0.05 = 保守的設定）。"none" = 不用先驗
#               （必須與 pid_v2_make_file2.sh 的檔逐位元相同）
# 印出這個檔和定案檔（pid_v2o13c_9s.csv）以及選用的比較檔的差異。
set -euo pipefail  # 任何錯誤即中止，未定義變數即報錯，管線中任一步失敗即算失敗
R2=/home/u2507115/BK_AICUP_2026-2  # 儲存庫位置
doc=$1; name=$2; lam=$3; tau=$4; w3=$5; temp=${6:-1.6}; cmp=${7:-}  # 文件、檔名、配對參數、先驗機率與溫度、比較檔
out=$R2/results/submission_${name%.csv}/submission_player.csv  # 預測程式的輸出位置
opts=(--assign big --big-lam "$lam" --big-tau "$tau")  # 多局題：軟性配對＋Sinkhorn 價格
[ "$w3" = none ] || opts+=(--one-prior3 "$w3" --prior-T "$temp")  # 加上 1 局題的先驗
python -I $R2/scripts/predict_pid_v2.py --tests-dir $R2/data/tests_1111 --fusion $R2/results/pid_fusion_$doc.json --trunk-store $R2/data/trunk/tests_1111_v2 --old-submission $R2/results/submission_old_embt_assign/submission_player.csv "${opts[@]}" --out $out  # 新增池由融合文件預測，原題池 800 列沿用舊提交檔
cp $out $R2/upload/$name  # 複製到 upload/
python -I $R2/scripts/pid_diff_files.py $R2/upload/pid_v2o13c_9s.csv $R2/upload/$name  # 與當時的定案檔比較
[ -z "$cmp" ] || python -I $R2/scripts/pid_diff_files.py $R2/upload/$cmp $R2/upload/$name  # 有指定時再與比較檔比較
echo "md5 $(md5sum $R2/upload/$name | cut -c1-32) $name"  # 印出新檔的 md5
