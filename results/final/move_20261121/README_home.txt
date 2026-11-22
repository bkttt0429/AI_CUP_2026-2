玩家辨識（競賽一）搬回個人電腦用的整理 — /home 這一半          2026-11-21

這個資料夾裡全部是「硬連結」：和原本的檔是同一份資料，沒有多佔空間；刪掉這個資料夾不會刪到原檔。

內容（約 4.0 GB）
  home/BK_AICUP_2026-2/upload/                  全部提交檔。定案：pid_v2fin11_c_a2k_p3.csv
                                                （md5 c7397211f0ca0af7225ad1d04566f969，Public 0.9313515）
  home/BK_AICUP_2026-2/results/                 組成文件、判定讀數、出檔報告、每局嵌入模型
  home/BK_AICUP_2026-2/data/seq_ft/             各網路在真題庫上的向量（重建定案檔要用）
  home/BK_AICUP_2026-2/data/trunk/              題庫的主幹特徵（較早的成分要用）
  home/BK_AICUP_2026-2/data/pid_v2_ply_original/  原題池候選人棋局的逐手向量
  home/BK_AICUP_2026-2/METHOD_NOTES.md          筆記（整理當時的版本；最新版以 GitHub 為準）
  home/BK_AICUP_2026-2/docs/PID_FINAL_NUMBERS.md  結案數字總表
  MANIFEST_home.sha256                          每個檔的 sha256

下載（在個人電腦上；傳檔要走 2222 埠）
  sftp -P 2222 nano4
  sftp> get -r -a /home/u2507115/AICUP2_MOVE_20261121 AICUP2_MOVE_20261121_home

核對（在下載回來的資料夾裡）
  sha256sum -c MANIFEST_home.sha256

另一半在 /work/u2507115/AICUP2_MOVE_20261121（檢查點與題庫的逐手輸入，約 15.9 GB）。
