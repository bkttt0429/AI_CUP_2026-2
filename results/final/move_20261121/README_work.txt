玩家辨識（競賽一）搬回個人電腦用的整理 — /work 這一半          2026-11-21

這個資料夾裡全部是「硬連結」：和原本的檔是同一份資料，沒有多佔空間；刪掉這個資料夾不會刪到原檔。

內容（約 15.9 GB）
  work/pid_v2/seqnet_final/           定案 pid_v2fin11_c 用的 136 個網路檢查點（10.4 GB）
  work/pid_v2/seqnet_fallback_only/   只有備案 pid_v2fin7_a 用到的 64 個（4.4 GB）
  work/pid_v2/SEQNET_LINKS.txt        家族成員名稱 -> 檢查點檔名的對照
  work/pid_v2/ply_official/           新增題庫棋局的逐手向量（題池微調的輸入）
  work/pid_v2/sup_pools/、supj/pools/ 題庫棋局的有監督特徵
  MANIFEST_work.sha256                每個檔的 sha256
  LEFT_OUT.txt                        沒有放進來、可以重做的項目

下載（在個人電腦上；傳檔要走 2222 埠，一次登入抓完）
  sftp -P 2222 nano4
  sftp> get -r -a /work/u2507115/AICUP2_MOVE_20261121
  （-a 可以續傳；中斷後重跑同一行即可）

核對（在下載回來的 AICUP2_MOVE_20261121 資料夾裡）
  sha256sum -c MANIFEST_work.sha256

另一半在 /home/u2507115/AICUP2_MOVE_20261121（提交檔、結果、題庫向量、筆記，約 4 GB）。
程式與筆記的最新版在 GitHub：bkttt0429/AI_CUP_2026-2，分支 nano4/pid-b3-assign。
