# AI_CUP_2026-2

AI CUP 2026 秋季賽「圍棋棋力與風格預測競賽—競賽一：從圍棋棋譜集辨識玩家」，TEAM_11820 的程式與紀錄。

- 任務：每題給同一位未知玩家的數局棋譜，自對應候選庫中辨識該玩家，輸出信心最高的前 5 名。
- 成績：Public 0.9313515、Private 0.9341401（最終提交檔 `pid_v2fin11_c_a2k_p3.csv`，md5 `c7397211f0ca0af7225ad1d04566f969`）。
- 本分支（`nano4/pid-b3-assign`）為最終提交所用的程式；`main` 只有開案時的第一版。

## 1. 文件

| 文件 | 內容 |
| --- | --- |
| [docs/TASK.md](docs/TASK.md) | 任務、資料與評分規則 |
| [docs/PID_FINAL_NUMBERS.md](docs/PID_FINAL_NUMBERS.md) | 結案數字總表：最終流程、136 個網路的組成、各步驟的增益、未採用的項目、重現指令 |
| [METHOD_NOTES.md](METHOD_NOTES.md) | 方法紀錄：每一項實驗於取得結果前登記的規則與其結果（節號 2.x） |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 問題分析與驗證協議（開案時） |
| [docs/NANO4.md](docs/NANO4.md) | 叢集上的操作步驟與早期流程的完整指令 |

## 2. 環境

本倉庫不帶自己的環境，與競賽二的倉庫 [AI_CUP_2026-1](https://github.com/bkttt0429/AI_CUP_2026-1) 共用：SGF 解析、KataGo 特徵抽取、序列網路的定義（`logs/nano4/m88_seqid.py`）都在該倉庫。兩個倉庫須並排放置。

```bash
git clone https://github.com/bkttt0429/AI_CUP_2026-1
git clone -b nano4/pid-b3-assign https://github.com/bkttt0429/AI_CUP_2026-2
cd AI_CUP_2026-1 && bash scripts/nano4/setup.sh      # 建立 .venv：Python 3.12、PyTorch 2.14.1（CUDA 12.6）與 requirements.txt
```

- 作業系統：Red Hat Enterprise Linux 9.6；排程：Slurm；GPU：NVIDIA H200（每節點 8 張）。
- 預訓練模型：KataGo Human SL 網路 `b18c384nbt-humanv0`，置於 `AI_CUP_2026-1/models/katago/`（取自 KataGo 官方釋出，不隨倉庫散布）。
- 所有程式由 AI_CUP_2026-1 的根目錄執行，使用其 `.venv`；`aicup_pid.paths` 會自動尋找並排的 AI_CUP_2026-1，亦可用環境變數 `AICUP1_ROOT`、`AICUP_TRAIN_DIR` 指定。
- 計算一律以 `sbatch scripts/nano4/run.sbatch <指令>` 送至計算節點；8 張 GPU 的分片以 `bash scripts/nano4/shards.sh 8 <指令>` 執行。
- `scripts/nano4/` 內的啟動腳本含本隊叢集上的絕對路徑（`/home/u2507115/BK_AICUP_2026-2`、`/work/u2507115/...`），於其他環境執行時須改為自己的路徑。

## 3. 資料

| 資料 | 位置 | 說明 |
| --- | --- | --- |
| 官方訓練集 v1.1 | `AI_CUP_2026-1/data/training/` | 100 萬局、8,007 位玩家；不隨倉庫散布 |
| 官方測試與候選檔（2026-11-08 v2） | `data/tests_1111/` | 原題池四檔與新增池四檔；不隨倉庫散布 |
| 外部棋譜 | 叢集上的獨立資料夾 | Fox Go Dataset（<https://github.com/featurecat/go-dataset>，GPL-3.0）；處理方式見第 4 節 |

`data/`、`upload/`、`*.npz`、`*.pt` 與 `results/**/*.csv` 均不進版控（見 `.gitignore`）。

## 4. 流程與各模組的輸入、輸出

提交檔共 1,600 列，分兩段產生：原題池 800 列由 `scripts/predict_pid.py`（統計成分＋每局嵌入＋分區塊配對）產生；新增池 800 列由 136 個序列網路產生，再由 `scripts/predict_pid_v2.py` 合併成一個檔。

### 4.1 資料處理

| 步驟 | 程式 | 輸入 | 輸出 |
| --- | --- | --- | --- |
| 訓練棋局的逐手特徵 | `AI_CUP_2026-1/logs/nano4/m90_ply_extract.py` | 訓練集 CSV、Human SL 權重 | 逐手分片：每局前 256 手的主幹向量（全盤 64＋落點 64 維）、落點、執色 |
| 新增池的逐手特徵 | `scripts/pid_v2_ply_extract.py` | `data/tests_1111/` 的候選與題目棋譜 | 每個分片一個 `.npz`：`GL`［256, 128］、`POS`、`MOV`、`LEN`、`SUBJ` |
| 每局摘要特徵 | `scripts/pid_trunk_extract.py --official2` | 同上 | 每局一列 256 維（兩個主幹向量的平均與標準差），供統計成分與每局嵌入使用 |
| 有監督投影的特徵 | `scripts/pid_sup_fit.py`、`scripts/pid_sup_ply_extract.py` | 訓練棋局、投影矩陣 | `sup_s<分片>_p<k>.npz`：逐手 68 維特徵 |
| 外部棋譜匯入 | `scripts/ext_fox_ingest.py --stage ingest` | Fox Go Dataset 的壓縮檔 | 精簡分片 `games_NNNN.npz`；帳號名稱以加鹽雜湊取代 |
| 重疊檢查 | `scripts/ext_fox_ingest.py --stage overlap` | 精簡分片、競賽訓練棋局 | JSON：前 60 手（含 8 種對稱）相同的局數（實測為 0） |
| 外部帳號與棋局的選取 | `scripts/ext_fox_select.py` | 精簡分片 | `select.csv`（競賽格式的棋譜、雜湊帳號、主體執色）與 `select.json` |
| 外部棋局的逐手特徵 | `scripts/pid_ext_ply_extract.py` | `select.csv` | 與訓練棋局相同格式的逐手分片 |

### 4.2 訓練

| 步驟 | 程式 | 輸入 | 輸出 |
| --- | --- | --- | --- |
| 序列網路 | `scripts/pid_seq_train.py` 及其變體（`_w`、`_w2`、`_w3`、`_wx`：加入棋形視窗） | 訓練與外部棋局的逐手分片、玩家分折（`AI_CUP_2026-1/splits/player_folds.csv`） | `ckpt_<網路>_<標記>e.pt`（權重的指數移動平均）、`train_<網路>_<標記>.json`（訓練紀錄） |
| 各家族的啟動腳本 | `scripts/nano4/pid_exp_s4w_train.sh`、`pid_exp_w2fam_train2.sh`、`pid_exp_f0_train.sh` 等 | 家族名稱與成員編號 | 每個家族 8 個網路；腳本開頭記有該家族的配方 |
| 每局嵌入 | `scripts/pid_embed_cv.py`（原題池）、`scripts/pid_v2_embed.py`（新增池）、`src/aicup_pid/embed.py` | 每局摘要特徵 | `results/embed/emb_final.pt`（原題池）、`results/embed/emb2_final.pt`（新增池） |

主要設定：Transformer 寬 384、8 層、8 個注意力頭（四個家族為寬 512、10 層）；AdamW、學習率 3×10⁻⁴、權重衰減 0.05、1,000 步暖身、餘弦排程；每個情節 256 位玩家各 8 局，其中一半來自外部帳號；棋盤對稱擴增、隨機遮手 0.3、依局數抽樣；3 萬至 5.5 萬步。各家族的完整參數見啟動腳本與 [docs/PID_FINAL_NUMBERS.md](docs/PID_FINAL_NUMBERS.md) 第 4、5 節。

### 4.3 推論

| 步驟 | 程式 | 輸入 | 輸出 |
| --- | --- | --- | --- |
| 題池微調與向量 | `scripts/pid_v2_seq_finetune.py --mode official --pool pub2\|priv2` | 基礎檢查點、該題庫候選人的逐手分片 | `official_<題庫>_net<v>_<標記>_t50.npz`：微調 50 步後，該題庫所有棋局的向量 |
| 統計成分 | `src/aicup_pid/`（`opening`、`orient`、`local`、`style`、`hd`、`embed`、`v2`） | 題庫的棋譜與每局摘要特徵 | 每題對每位候選人的分數矩陣 |
| 融合文件 | `results/pid_fusion_<名稱>.json` | — | 成分清單、9 個融合權重、各網路向量檔的清單與比重；最終為 `pid_fusion_v2fin11_c.json` |
| 新增池預測與合併 | `scripts/predict_pid_v2.py` | 融合文件、向量檔、原題池的 800 列 | `submission_player.csv`（1,600 列）與檢查報告 |
| 配對與先驗 | `src/aicup_pid/assign.py`（`assign_soft`、`sinkhorn_prices`） | 融合後的分數矩陣 | 多局題的軟性一對一配對、1 局題先驗後的前 5 名 |
| 原題池預測 | `scripts/predict_pid.py --assign-blocks` | 原題池四檔、`pid_fusion3o_ref_trunk_embt.json` | 原題池的 800 列 |

### 4.4 離線評估

| 步驟 | 程式 | 說明 |
| --- | --- | --- |
| 官方候選人切出的擬題池 | `scripts/pid_v2_offsim.py`、`pid_v2_offsim_par.py` | 每位候選人保留 3 局為題目、7 局為參考；輸出各題數的分數與成對比較的信賴區間 |
| 訓練玩家的模擬題池 | `scripts/pid_v2_sim*.py` | 與新增池同形的模擬題池；用於擬合融合權重 |
| 檔案差異 | `scripts/pid_diff_files.py` | 比較兩個提交檔逐題的差異 |

## 5. 重現

**由已保存的向量重建最終提交檔**（不需 GPU；結果與上傳檔逐位元相同）：

```bash
cd AI_CUP_2026-1
python -I ../AI_CUP_2026-2/scripts/predict_pid_v2.py \
  --tests-dir ../AI_CUP_2026-2/data/tests_1111 \
  --fusion ../AI_CUP_2026-2/results/pid_fusion_v2fin11_c.json \
  --trunk-store ../AI_CUP_2026-2/data/trunk/tests_1111_v2 \
  --old-submission ../AI_CUP_2026-2/results/submission_old_embt_assign/submission_player.csv \
  --assign big --big-lam 2 --big-tau 0.5 --one-prior3 0.05 --prior-T 1.6 \
  --out results/submission_pid_v2fin11_c_a2k_p3/submission_player.csv
```

需要的檔案：融合文件所列的向量檔（`data/seq_ft/official_*_t50.npz`）、每局嵌入 `results/embed/emb2_final.pt`、每局摘要特徵 `data/trunk/tests_1111_v2`、原題池的 800 列。融合文件內的路徑為本隊叢集上的絕對路徑，須先改為實際位置。

**由基礎檢查點重建**：以 `scripts/pid_v2_seq_finetune.py` 對兩個新增題庫各微調 50 步後輸出向量，再執行上述指令。微調後的權重未保存；重新微調時向量的單一分量會有 0.002 至 0.007 的差異，重建結果可能有少數名次不同（METHOD_NOTES 2.244、2.257）。

**從頭訓練**：依第 4.1、4.2、4.3 節的順序執行；各家族的配方、種子與步數記於對應的啟動腳本。

## 6. 不在倉庫內的檔案

- 136 個基礎檢查點、各題庫的輸出向量、每局嵌入的權重與提交檔：檔案過大或屬競賽資料，保存於本隊的本地備份，可依主辦單位要求提供；清單與 sha256 見 `results/final/move_20261121/MANIFEST_home.sha256`、`MANIFEST_work.sha256`。
- `data/day_scripts/` 下的輔助腳本（例如由家族清單產生融合文件的 `build_v2h.py`）：位於不進版控的 `data/` 內，目前不在倉庫。重建最終提交檔不需要這些腳本（最終的融合文件已在 `results/`）；要組出新的網路組成時才需要。

## 7. 測試

```bash
cd AI_CUP_2026-1
python -I -m pytest -q ../AI_CUP_2026-2/tests
```

## 8. 外部資源

- KataGo v1.15.0 與其 Human SL 網路 `b18c384nbt-humanv0.ckpt`（<https://github.com/lightvector/KataGo/releases/tag/v1.15.0>；網路權重依 KataGo Neural Network License，<https://katagotraining.org/network_license/>）：凍結的逐手特徵抽取。
- Fox Go Dataset（<https://github.com/featurecat/go-dataset>，提交 acace50，GPL-3.0）：訓練時額外的身分類別；帳號名稱僅以加鹽雜湊使用，未與競賽資料比對身分。
- 官方資料集與教學：<https://github.com/AILAB-NDHU/aicup2026-go-dataset>、<https://github.com/AILAB-NDHU/AICup-2026-Tutorial>。
- 生成式 AI：程式撰寫、實驗執行與紀錄整理使用 Anthropic Claude（Claude Code），部分實驗使用 OpenAI Codex；採用與上傳的決定由隊員做成。

競賽資料僅供本競賽使用，不得用於識別真實使用者。
