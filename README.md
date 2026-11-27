# AI_CUP_2026-2

AI CUP 2026 秋季賽「圍棋棋力與風格預測競賽—競賽一：從圍棋棋譜集辨識玩家」，TEAM_11820 的程式與紀錄。

- 任務：每題給同一位未知玩家的數局棋譜，自對應候選庫中辨識該玩家，輸出信心最高的前 5 名。
- `main` 即最終提交所用的程式（原分支 `nano4/pid-b3-assign`；2026-11-27 起儲存庫整併為單一分支，紀錄見 [docs/BRANCHES.md](docs/BRANCHES.md)）。
- 專案規則（AI 工具與成員都要遵守）見 [AGENTS.md](AGENTS.md)。

<!-- final-status:begin -->
| 項目 | 內容 |
| --- | --- |
| 競賽 | AI CUP 2026 秋季賽「圍棋棋力與風格預測競賽—競賽一：從圍棋棋譜集辨識玩家」 |
| 隊伍 | TEAM_11820 |
| 最終提交檔 | `pid_v2fin11_c_a2k_p3.csv`（不在儲存庫內） |
| 最終提交檔 md5 | `c7397211f0ca0af7225ad1d04566f969` |
| Public 成績 | 0.9313515 |
| Private 成績 | 0.9341401（第 1 名） |
<!-- final-status:end -->

## 1. 文件與目錄結構

### 1.1 目錄結構

```text
AI_CUP_2026-2/
├── AGENTS.md          專案規則（AI 工具與成員都要遵守）
├── src/aicup_pid/     玩家辨識套件：統計成分、融合、配對、序列網路的資料處理
├── scripts/           最終流程的程式：特徵抽取、外部資料、序列網路訓練、題池微調、預測與合併
│   └── nano4/         最終網路家族的訓練與向量輸出啟動腳本（nano4 是國網的叢集名稱）
├── results/           最終融合文件、最終提交檔的檢查報告、embed/、final/（不在儲存庫內的權重清單與資料閘門紀錄）
│   └── archive/       封存的實驗結果
├── tests/             pytest 測試
├── tools/             規則檢查：run_checks.sh、check_code_rules.py、doc_rules.py
├── docs/              現行文件（任務、結案數字、分支紀錄、成績來源 final_status.json、新舊路徑對照）
│   ├── records/       封存紀錄：方法紀錄、計畫、備忘
│   └── work/          工作檔（YYYY-MM-DD-主題.md）
└── experiments/       封存的實驗程式與啟動腳本（以 tag pre-restructure-20261127 的目錄結構執行）
```

執行期資料夾不進版控：`data/`、`upload/`。2026-11-27 整理前的結構保存在 tag `pre-restructure-20261127`，每個檔案的新舊路徑見 [docs/layout_migration.tsv](docs/layout_migration.tsv)。

### 1.2 文件

| 文件 | 內容 |
| --- | --- |
| [docs/TASK.md](docs/TASK.md) | 任務、資料與評分規則 |
| [docs/PID_FINAL_NUMBERS.md](docs/PID_FINAL_NUMBERS.md) | 結案數字總表：最終流程、136 個網路的組成、各步驟的增益、未採用的項目、重現指令 |
| [docs/records/METHOD_NOTES.md](docs/records/METHOD_NOTES.md) | 方法紀錄：每一項實驗於取得結果前登記的規則與其結果（節號 2.x） |
| [docs/records/ARCHITECTURE.md](docs/records/ARCHITECTURE.md) | 問題分析與驗證協議（開案時） |
| [docs/records/NANO4.md](docs/records/NANO4.md) | 叢集上的操作步驟與早期流程的完整指令 |

## 2. 環境

本儲存庫不帶自己的環境，與競賽二的儲存庫 [AI_CUP_2026-1](https://github.com/bkttt0429/AI_CUP_2026-1) 共用：SGF 解析、KataGo 特徵抽取、序列網路的定義（`scripts/m_series/m88_seqid.py`）都在該儲存庫。兩個儲存庫須並排放置。

```bash
git clone https://github.com/bkttt0429/AI_CUP_2026-1
git clone https://github.com/bkttt0429/AI_CUP_2026-2
cd AI_CUP_2026-1
# setup.sh 的兩個前置檔案：KataGo 的 Python 套件（d91ea855，只取 python/）與 Human SL 網路（KataGo v1.15.0 釋出）
git clone --filter=blob:none --no-checkout https://github.com/lightvector/KataGo.git repo/KataGo
git -C repo/KataGo sparse-checkout set python
git -C repo/KataGo checkout d91ea855110dae533f0aada947b2b7d78cc8a4e1
mkdir -p models/katago && curl -fL -o models/katago/b18c384nbt-humanv0.ckpt https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.ckpt
echo "84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27  models/katago/b18c384nbt-humanv0.ckpt" | sha256sum -c
bash scripts/nano4/setup.sh          # 建立 .venv：Python 3.12、PyTorch 2.14.1（CUDA 12.6）、requirements.txt 與 requirements-dev.txt
.venv/bin/python -I repo/KataGo/python/migrate_defaultdict_to_dict.py -checkpoint models/katago/b18c384nbt-humanv0.ckpt   # 比賽時也做了這一步（說明見 AI_CUP_2026-1 的 README 第 2 節）
```

- 作業系統：Red Hat Enterprise Linux 9.6；排程：Slurm；GPU：NVIDIA H200（每節點 8 張）。
- 預訓練模型：KataGo Human SL 網路 `b18c384nbt-humanv0`，置於 `AI_CUP_2026-1/models/katago/`（取自 KataGo 官方釋出，不隨儲存庫散布）。
- 所有程式由 AI_CUP_2026-1 的根目錄執行，使用其 `.venv`；`aicup_pid.paths` 會自動尋找並排的 AI_CUP_2026-1，亦可用環境變數 `AICUP1_ROOT`、`AICUP_TRAIN_DIR` 指定。
- 計算一律以 `sbatch scripts/nano4/run.sbatch <指令>` 送至計算節點；8 張 GPU 的分片以 `bash scripts/nano4/shards.sh 8 <指令>` 執行。
- `scripts/nano4/` 內的啟動腳本，以及 `scripts/` 部分程式的預設路徑（例如 `pid_v2_seq_finetune.py`、`pid_seq_train_w*.py` 的 `--out`、`ext_fox_ingest.py` 的 `--aicup1`），含本隊叢集上的絕對路徑（`/home/u2507115/BK_AICUP_2026-2`、`~/BK_AICUP_2026-1`、`/work/u2507115/...`；`BK_AICUP_2026-1`、`BK_AICUP_2026-2` 是兩個儲存庫當時的資料夾名稱），於其他環境執行時須改為自己的路徑。

## 3. 資料

| 資料 | 位置 | 說明 |
| --- | --- | --- |
| 官方訓練集 v1.1 | `AI_CUP_2026-1/data/training/` | 100 萬局、8,007 位玩家；不隨儲存庫散布 |
| 官方測試與候選檔（2026-11-08 v2） | `data/tests_1111/` | 原題池四檔與新增池四檔；不隨儲存庫散布 |
| 外部棋譜 | 叢集上的獨立資料夾 | Fox Go Dataset（<https://github.com/featurecat/go-dataset>，GPL-3.0）；處理方式見第 4 節 |

`data/`、`upload/`、`*.npz`、`*.pt` 與 `results/**/*.csv` 均不進版控（見 `.gitignore`）。

## 4. 流程與各模組的輸入、輸出

提交檔共 1,600 列，分兩段產生：原題池 800 列由 `scripts/predict_pid.py`（統計成分＋每局嵌入＋分區塊配對）產生；新增池 800 列由 136 個序列網路產生，再由 `scripts/predict_pid_v2.py` 合併成一個檔。

### 4.1 資料處理

| 步驟 | 程式 | 輸入 | 輸出 |
| --- | --- | --- | --- |
| 訓練棋局的逐手特徵 | `AI_CUP_2026-1/scripts/m_series/m90_ply_extract.py` | 訓練集 CSV、Human SL 權重 | 逐手分片：每局前 256 手的主幹向量（全盤 64＋落點 64 維）、落點、執色 |
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
| 序列網路 | `scripts/pid_seq_train_w.py`、`pid_seq_train_w2.py`、`pid_seq_train_w3.py`（加入棋形視窗的三個訓練程式；最初的 `pid_seq_train.py` 與其他變體已封存於 `experiments/`） | 訓練與外部棋局的逐手分片、玩家分折（`AI_CUP_2026-1/splits/player_folds.csv`） | `ckpt_<網路>_<標記>e.pt`（權重的指數移動平均）、`train_<網路>_<標記>.json`（訓練紀錄） |
| 各家族的啟動腳本 | `scripts/nano4/pid_exp_s3bw_train.sh`（o32F–o35F）、`pid_exp_f0_train.sh`（o44F–o47F）、`pid_exp_len_train.sh`（o53F、o56F）、`pid_exp_len2_train.sh`（o57F、o58F）、`pid_exp_capfam_train.sh`（o49F）、`pid_exp_ns_train.sh`（o60F–o63F） | 家族名稱與成員編號 | 17 個家族、每個家族 8 個網路；腳本開頭記有該家族的配方 |
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
| 官方候選人切出的擬題池 | `scripts/pid_v2_offsim_par.py`（擬合最終融合權重 w11c 時使用；逐題的單機版 `pid_v2_offsim.py` 已封存於 `experiments/`） | 每位候選人保留 3 局為題目、7 局為參考；輸出各題數的分數與成對比較的信賴區間 |
| 訓練玩家的模擬題池 | `experiments/pid_v2_sim*.py`（封存；`scripts/pid_v2_seq_finetune.py --mode sim` 匯入的 `pid_v2_sim2.py` 與精確視窗 `pid_winx.py` 也在 `experiments/`，要用時兩個儲存庫都切到 tag `pre-restructure-20261127`） | 與新增池同形的模擬題池；早期用於擬合融合權重 |
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

**由基礎檢查點重建**：以 `scripts/pid_v2_seq_finetune.py` 對兩個新增題庫各微調 50 步後輸出向量，再執行上述指令。微調後的權重未保存；重新微調時向量的單一分量會有 0.002 至 0.007 的差異，重建結果可能有少數名次不同（METHOD_NOTES 2.244、2.257）。微調程式的 `SUP_SETS` 常數寫死了叢集上監督特徵集的位置（`/work/u2507115/pid_v2/...`，沒有對應的命令列參數，要直接修改）；其中 `probep/`（統計與投影）與 `supj/` 的訓練局特徵沒有放進本地備份，需依程式重新產生（見 `results/final/move_20261121/LEFT_OUT_work.txt`）。

**從頭訓練**：依第 4.1、4.2、4.3 節的順序執行；各家族的配方、種子與步數記於對應的啟動腳本。

## 6. 不在儲存庫內的檔案

- 136 個基礎檢查點、各題庫的輸出向量、每局嵌入的權重與提交檔：檔案過大或屬競賽資料，保存於本隊的本地備份，可依主辦單位要求提供；清單與 sha256 見 `results/final/move_20261121/MANIFEST_home.sha256`、`MANIFEST_work.sha256`。
- `data/day_scripts/` 下的輔助腳本（例如由家族清單產生融合文件的 `build_v2h.py`）：不在儲存庫，也不在本地備份的清單內（見 [docs/PID_FINAL_NUMBERS.md](docs/PID_FINAL_NUMBERS.md) 第 10 節）。重建最終提交檔不需要這些腳本（最終的融合文件已在 `results/`）；要組出新的網路組成時才需要。

## 7. 測試

```bash
cd AI_CUP_2026-1
python -I -m pytest -q ../AI_CUP_2026-2/tests
```

規則檢查（中文註解、重複函式、封存區、文件規則、ruff、pytest；見 [AGENTS.md](AGENTS.md) 第 7 節）：在本儲存庫根目錄執行 `bash tools/run_checks.sh --all`，或從 AI_CUP_2026-1 的根目錄執行 `bash ../AI_CUP_2026-2/tools/run_checks.sh --all`。兩個儲存庫各有一份 `tools/run_checks.sh`，各自只檢查所在的儲存庫。

## 8. 外部資源

- KataGo v1.15.0 與其 Human SL 網路 `b18c384nbt-humanv0.ckpt`（<https://github.com/lightvector/KataGo/releases/tag/v1.15.0>；網路權重依 KataGo Neural Network License，<https://katagotraining.org/network_license/>）：凍結的逐手特徵抽取。
- Fox Go Dataset（<https://github.com/featurecat/go-dataset>，提交 acace50，GPL-3.0）：訓練時額外的身分類別；帳號名稱僅以加鹽雜湊使用，未與競賽資料比對身分。
- 官方資料集與教學：<https://github.com/AILAB-NDHU/aicup2026-go-dataset>、<https://github.com/AILAB-NDHU/AICup-2026-Tutorial>。
- 生成式 AI：程式撰寫、實驗執行與紀錄整理使用 Anthropic Claude（Claude Code），部分實驗使用 OpenAI Codex；採用與上傳的決定由隊員做成。

競賽資料僅供本競賽使用，不得用於識別真實使用者。
