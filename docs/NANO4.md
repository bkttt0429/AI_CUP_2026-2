# nano4 操作（玩家辨識）

和 AI_CUP_2026-1 的 `docs/NANO4.md` 同一套規矩：所有計算走 Slurm、`python -I`、結果檔進 git、程式走分支 + PR。本倉庫**不帶自己的環境**：用 AI_CUP_2026-1 的 `.venv`、`scripts/nano4/run.sbatch` 與資料；兩個倉庫並排放（nano4 上是 `~/BK_AICUP_2026-1`、`~/BK_AICUP_2026-2`；`aicup_pid.paths` 也認 `AICUP_2026-*`），`aicup_pid.paths` 會自動找到旁邊的 AI_CUP_2026-1（也可以用環境變數 `AICUP1_ROOT`、`AICUP_TRAIN_DIR` 指定）。

## 0. 取得程式

```bash
cd ~ && git clone https://github.com/bkttt0429/AI_CUP_2026-2 BK_AICUP_2026-2   # 之後 git -C ~/BK_AICUP_2026-2 pull
```

## 1. 測試（CPU，一分鐘內；從 AI_CUP_2026-1 的根目錄送件，`python` 是那邊 .venv 的；nano4 一張 GPU 的工作最多 `-c 12`；`--exclude` 的節點：001／003／004／007／062 CUDA 看不到 GPU，009／039 慢 12 倍，006／008 I/O 慢——2026-11-03 B2 的工作在 008 讀 55,545 局卡了 15 分鐘，別處 23 秒）

```bash
cd ~/BK_AICUP_2026-1
sbatch --exclude=25a-hgpn[001,003,004,006,007,008,009,039,062] -t 00:30:00 -J pidtest scripts/nano4/run.sbatch python -I -m pytest -q ../BK_AICUP_2026-2/tests
```

## 2. 模擬候選庫的 CV（CPU；`scripts/pid_cv.py`；2026-10-29，issue #1）

兩個模擬候選庫：folds 1+2、folds 3+4（每庫最多 400 人、每人 100 局候選 + 其餘切 5／10／15／20 局的題），fold 0 不碰。兩個成分（開局模型 `opening`、風格統計 `style`）各自與融合的 Top-5；融合權重用另一個庫擬合。預先指定 `fusion vs opening`（玩家 bootstrap、兩庫方向一致）；只報告 `style vs opening`、`fusion vs style`。輸出 `results/pid_cv.json` 與給 `predict_pid.py` 用的 `results/pid_fusion.json`（在兩庫全部題目上擬合的權重）。

```bash
cd ~/BK_AICUP_2026-1
sbatch --exclude=25a-hgpn[001,003,004,006,007,008,009,039,062] -c 12 -t 02:00:00 -J pidcv scripts/nano4/run.sbatch python -I ../BK_AICUP_2026-2/scripts/pid_cv.py --pools 1,2 3,4 --workers 12 --out ../BK_AICUP_2026-2/results/pid_cv.json --fusion-out ../BK_AICUP_2026-2/results/pid_fusion.json
```

要貼回 issue #1 的：每庫的 `eligible`／`players`／`questions_by_n`／`players_by_rank`（抽到的候選人幾乎都是重度玩家，看看等級分布），三列（`opening`、`style`、`fusion`）的 score／top1／top5／mrr 與 by n，三條比較，融合權重，秒數。`pid_cv.json`、`pid_fusion.json` 進 git。預期：讀 100 萬局的 player／color 欄約 1–2 分鐘，每庫約 6 萬局的解析與風格特徵用 24 核約 5–10 分鐘，開局模型評分每庫約 1–2 分鐘。

讀法：`opening` 單獨 ≥ baseline 0.28 代表 H1 成立；`fusion vs opening` 通過代表風格統計有獨立增益（H2 的第一部分）。都達到就用 `predict_pid.py` 準備 11/04 的第一版提交；沒達到就看 by n 與 top5 判斷是模型還是證據的問題，記進 METHOD_NOTES。

**結果（nano4，2026-10-31）**：pytest 11 passed（job 656155）；CV job 656156，24 核 2 分 36 秒；fusion 0.9483、opening 0.9408、style 0.6818，預先指定 `fusion vs opening` +0.0075 [+0.0049, +0.0104] 過（METHOD_NOTES §2.2）。`pid_cv.json`、`pid_fusion.json` 在 `main`。

**洩漏稽核**（`scripts/pid_audit.py`，只報告；nano4 job 656405，`results/pid_audit.json`）：同樣的候選庫，查重複棋譜（含 8 重對稱）、去重後的分數、開局模型按讀到的手數、候選局在玩家間打亂的負對照、前 12 手完全重複率。`--n-players`／`--cand-games`／`--folds-csv` 與 `pid_cv.py` 同義（測試用）。

```bash
cd ~/BK_AICUP_2026-1
sbatch --exclude=25a-hgpn[001,003,004,006,007,008,009,039,062] -c 12 -t 02:00:00 -J pidaudit scripts/nano4/run.sbatch python -I ../BK_AICUP_2026-2/scripts/pid_audit.py --pools 1,2 3,4 --workers 12 --out ../BK_AICUP_2026-2/results/pid_audit.json
```

## 3. 11/04：官方檔案 → 提交（`scripts/predict_pid.py`）

```bash
cd ~/BK_AICUP_2026-1
sbatch --exclude=25a-hgpn[001,003,004,006,007,008,009,039,062] -c 12 -t 02:00:00 -J pidsub scripts/nano4/run.sbatch python -I ../BK_AICUP_2026-2/scripts/predict_pid.py \
    --candidates ../BK_AICUP_2026-2/data/tests/player_identification_candidates_public.csv ../BK_AICUP_2026-2/data/tests/player_identification_candidates_private.csv \
    --questions ../BK_AICUP_2026-2/data/tests/player_identification_test_public.csv ../BK_AICUP_2026-2/data/tests/player_identification_test_private.csv \
    --fusion ../BK_AICUP_2026-2/results/pid_fusion.json --out ../BK_AICUP_2026-2/results/submission/submission_player.csv --expect-rows 800 --workers 12
```

exit 0 且 `submission_player.report.json` 的 `problems` 為空才上傳；`dropped_games`、`fallback_questions` 要是空的（不是就先貼出來）。在這之前先做 `docs/ARCHITECTURE.md` §4 的檔案檢查（候選庫對訓練集逐局比對等）。

## 4. 11/04 流程：檔案檢查 → 第一版提交 → 對帳（`scripts/pid_check_files.py`；分岔計畫 §3）

### 4.1 11/04 前的演練（mock 官方檔，2026-11-03）

模擬候選庫寫成官方格式（每人一題、各題長盡量 100 題；folds 1+2 → `pub_`、3+4 → `priv_`），再把檢查腳本整套跑過一次。mock 的候選局全是訓練局，所以第 1 項（訓練集重複）在這裡是正控制（應 100% 命中、每位候選人恰對應一個訓練 `player_id`）；第 7 項（自我辨識）的分數就是 11/04 判 A3(ii) 用的參考值。

```bash
cd ~/BK_AICUP_2026-1
sbatch --exclude=25a-hgpn[001,003,004,006,007,008,009,039,062] -c 12 -t 01:00:00 -J pidmock scripts/nano4/run.sbatch python -I ../BK_AICUP_2026-2/scripts/pid_mock_official.py --pools 1,2 3,4 --out-dir ../BK_AICUP_2026-2/data/mock_official --seed 0
sbatch --exclude=25a-hgpn[001,003,004,006,007,008,009,039,062] -c 12 -t 02:00:00 -J pidchk scripts/nano4/run.sbatch python -I ../BK_AICUP_2026-2/scripts/pid_check_files.py \
    --candidates ../BK_AICUP_2026-2/data/mock_official/player_identification_candidates_public.csv ../BK_AICUP_2026-2/data/mock_official/player_identification_candidates_private.csv \
    --questions ../BK_AICUP_2026-2/data/mock_official/player_identification_test_public.csv ../BK_AICUP_2026-2/data/mock_official/player_identification_test_private.csv \
    --fusion ../BK_AICUP_2026-2/results/pid_fusion.json --out ../BK_AICUP_2026-2/results/checks/mock_official_checks.json --workers 12
```

mock 的 CSV 在 `data/`（不進 git）；`results/checks/*.json` 進 git。2026-11-03 的實測：索引 61 秒、每庫 240 秒、整支 9 分鐘；數字見 METHOD_NOTES §2.4。

演練的第二半：提交路徑本身也在 mock 上跑一次，並用 `pid_reconcile.py` 對帳（檢查腳本的 `main_line.top1` 對 `submission_player.csv` 的 `top_1` 必須逐題相同）、順便用真值檔算 mock 分數：

```bash
sbatch --exclude=25a-hgpn[001,003,004,006,007,008,009,039,062] -c 12 -t 02:00:00 -J pidsubmock scripts/nano4/run.sbatch bash -c 'python -I ../BK_AICUP_2026-2/scripts/predict_pid.py --candidates ../BK_AICUP_2026-2/data/mock_official/player_identification_candidates_public.csv ../BK_AICUP_2026-2/data/mock_official/player_identification_candidates_private.csv --questions ../BK_AICUP_2026-2/data/mock_official/player_identification_test_public.csv ../BK_AICUP_2026-2/data/mock_official/player_identification_test_private.csv --fusion ../BK_AICUP_2026-2/results/pid_fusion.json --out ../BK_AICUP_2026-2/results/mock_official/submission_player.csv --expect-rows 800 --workers 12 && python -I ../BK_AICUP_2026-2/scripts/pid_reconcile.py --checks ../BK_AICUP_2026-2/results/checks/mock_official_checks.json --submission ../BK_AICUP_2026-2/results/mock_official/submission_player.csv --truth ../BK_AICUP_2026-2/data/mock_official/truth_public.csv ../BK_AICUP_2026-2/data/mock_official/truth_private.csv --out ../BK_AICUP_2026-2/results/checks/mock_official_reconcile.json'
```

### 4.2 11/04 當天

1. 官方四個檔放到 `~/BK_AICUP_2026-2/data/tests/`（檔名照官方）。
2. **先檢查、再建模**：同 §4.1 的第二道指令，把 `data/mock_official/...` 換成 `data/tests/...`、`--out` 換成 `results/checks/official_checks.json`。看 log 最後的 `flags`：
   - `A1 training overlap True` → 候選局在訓練集裡（封閉集）。只記錄，主線照常提交；要不要用等主辦單位回覆。
   - `A3(i) opening shift True`（兩庫 k=4 的觀測值都 > 1.5 × 虛無最大值）或自我辨識 fusion 分數 < mock 參考值 0.936 − 0.10 = 0.836 → 開局訊號在官方資料上變弱，支線 C 與 B 並行。
   - `one-to-one likely {pub_: True, priv_: True}` → 一人一題，支線 B3 可以上。
   - 另看：`illegal records`（`predict_pid.py` 會丟掉同樣的局，數量要一致）、`question-vs-candidate duplicates`（應 0）、`num_games`（各 100）。
3. 第一版提交：§3 的 `predict_pid.py` 指令。`exit 0`、`problems` 空、`dropped_games` 與 `fallback_questions` 空才上傳；上傳前跑 `pid_reconcile.py --checks results/checks/official_checks.json --submission results/submission/submission_player.csv --out results/checks/official_reconcile.json`（不給 `--truth`），`top_1 agree 800/800`、exit 0 才上傳。
   - **B3（CV 通過，§5.1）**：若 `flags` 的 `one-to-one likely` 兩庫都 True（第 8 項 ≥ 350）且使用者同意，第二版提交 = 同一指令加 `--assign auto`，寫到另一個 `--out`（例如 `results/submission_assign/submission_player.csv`）；report 的 `assign.applied`、`n_top1_changed` 貼回 issue。對帳時 `pid_reconcile.py` 的 Top-1 會有差異（差異題數應等於 `n_top1_changed`），這是預期的。
   - **B2（CV 通過，§5.3）**：`--fusion results/pid_fusion3.json`（opening + style + local 三成分，權重在兩庫全部題目上擬合）是另一組候選；與 `--assign auto` 可疊加。對帳時用同一份文件跑 `pid_check_files.py`（它認三成分文件；第 7 項自我辨識仍用兩成分等權）。
   - 四個候選：fusion／fusion3 × 不配對／`--assign auto`。每日 5 次上傳、最後一次生效：先交 fusion 看 Public，再依序交其他版本比較；哪個留到最後由使用者決定。
4. 結果（`official_checks.json`、`submission_player.report.json`、Public 分數）貼到 issue，寫進 METHOD_NOTES；Public 分數對自我辨識分數的落差決定支線的優先序（分岔計畫 §3.2、§3.3）。

## 5. 支線 B（分岔計畫 §4；全部 CPU、`-c 12`、fold 0 不碰；結果 JSON 進 git）

### 5.1 B3 全局配對（`scripts/pid_assign_cv.py`）

每人一題的模擬候選庫（folds 1+2、3+4 各 5 個 seed，每庫 400 人、各 n 100 題），融合權重交叉擬合（用另一個 fold 組的庫），兩臂：`fusion`（逐題 Top-5）與 `assign`（Hungarian 定 Top-1，其餘照原順序）。預先指定 `assign vs fusion`：玩家 bootstrap、兩個 fold 組同向、整體 ≥ 0.003 或 n=5 ≥ 0.005。`--control` 另加每個 fold 組一個「玩家會重複」的對照庫（只報告：一人一題不成立時配對的傷害，以及 Top-1 不同候選人數能不能分出兩種設計）。

```bash
cd ~/BK_AICUP_2026-1
sbatch --exclude=25a-hgpn[001,003,004,006,007,008,009,039,062] -c 12 -t 02:00:00 -J b3cv scripts/nano4/run.sbatch python -I ../BK_AICUP_2026-2/scripts/pid_assign_cv.py --pools 1,2 3,4 --seeds 0 1 2 3 4 --control --workers 12 --out ../BK_AICUP_2026-2/results/assign/assign_cv.json
```

提交端：`predict_pid.py --assign auto`（該庫 400 題的 Top-1 有 ≥ 350 個不同候選人才套用；`on` 強制、`off` 預設）。11/04 用不用由 CV 結果 + `pid_check_files.py` 第 8 項 + 使用者決定。

### 5.2 B1 開局模型超參（`scripts/pid_opening_sweep.py`）

pid_cv 的兩個模擬庫（seed 0），風格模型只擬合一次；base（depth 12、α 2、β 5、α_full 1）與事先指定的格子（depth 8／16／20、α 1／4、β 2／10、α_full 0.5／2／1000=關掉完整前綴）各重擬開局模型、交叉擬合融合權重，每個設定 vs base 於融合分數（同一套門檻）。

```bash
sbatch --exclude=25a-hgpn[001,003,004,006,007,008,009,039,062] -c 12 -t 03:00:00 -J b1sweep scripts/nano4/run.sbatch python -I ../BK_AICUP_2026-2/scripts/pid_opening_sweep.py --pools 1,2 3,4 --workers 12 --out ../BK_AICUP_2026-2/results/opening/opening_sweep.json
```

通過的設定要接進提交，得重跑 `pid_cv.py --fusion-out` 產生新的 `pid_fusion.json`（`cfg` 跟著進文件），並開「[提交]」issue。

### 5.3 B2 整局局部樣式成分（`scripts/pid_local_cv.py`）

pid_cv 的兩個模擬庫（seed 0）；開局＋風格照舊，加 `aicup_pid.local`（黑棋第 `--start` 手起每手的位移類別／線數／階段 token，每人 Dirichlet 平滑多項式）當第三成分；兩成分與三成分的融合權重都交叉擬合；預先指定 `fusion3 vs fusion`。`--fusion-out` 把三成分權重在全部題目上擬合成 `results/pid_fusion3.json` 給 `predict_pid.py`。

```bash
sbatch --exclude=25a-hgpn[001,003,004,006,007,008,009,039,062] -c 12 -t 01:00:00 -J b2cv scripts/nano4/run.sbatch python -I ../BK_AICUP_2026-2/scripts/pid_local_cv.py --pools 1,2 3,4 --workers 12 --out ../BK_AICUP_2026-2/results/local/local_cv.json --fusion-out ../BK_AICUP_2026-2/results/pid_fusion3.json
```

### 5.4 穩健性線（使用者 2026-11-03 核定「其他處理的方法」的建議順序）

全部 CPU、`-c 12`、只報告或事先指定如下；佇列滿時用 `until sbatch ...; do sleep 60; done` 等槽。

```bash
cd ~/BK_AICUP_2026-1
X="--exclude=25a-hgpn[001,003,004,006,007,008,009,039,062]"
# 候選局減量曲線（100 → 50 → 20；fusion 與 fusion3）
for n in 50 20; do sbatch $X -c 12 -t 01:00:00 -J decay$n scripts/nano4/run.sbatch python -I ../BK_AICUP_2026-2/scripts/pid_local_cv.py --pools 1,2 3,4 --cand-games $n --workers 12 --out ../BK_AICUP_2026-2/results/local/local_cv_cand$n.json; done
# 難負例候選庫：隨機 400 人 vs 同級帶 vs 開局 repertoire 最相近的 400 人；固定的提交權重，不重擬
sbatch $X -c 12 -t 02:00:00 -J hardpools scripts/nano4/run.sbatch python -I ../BK_AICUP_2026-2/scripts/pid_hard_pools.py --folds 1 2 3 4 --workers 12 --out ../BK_AICUP_2026-2/results/hard/hard_pools.json
# 轉導式自我訓練（每人一題的庫；事先指定 self3 vs assign3）
sbatch $X -c 12 -t 02:00:00 -J selftrain scripts/nano4/run.sbatch python -I ../BK_AICUP_2026-2/scripts/pid_selftrain_cv.py --pools 1,2 3,4 --seeds 0 1 2 3 4 --workers 12 --out ../BK_AICUP_2026-2/results/selftrain/selftrain_cv.json
```

### 5.5 B5 讓子旗標成分（`scripts/pid_hd_cv.py`；2026-11-04 使用者「開始」）

官方 SGF 只留 `SZ`、`HD`、落子；HD 只有 0／1，且每位候選 100 局的 HD1 比例跨人標準差 0.22（二項虛無 0.037；METHOD_NOTES §2.12）。`aicup_pid.hd.HDModel` 是每人對 HD 類別（0／1／≥2）的 Dirichlet 多項式，當第四個融合成分；事先指定 `fusion4 vs fusion3`，只報告 `hd vs style`、`fusion4 vs fusion`。

```bash
cd ~/BK_AICUP_2026-1
X="--exclude=25a-hgpn[001,003,004,005,006,007,008,009,010,039,062,155]"
sbatch $X -c 12 -t 02:00:00 -J hdcv scripts/nano4/run.sbatch python -I ../BK_AICUP_2026-2/scripts/pid_hd_cv.py --pools 1,2 3,4 --workers 12 \
    --out ../BK_AICUP_2026-2/results/hd/hd_cv.json --fusion-out ../BK_AICUP_2026-2/results/pid_fusion4.json
```

通過後 `predict_pid.py --fusion results/pid_fusion4.json`（可加 `--assign auto`）是新的提交候選；`pid_check_files.py` 也認四成分文件。注意：模擬庫的訓練局含 HD ≥ 2（官方庫沒有），CV 的 HD 類別分布在 log 裡。

