# nano4 操作（玩家辨識）

和 AI_CUP_2026-1 的 `docs/NANO4.md` 同一套規矩：所有計算走 Slurm、`python -I`、結果檔進 git、程式走分支 + PR。本倉庫**不帶自己的環境**：用 AI_CUP_2026-1 的 `.venv`、`scripts/nano4/run.sbatch` 與資料；兩個倉庫並排放（`~/AICUP_2026-1`、`~/AICUP_2026-2`），`aicup_pid.paths` 會自動找到旁邊的 AI_CUP_2026-1（也可以用環境變數 `AICUP1_ROOT`、`AICUP_TRAIN_DIR` 指定）。

## 0. 取得程式

```bash
cd ~ && git clone https://github.com/bkttt0429/AI_CUP_2026-2 AICUP_2026-2   # 之後 git -C ~/AICUP_2026-2 pull
```

## 1. 測試（CPU，一分鐘內；從 AI_CUP_2026-1 的根目錄送件，`python` 是那邊 .venv 的）

```bash
cd ~/AICUP_2026-1
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -t 00:30:00 -J pidtest scripts/nano4/run.sbatch python -I -m pytest -q ../AICUP_2026-2/tests
```

## 2. 模擬候選庫的 CV（CPU；`scripts/pid_cv.py`；2026-10-29，issue #1）

兩個模擬候選庫：folds 1+2、folds 3+4（每庫最多 400 人、每人 100 局候選 + 其餘切 5／10／15／20 局的題），fold 0 不碰。兩個成分（開局模型 `opening`、風格統計 `style`）各自與融合的 Top-5；融合權重用另一個庫擬合。預先指定 `fusion vs opening`（玩家 bootstrap、兩庫方向一致）；只報告 `style vs opening`、`fusion vs style`。輸出 `results/pid_cv.json` 與給 `predict_pid.py` 用的 `results/pid_fusion.json`（在兩庫全部題目上擬合的權重）。

```bash
cd ~/AICUP_2026-1
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 24 -t 02:00:00 -J pidcv scripts/nano4/run.sbatch python -I ../AICUP_2026-2/scripts/pid_cv.py --pools 1,2 3,4 --workers 24 --out ../AICUP_2026-2/results/pid_cv.json --fusion-out ../AICUP_2026-2/results/pid_fusion.json
```

要貼回 issue #1 的：每庫的 `eligible`／`players`／`questions_by_n`／`players_by_rank`（抽到的候選人幾乎都是重度玩家，看看等級分布），三列（`opening`、`style`、`fusion`）的 score／top1／top5／mrr 與 by n，三條比較，融合權重，秒數。`pid_cv.json`、`pid_fusion.json` 進 git。預期：讀 100 萬局的 player／color 欄約 1–2 分鐘，每庫約 6 萬局的解析與風格特徵用 24 核約 5–10 分鐘，開局模型評分每庫約 1–2 分鐘。

讀法：`opening` 單獨 ≥ baseline 0.28 代表 H1 成立；`fusion vs opening` 通過代表風格統計有獨立增益（H2 的第一部分）。都達到就用 `predict_pid.py` 準備 11/04 的第一版提交；沒達到就看 by n 與 top5 判斷是模型還是證據的問題，記進 METHOD_NOTES。

**結果（nano4，2026-10-31）**：pytest 11 passed（job 656155）；CV job 656156，24 核 2 分 36 秒；fusion 0.9483、opening 0.9408、style 0.6818，預先指定 `fusion vs opening` +0.0075 [+0.0049, +0.0104] 過（METHOD_NOTES §2.2）。`pid_cv.json`、`pid_fusion.json` 在 `main`。

**洩漏稽核**（`scripts/pid_audit.py`，只報告；nano4 job 656405，`results/pid_audit.json`）：同樣的候選庫，查重複棋譜（含 8 重對稱）、去重後的分數、開局模型按讀到的手數、候選局在玩家間打亂的負對照、前 12 手完全重複率。`--n-players`／`--cand-games`／`--folds-csv` 與 `pid_cv.py` 同義（測試用）。

```bash
cd ~/AICUP_2026-1
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 24 -t 02:00:00 -J pidaudit scripts/nano4/run.sbatch python -I ../AICUP_2026-2/scripts/pid_audit.py --pools 1,2 3,4 --workers 24 --out ../AICUP_2026-2/results/pid_audit.json
```

## 3. 11/04：官方檔案 → 提交（`scripts/predict_pid.py`）

```bash
cd ~/AICUP_2026-1
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 24 -t 02:00:00 -J pidsub scripts/nano4/run.sbatch python -I ../AICUP_2026-2/scripts/predict_pid.py \
    --candidates ../AICUP_2026-2/data/tests/player_identification_candidates_public.csv ../AICUP_2026-2/data/tests/player_identification_candidates_private.csv \
    --questions ../AICUP_2026-2/data/tests/player_identification_test_public.csv ../AICUP_2026-2/data/tests/player_identification_test_private.csv \
    --fusion ../AICUP_2026-2/results/pid_fusion.json --out ../AICUP_2026-2/results/submission/submission_player.csv --expect-rows 800 --workers 24
```

exit 0 且 `submission_player.csv.report.json` 的 `problems` 為空才上傳；`dropped_games`、`fallback_questions` 要是空的（不是就先貼出來）。在這之前先做 `docs/ARCHITECTURE.md` §4 的檔案檢查（候選庫對訓練集逐局比對等）。

## 4. 供競賽二用：測試玩家是否在訓練集裡（`scripts/pid_rank_overlap.py`；AI_CUP_2026-1 issue #22）

CPU 一支工作。讀 100 萬局只取前 12 手（24 核約 10 分鐘，RAM 數 GB），擬合 8,007 人 × 黑／白兩個開局模型，fold 1 校準 τ、fold 0 評一次（與 `fold0_fused_opp_proba.csv` 同一組 9,872 題），再做同一人多題合併的檢查：

```bash
cd ~/AICUP_2026-1
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 24 -t 02:00:00 -J rankovl scripts/nano4/run.sbatch python -I ../AICUP_2026-2/scripts/pid_rank_overlap.py --scores results/scores --calib-fold 1 --eval-fold 0 --fused-proba results/l1b/fold0_fused_opp_proba.csv --siblings --workers 24 --out ../AICUP_2026-2/results/rank_overlap.json
```

貼回 AI_CUP_2026-1 issue #22：校準折的 Top-1／Top-5 按 n 與 τ；評估折的 Top-1／Top-5 按 n、`id_rank_score`、margin ≥ τ 的比例與精確率、認對／認錯的 margin 分位數；`combined` vs `fused_opp`（差、CI）；`siblings` 每個 n 的 `top1_same_player`。`rank_overlap.json` 進 git（本倉庫）。RAM 不夠就把 `--workers` 降到 12。

11/04 拿到官方檔案後（τ 取 json 的 `calibration.tau`）：

```bash
sbatch --exclude=25a-hgpn[001,003,004,007,039,062] -c 24 -t 02:00:00 -J rankovlT scripts/nano4/run.sbatch python -I ../AICUP_2026-2/scripts/pid_rank_overlap.py --test-csv data/tests/rank_prediction_test_public.csv data/tests/rank_prediction_test_private.csv --tau <tau> --workers 24 --out ../AICUP_2026-2/results/rank_overlap_test.json --out-table ../AICUP_2026-2/results/rank_overlap_test.csv
```

貼回 `share_above_tau`（按 n）與 margin 分位數；和 fold 0 認對／認錯的分布比。是否採用由使用者決定。

