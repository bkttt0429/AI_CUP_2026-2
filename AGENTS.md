# 專案規則

所有 AI 工具（Claude Code、Codex、Cursor 等）和成員都遵守這份規則。

本儲存庫是 AI CUP 2026 秋季賽「圍棋棋力與風格預測競賽—競賽一：從圍棋棋譜集辨識玩家」TEAM_11820 的程式與紀錄。比賽已結束，成績與最終提交檔見 `README.md` 的成績表（來源為 `docs/final_status.json`）。本儲存庫沒有自己的環境，與競賽二的 [AI_CUP_2026-1](https://github.com/bkttt0429/AI_CUP_2026-1) 共用程式與環境，兩者要並排放置（見第 10 節）。
第 1 到 4 節的行為準則改寫自 [andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills)。

## 1. 動手前先想清楚

- 明確說出假設；不確定就問，不要默默選一種解讀。
- 有更簡單的做法就直接提出。

## 2. 簡單優先

- 不加沒被要求的功能、設定項或彈性。
- 單次使用的邏輯不要抽象；同樣邏輯出現第 3 次才抽到共用模組。

## 3. 只改必要的地方

- 不順手修改無關的程式碼、註解或排版。
- 自己的修改造成沒人使用的程式，要一併刪除。
- 比賽已結束：會改變最終提交檔、融合文件、權重清單或報告數字的修改，一律先問使用者。
- `scripts/` 與 `src/` 是重建最終提交檔的程式，維持比賽時的行為；`experiments/`、`docs/records/`、`results/archive/` 是封存，不再修改。
- 檔名是引用鍵：方法紀錄（節號 2.x）與 `docs/PID_FINAL_NUMBERS.md` 以檔名引用程式、啟動腳本與融合文件，不要改名。

## 4. 以可驗證的目標推進

- 先定義怎麼算完成，再反覆驗證到通過。
- 重構前後都要跑測試；會影響輸出的程式，要比對輸出是否一致。
- 動到最終流程時，`README.md` 第 5 節由保存的向量重建的提交檔，必須與上傳檔逐位元相同。
- 只加註解或只改路徑的修改，要以語法樹比對證明行為不變。

## 5. 不重複、不冗餘

- 新增函式前先搜尋 `src/aicup_pid/`、競賽二的 `src/aicup/` 和既有程式，能重用就重用。
- 兩個競賽共用的邏輯放在競賽二的 `src/aicup/`；玩家辨識專用的放 `src/aicup_pid/`；`scripts/` 只放流程的進入點，`scripts/nano4/` 只放 Slurm 啟動腳本。不要從 `experiments/` 複製程式；要再用時搬回 `scripts/` 或 `src/`，並補上測試與註解。
- 不留死碼、未使用的 import 或變數。
- 最終流程既有的重複（例如三份序列網路訓練程式 `pid_seq_train_w*.py`）維持比賽時的程式，記錄在 `tools/duplicates_baseline.txt`；新的程式不得再新增重複。

## 6. 每行程式碼都要有中文註解（強制）

適用 `src/`、`scripts/`、`tools/` 底下的 `.py`、`.sh` 和 `.sbatch`；`tests/` 和各封存區不適用：

- 每一行程式碼（含跨行運算式的每一行）都要在**同一行行尾**加上繁體中文註解。
- 整行註解也必須是繁體中文。
- 新寫的 docstring 用繁體中文。比賽時既有的 docstring 維持原文（多支程式以 `__doc__` 產生 `--help`），列在 `tools/docstrings_baseline.txt`。
- docstring 的每一行、多行字串的內部行、heredoc 的內容、空行、shebang 和 `#SBATCH` 指令行不需要註解。
- 不用反斜線 `\` 續行，因為它後面不能接註解；Python 改用括號，shell 把指令併成一行或改用陣列。
- 註解要說明這一行的目的或原因，不要只是把程式翻譯一遍。

## 7. 強制檢查

所有 AI 工具和成員，在回報完成或 commit 前都要執行：

```bash
bash tools/run_checks.sh --all
```

它會檢查中文註解、重複函式、封存區是否被修改和文件規則，並執行 `ruff` 與 `pytest`（測試從並排的 AI_CUP_2026-1 根目錄執行）。沒有通過就不算完成，也不要為了通過檢查而停用規則。結束碼 2 表示有檢查沒有執行（例如缺少環境，或在國網登入節點上），同樣不算完成。

| 使用情境 | 自動檢查 |
| --- | --- |
| Claude Code、Codex、Cursor 等工具，或手動修改 | 沒有 hook，要自己執行上面的指令 |
| 國網 nano4 | 登入節點不可執行測試，要從 AI_CUP_2026-1 的根目錄送到計算節點：`sbatch -J checks -t 00:30:00 scripts/nano4/run.sbatch bash ../AI_CUP_2026-2/tools/run_checks.sh --all` |
| 所有 push 和 pull request | GitHub Actions（`.github/workflows/checks.yml`） |

`ruff` 對 `tools/` 套用基本規則（E4、E7、E9、F，含未使用的 import 與變數）；`src/`、`scripts/`、`tests/` 只檢查語法錯誤與未定義的名稱，因為最終流程的程式維持比賽時的寫法。工具擋不住「寫法不同但功能相同」的重複，這部分仍要靠第 5 節的規則。

## 8. 文件規則（強制）

不同工具和成員會同時工作，這些規則是為了避免文件互相蓋掉或變亂。

1. **每個事實只有一個來源。** 最終提交檔、md5 與成績一律來自 `docs/final_status.json`；`README.md` 的成績表由 `python -I tools/doc_rules.py --update` 依它產生，不要手動改，其他地方要提這些數字時改用連結。最終網路組成與融合權重以 `results/pid_fusion_v2fin11_c.json` 為準。
2. **計畫和進度寫在工作檔。** 每項工作一個 `docs/work/YYYY-MM-DD-主題.md`，開頭要有下面三個欄位。只改自己負責的工作檔，不要改別人的。

   ```markdown
   - **負責人：** 名字
   - **狀態：** 進行中／等待中／已完成／已取消
   - **最後更新：** YYYY-MM-DD HH:MM
   ```
3. **工作檔要即時更新。** 每完成一個階段、工作中斷或狀況改變，就立刻更新並推上 GitHub，讓隊友隨時可以接手。為了即時，**只有 `docs/work/` 的工作檔**可以直接 push 到 `main`。
4. **固定文件只能走 pull request。** `README.md`、`AGENTS.md`、`docs/` 底下的固定文件，都不可以為了回報進度而修改。
5. **不要新增其他文件。** 允許的文件清單在 `tools/doc_rules.py`。工具自動產生的 `PLAN.md` 等檔案，內容請寫進工作檔。
6. **封存的紀錄保持原樣。** `docs/records/`、`experiments/` 和 `results/archive/` 是當時的紀錄，裡面的舊路徑不改；新舊路徑對照見 `docs/layout_migration.tsv`。`results/final/move_20261121/` 的清單記的是本地備份的路徑，也不改寫。

`tools/check_code_rules.py` 和 GitHub Actions 會檢查文件位置、工作檔欄位、現行文件的內部連結和成績表格；直接推到 `main` 的 commit 如果改了工作檔以外的內容，GitHub Actions 也會失敗（引入這套規則的那次推送除外）。`main` 沒有分支保護，所以失敗只會顯示紅叉，不會真的擋下來。

## 9. 專案慣例

- 環境：本儲存庫沒有自己的環境。在並排的 AI_CUP_2026-1 執行 `bash scripts/nano4/setup.sh` 建立 `.venv`，所有程式從 AI_CUP_2026-1 的根目錄、以該 `.venv` 執行；`aicup_pid.paths` 會自動找到並排的 AI_CUP_2026-1，也可用環境變數 `AICUP1_ROOT` 指定。沒有 `.venv` 時，`tools/run_checks.sh` 以結束碼 2 回報未完成。
- 計算：國網 nano4 的所有計算（含 pytest、分析與語法樹比對）一律以 AI_CUP_2026-1 的 `sbatch scripts/nano4/run.sbatch <指令>` 送到計算節點；登入節點只做編輯、送件與查詢。`scripts/nano4/` 的啟動腳本含本隊叢集上的絕對路徑，在其他環境執行時要改成自己的路徑。
- 目錄：
  - `src/aicup_pid/`：玩家辨識套件（統計成分、融合、配對、序列網路的資料處理）。
  - `scripts/`：最終流程的程式（特徵抽取、外部資料、序列網路訓練、題池微調、預測與合併）。
  - `scripts/nano4/`：最終網路家族的訓練與向量輸出啟動腳本。
  - `tests/`：pytest 測試；部分測試以 `experiments/` 的封存程式為對象。
  - `tools/`：檢查工具與其基準檔。
  - `docs/`：現行文件（`TASK.md`、`PID_FINAL_NUMBERS.md`、`BRANCHES.md`）；`docs/records/` 是封存紀錄（方法紀錄、計畫、備忘）；`docs/work/` 是工作檔。
  - `experiments/`：封存的實驗程式與啟動腳本，維持原目錄深度；要執行時以兩個儲存庫的 tag `pre-restructure-20261127` 的目錄結構為準。
  - `results/`：最終融合文件、最終提交檔的檢查報告、`results/embed/`、`results/final/`（不在儲存庫內的權重清單與資料閘門紀錄）；`results/archive/` 是封存的實驗結果。
  - 執行期資料夾（不進版控）：`data/`、`upload/`。
- 資料：官方測試與候選檔置於 `data/`，不進版控；外部資料（Fox Go Dataset，GPL-3.0）要在 README 的外部資源與報告註明來源、版本與授權，帳號名稱只以加鹽雜湊使用，不做去匿名化。競賽資料僅供本競賽使用，不得用於識別真實使用者。
- 不進版控：`data/`、`upload/`、`*.npz`、`*.pt`、`results/**/*.csv` 與提交檔。GitHub 部署金鑰與權杖不寫進程式或對話紀錄。
- 分支：`main` 是唯一的長期分支。修改在短期分支開 pull request，`check` 通過後用一般合併（不要 squash），規則檢查才認得出這是 PR，合併後刪除分支。不改寫歷史：不 force push、不 rebase 已推送的提交。
- 檔案結構、流程與各模組的輸入、輸出寫在 `README.md`，開始工作前要先讀。`README.md` 第 5 節與 `docs/PID_FINAL_NUMBERS.md` 第 10 節的節號被官方報告引用，不要變動。結構有變動時，同步更新 `README.md`、`docs/layout_migration.tsv`，以及官方報告第柒節列出的路徑（報告在儲存庫外產生，改完要重建）。

## 10. 兩個儲存庫的關係

- AI_CUP_2026-2（本儲存庫，競賽一）與 AI_CUP_2026-1（競賽二）要並排放置在同一個上層資料夾。
- 本儲存庫使用競賽二的 `src/aicup/`、`scripts/` 第一層的特徵程式（`trunk_extract.py` 等）、`scripts/m_series/` 的 `m88_seqid.py`、`o26_parallel_forward.py`、`m84_rows_extract.py`、`m90_ply_extract.py`、`splits/player_folds.csv` 與其環境。
- 競賽二的對手辨識程式匯入本儲存庫的 `src/aicup_pid/`。移動或改名這些檔時，兩個儲存庫要在同一批推送裡一起更新。
