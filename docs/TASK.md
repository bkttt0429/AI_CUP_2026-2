# 任務、資料與規則（玩家辨識，競賽一）

來源：AIdea 競賽頁（`https://www.aidea-web.tw/topic/dc0b799e-690f-4e84-a597-93e68d8ba688`，使用者轉述的資料說明）、官方 Tutorial README（`https://github.com/AILAB-NDHU/AICup-2026-Tutorial`，2026-10-29 讀取）、資料集 repo（`https://github.com/AILAB-NDHU/aicup2026-go-dataset`）。雲端 session 的網路白名單擋住 `aidea-web.tw`，競賽頁本身還沒直接核對。

## 資料

- 來源：日本棋院幽玄網站 19 路棋譜（2021–2023，已授權）。匿名化：移除姓名、段位、勝負、註解，只留棋盤設定與落子；`player_id` 不可回推身分。資料集只供本競賽使用，**不得用於識別真實使用者**。
- **訓練集（兩項任務共用）**：100 萬局，10 級各 10 萬局、執黑執白各半；`train_D.csv … train_6D.csv`；欄位 `player_id, game_id, rank, color, sgf_content`。和 AI_CUP_2026-1 的 REPORT.md §3 同一份（8,007 位玩家，每人只屬一級，`game_id` 只在檔內唯一，SGF 只有 `SZ[19]` 與落子）。
- **候選庫**：`player_identification_candidates_public.csv`、`…_private.csv`，各 400 位候選玩家 × 100 局，共 40,000 局，**只收執黑棋譜**；欄位 `game_id, player_id, sgf_content, color`（恆為 B）。正解必在對應候選庫中。Public 與 Private 候選庫不得混用。
- **測試題**：`player_identification_test_public.csv`、`…_private.csv`，各 400 題；每題 5／10／15／20 局（各 100 題），只用執黑棋譜；欄位 `question_id, num_games, sgf_1…sgf_20（不足補空）, color_1…color_20（恆為 B）`。`question_id` 前綴 `pub_`／`priv_`。
- 候選庫與測試題的 `player_id` 是否與訓練集一致、候選玩家是否在訓練集內：**未知**，11/04 拿到檔案後第一件事就是比對（見 ARCHITECTURE §4）。

## 評分

每題：正解在 Top-5 的第 r 名得 e^{−(r−1)}，不在前 5 名得 0。

| r | 1 | 2 | 3 | 4 | 5 | 不在 |
| --- | --- | --- | --- | --- | --- | --- |
| 分 | 1.000 | 0.368 | 0.135 | 0.050 | 0.018 | 0 |

分數為題目平均；Public、Private 各自以自己的 400 題計算。**Top-1 幾乎是全部**：第 2 名只值 0.37，第 3 名以後合計不到 0.2。官方 baseline（`GoPlayerResNet` + Triplet Loss，17 層平面：我方 8 手歷史 + 對方 8 手 + 執色；以 embedding 距離配對）Public **0.2845**。

## 提交

- 一個檔 `submission_player.csv`，欄位 `question_id, top_1, top_2, top_3, top_4, top_5`，**800 列**（400 public + 400 private）。
- 五個預測必須互不相同、且來自該題所屬測試集的候選庫（`pub_` 題不得填 `priv_p_` 玩家）。缺任一題整份無效。
- 每日上傳次數與「最後一次計分」規則待競賽頁核對（競賽二是每天 5 次、以最後一次為準）。

## 時程（與競賽二相同，待競賽頁核對）

報名 11/03 23:59 截止（**兩個競賽要分別報名**）；測試集與候選庫 11/04 11:00 發佈；截止 11/24 23:59；Private 榜 11/25。
