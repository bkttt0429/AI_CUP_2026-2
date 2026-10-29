# AI_CUP_2026-2

2026 AI CUP 圍棋競賽「玩家辨識」（競賽一）：每題給同一位未知玩家的 5／10／15／20 局**執黑**棋譜，從對應候選庫的 400 位玩家（每人 100 局執黑棋譜）中辨識出這個人，輸出信心度最高的前 5 名。

- 任務、資料與規則：[docs/TASK.md](docs/TASK.md)
- 問題本質、方法梯子與驗證協議：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- 方法筆記（已確認的事實與收斂的決策）：[METHOD_NOTES.md](METHOD_NOTES.md)

訓練資料和棋力預測競賽（[AI_CUP_2026-1](https://github.com/bkttt0429/AI_CUP_2026-1)）**完全相同**（100 萬局、8,007 位玩家）；SGF 解析、KataGo 特徵、位置快取、評分與 nano4 的操作都沿用那個倉庫的 `aicup` 套件，本倉庫只放玩家辨識專用的程式與結果。資料、權重與虛擬環境不進 git。
