#!/usr/bin/env python
"""Game keys of every training game of the players of FOLD 0 (METHOD_NOTES 2.201), for scripts/pid_sup_ply_extract.py --keys.

Fold 0 was never read by this repository until the user's consent of 2026-11-18 00:29 ("同意用 fold 0"); it is used for the
training of the sequence networks only (--all-folds), never for any validation tool. The counterpart for folds 1-4 is
scripts/pid_fullrows_keys.py.   pid_fold0_keys.py <out.txt>"""
import os  # 取得行程編號作為暫存檔名
import sys  # 命令列參數與匯入路徑
from pathlib import Path  # 以路徑物件處理檔案

import pandas as pd  # 讀取玩家分折表

R2 = Path(__file__).resolve().parents[1]  # 本儲存庫根目錄（本檔位於 scripts/）
sys.path.insert(0, str(R2 / "src"))  # 讓 aicup_pid 套件可以匯入
import aicup_pid  # noqa: E402  匯入前須先設定 sys.path
from aicup_pid import v2  # noqa: E402  提供訓練棋局清單
from aicup_pid.paths import FOLDS_CSV  # noqa: E402  玩家分折表的位置

games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")  # 每一局訓練棋局附上擁有者所屬的折
keys = set(games.game_key[games.fold == 0])  # 第 0 折玩家的棋局鍵
assert not (set(games.game_key[games.fold != 0]) & keys), "a game listed under a fold-0 player and another player"  # 同一局不可同時列在第 0 折玩家與其他玩家之下
tmp = Path(f"{sys.argv[1]}.{os.getpid()}.tmp")  # 每個行程各自的暫存檔
tmp.write_text("\n".join(sorted(keys)) + "\n")  # 先寫進暫存檔
tmp.replace(sys.argv[1])  # 多個節點可能同時寫這個檔：內容相同，且整個出現或完全不出現
print(f"{len(keys)} training games of the {games.player_id[games.fold == 0].nunique()} players of fold 0 "  # 回報第 0 折的局數與玩家數
      f"(folds 1-4: {int((games.fold != 0).sum())} games)")  # 以及第 1-4 折的局數
