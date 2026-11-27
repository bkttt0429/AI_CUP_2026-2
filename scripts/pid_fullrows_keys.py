#!/usr/bin/env python
"""Game keys of every training game of the players of folds 1-4 (never fold 0), for scripts/pid_trunk_extract.py --keys --full
(METHOD_NOTES 2.75). Keys that already have a row in the given directories are left out.   pid_fullrows_keys.py <out.txt> [rows dir ...]"""
import glob  # 列出已抽好的列檔
import sys  # 命令列參數與匯入路徑
from pathlib import Path  # 以路徑物件處理檔案

import numpy as np  # 讀取 .npz 列檔中的棋局鍵
import pandas as pd  # 讀取玩家分折表

R2 = Path(__file__).resolve().parents[1]  # 本儲存庫根目錄（本檔位於 scripts/）
sys.path.insert(0, str(R2 / "src"))  # 讓 aicup_pid 套件可以匯入
import aicup_pid  # noqa: E402  匯入前須先設定 sys.path
from aicup_pid import v2  # noqa: E402  提供訓練棋局清單
from aicup_pid.paths import FOLDS_CSV  # noqa: E402  玩家分折表的位置

games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")  # 每一局訓練棋局附上擁有者所屬的折
keys = set(games.game_key[games.fold != 0])  # 第 1-4 折玩家的棋局鍵
assert not (set(games.game_key[games.fold == 0]) & keys), "a game listed under a fold-0 player and another player"  # 同一局不可同時列在第 0 折玩家與其他玩家之下
have = set()  # 已經有列的棋局鍵
for d in sys.argv[2:]:  # 逐一檢查指定的列資料夾
    for f in glob.glob(f"{d}/*.npz"):  # 資料夾內的每個列檔
        if not f.endswith(".tmp.npz"):  # 略過寫到一半的暫存檔
            have.update(np.load(f)["keys"].astype(str).tolist())  # 收集已抽過的棋局鍵
todo = sorted(keys - have)  # 還沒有列、需要抽取的棋局
tmp = Path(sys.argv[1] + ".tmp")  # 暫存檔
tmp.write_text("\n".join(todo) + "\n")  # 先寫進暫存檔
tmp.replace(sys.argv[1])  # 其他節點會等這個檔：它整個出現或完全不出現
print(f"{len(keys)} training games of folds 1-4, {len(keys & have)} already have rows, {len(todo)} to extract")  # 回報總局數、已有列與待抽的局數
