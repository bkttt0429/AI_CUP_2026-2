#!/usr/bin/env python
"""Player embedding for the v2 pools (METHOD_NOTES 2.29): the B8 network (aicup_pid.embed) trained on training players' rows of
BOTH colours, in the v2 geometry: support of 10 games, query sets of 1 / 2 / 3 / 5 games, any colour mix. Rows: the rank
session's per-game-side trunk rows (aicup_pid.v2.load_side_rows; same features as data/trunk, both colours, about 855k games).
Writes emb2_p12.pt (trained on folds 3, 4: for simulated pools of folds 1, 2), emb2_p34.pt and emb2_final.pt (folds 1-4; no
official game enters any training; fold 0 never used)."""
import argparse  # 解析命令列參數
import sys  # 設定 sys.path 與結束碼
import time  # 計時
from pathlib import Path  # 處理路徑

import numpy as np  # 陣列運算
import pandas as pd  # 讀取玩家分折表
import torch  # 儲存模型權重

ROOT = Path(__file__).resolve().parents[1]  # 本儲存庫根目錄
sys.path.insert(0, str(ROOT / "src"))  # 讓 aicup_pid 可以匯入
import aicup_pid  # noqa: E402  共用的訓練資料位置（TRAIN_DIR）
from aicup_pid.embed import PlayerEmbedding  # noqa: E402  B8 玩家嵌入網路
from aicup_pid.paths import AICUP1, FOLDS_CSV  # noqa: E402  AI_CUP_2026-1 的位置與玩家分折表
from aicup_pid.v2 import load_phase_rows, load_side_rows, training_games  # noqa: E402  讀每局列與訓練棋局清單


def main(argv=None):  # 命令列進入點
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以模組說明第一行當描述
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "embed"))  # 模型輸出資料夾
    ap.add_argument("--steps", type=int, default=3000)  # 訓練步數
    ap.add_argument("--support", type=int, default=10)  # 支撐集局數（同新增池候選人的 10 局）
    ap.add_argument("--tag", default="emb2")  # 輸出檔名前綴
    ap.add_argument("--phase-dir", default=None, help="experiments/pid_v2_phase_rows.py output: train on the 640-dim rows with phase means (use --tag emb3)")  # 改用含階段平均的 640 維列
    ap.add_argument("--seeds", nargs="+", type=int, default=[0], help="seed 0 keeps the plain file names; seed k > 0 writes <tag>_<name>_s<k>.pt")  # 要訓練的種子
    a = ap.parse_args(argv)  # 解析參數
    t0 = time.time()  # 計時起點
    rows = load_phase_rows(AICUP1, a.phase_dir) if a.phase_dir else load_side_rows(AICUP1)  # 讀取每局單方的列
    games = training_games(aicup_pid.TRAIN_DIR)  # 所有訓練棋局（玩家、等級、鍵、顏色）
    folds = pd.read_csv(FOLDS_CSV, dtype={"player_id": str})  # 玩家分折表
    g = games.merge(folds[["player_id", "fold"]], on="player_id")  # 每局標上玩家所屬的折
    g = g[g.game_key.isin(rows.keys())]  # 只留有列的棋局
    print(f"{len(rows)} rows of {len(next(iter(rows.values())))} dims, {len(g)} games of {g.player_id.nunique()} players with a fold; colours {g.color.value_counts().to_dict()} ({time.time() - t0:.0f} s)", flush=True)  # 回報資料規模
    for name, fs in (("p12", (3, 4)), ("p34", (1, 2)), ("final", (1, 2, 3, 4))):  # 三個模型與各自的訓練折
        sub = g[g.fold.isin(fs)]  # 這個模型可用的棋局
        by = {p: np.stack([rows[k] for k in ks]) for p, ks in sub.groupby("player_id")["game_key"]}  # 每位玩家的列疊成矩陣
        for seed in a.seeds:  # 每個種子各訓練一次
            m = PlayerEmbedding(steps=a.steps, support=a.support, query_sizes=(1, 2, 3, 5), p_aug=0.0, min_games=12, seed=seed)  # 新增池的幾何（題目 1／2／3／5 局）；不用對稱擴增；至少 12 局的玩家才參加訓練
            t1 = time.time()  # 單一模型計時
            m.fit(by, None, log=None)  # 訓練嵌入（不提供對稱變換後的列）
            dest = Path(a.out_dir) / (f"{a.tag}_{name}.pt" if seed == 0 else f"{a.tag}_{name}_s{seed}.pt")  # 種子 0 用原檔名
            torch.save(m.state_dict(), dest)  # 存模型
            print(f"{name} seed {seed}: folds {fs}, {m.players_trained} players, final loss {m.history[-1]['loss']:.3f}, {time.time() - t1:.0f} s -> {dest}", flush=True)  # 回報訓練結果


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值為結束碼
