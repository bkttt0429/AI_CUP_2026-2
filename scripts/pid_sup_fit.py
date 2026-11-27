#!/usr/bin/env python
"""The supervised projection of the trunk vectors for probe B (METHOD_NOTES 2.88): fitted on per-game rows of unprojected
statistics (scripts/pid_trunk_extract.py --full) of the training players of --fit-folds only (the players of the folds whose
simulated pools score the probe never enter; fold 0 is never read).
  pid_sup_fit.py --rows DIR [DIR ...] --fit-folds 3 4 --k 64 --out proj.npz
Output: mu [768], sd [768], W [768, k] with feature = ((x - mu) / sd) @ W for x = [pooled vector, played-point vector].
  --blocks g x e   the projection over [pooled, played-point, policy-EXPECTED played-point] means instead (METHOD_NOTES 2.102;
                   rows of scripts/pid_pol_extract.py, which hold `pole` next to `full`); the output then carries `blocks`,
                   and scripts/pid_sup_ply_extract.py computes the expected vector when it sees "e" there."""
import argparse  # 解析命令列參數
import glob  # 列出每局列的檔案
import sys  # 調整模組搜尋路徑與結束碼
import time  # 計時
from pathlib import Path  # 路徑操作

import numpy as np  # 陣列運算
import pandas as pd  # 讀取玩家分折表

ROOT = Path(__file__).resolve().parents[1]  # 本儲存庫根目錄
for d in (ROOT / "src", ROOT / "scripts"):  # 本儲存庫的套件與腳本資料夾
    if str(d) not in sys.path:  # 尚未在搜尋路徑中
        sys.path.insert(0, str(d))  # 加到最前面
import aicup_pid  # noqa: E402  本儲存庫套件，提供訓練集路徑（匯入時也把 AI_CUP_2026-1 的 src 與 scripts 放上搜尋路徑）
from aicup_pid import v2  # noqa: E402  取得訓練棋局清單
from aicup_pid.paths import FOLDS_CSV  # noqa: E402  玩家分折表的位置
import pid_sup  # noqa: E402  投影的求解函式


def main(argv=None):  # 求投影並寫出 npz
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])  # 以模組說明第一行當描述
    ap.add_argument("--rows", nargs="+", required=True)  # 每局統計列的資料夾
    ap.add_argument("--fit-folds", nargs="+", type=int, default=[3, 4])  # 只用這些折的玩家求投影
    ap.add_argument("--k", type=int, default=64)  # 投影後的維度
    ap.add_argument("--min-games", type=int, default=8)  # 玩家至少要有的局數
    ap.add_argument("--out", required=True)  # 輸出的投影檔
    ap.add_argument("--blocks", nargs="+", default=None, choices=("g", "x", "e"), help="blocks of per-game means to project jointly (default: g x through pid_sup.fit)")  # 要一起投影的平均區塊
    a = ap.parse_args(argv)  # 解析參數
    t0 = time.time()  # 計時起點
    assert 0 not in a.fit_folds, "fold 0 is never read"  # 第 0 折一律不使用
    games = v2.training_games(aicup_pid.TRAIN_DIR).merge(pd.read_csv(FOLDS_CSV, dtype={"player_id": str})[["player_id", "fold"]], on="player_id")  # 每局訓練棋局附上其玩家的折
    owner_of = dict(zip(games.game_key[games.fold.isin(a.fit_folds)], games.player_id[games.fold.isin(a.fit_folds)]))  # 指定折的棋局：棋局鍵 -> 玩家
    rows, own = {}, {}  # 棋局鍵 -> 每局列（own 沒有用到）
    for d in a.rows:  # 每個資料夾
        for f in sorted(glob.glob(f"{d}/*.npz")):  # 每個檔案
            if f.endswith(".tmp.npz"):  # 寫到一半的暫存檔
                continue  # 略過
            z = np.load(f)  # 讀入檔案
            if "full" not in z.files or (a.blocks and "e" in a.blocks and "pole" not in z.files):  # 缺少需要的區塊
                continue  # 略過此檔
            C_ = z["full"].shape[1] // 4  # 主幹通道數（full 含四個區塊）
            if a.blocks:  # 選定的平均區塊並排在一起
                src = {"g": z["full"][:, 0:C_], "x": z["full"][:, 2 * C_:3 * C_]}  # 全盤向量與落點向量的每局平均
                if "e" in a.blocks:  # 需要預期向量
                    src["e"] = z["pole"][:, 0:C_]  # policy 預期落點向量的平均
                mat = np.hstack([src[b] for b in a.blocks])  # 依指定順序並排
            else:  # 預設做法
                mat = z["full"]  # 整個 full 交給 pid_sup.fit 取區塊
            for k, r in zip(z["keys"].astype(str), mat):  # 每局一列
                if k in owner_of and k not in rows:  # 屬於指定折且尚未收過
                    rows[k] = r.astype(np.float32)  # 收下這一列
    keys = sorted(rows)  # 固定順序
    owners = np.array([owner_of[k] for k in keys])  # 每局的玩家
    u, c = np.unique(owners, return_counts=True)  # 每位玩家的局數
    keep = np.isin(owners, u[c >= a.min_games])  # 只留局數足夠的玩家
    X, owners = np.stack([rows[k] for k in keys])[keep], owners[keep]  # 求投影用的資料
    mu, sd, W, lam = pid_sup.fit_means(X, owners, a.k) if a.blocks else pid_sup.fit(X, owners, a.k)  # 求有監督投影
    print(f"{len(X)} games of {len(set(owners.tolist()))} players of folds {a.fit_folds}; between / within ratio of directions 1, 8, 16, 32, {a.k}: "  # 印出局數、玩家數與折
          + " ".join(f"{lam[i - 1]:.2f}" for i in (1, 8, 16, 32, a.k)) + f" ({time.time() - t0:.0f} s)")  # 以及幾個方向的玩家間／玩家內變異比
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)  # 建立輸出資料夾
    np.savez(a.out, mu=mu, sd=sd, W=W, ratio=lam.astype(np.float32), fit_folds=np.array(a.fit_folds), games=np.array(len(X)), **({"blocks": np.array(a.blocks)} if a.blocks else {}))  # 寫出投影、變異比與設定
    print(f"wrote {a.out}")  # 回報輸出位置
    return 0  # 成功


if __name__ == "__main__":  # 直接執行時
    sys.exit(main())  # 以 main 的回傳值結束
