"""Trunk-style component (fork plan line B4, trunk variant; METHOD_NOTES 2.21).

scripts/pid_trunk_extract.py stores one 256-dimensional row per game: mean and standard deviation of the humanv0 trunk's
board-pooled and played-point vectors (each projected to 64 dimensions) over the target player's positions. The component
is the style LDA of aicup_pid.style (shared shrunk within-player covariance, Gaussian log-likelihood of the question's
mean row) on those rows instead of the 32 hand-made statistics. `trunk_feats` picks the columns: all (256), mean (the two
means, 128), pooled (128), local (played point, 128).
"""
import glob  # 列出資料夾內的分段檔

import numpy as np  # 讀取 npz 與陣列運算

from aicup_pid.style import StyleModel  # 沿用 aicup_pid.style 的風格 LDA 模型，改吃主幹列

COLUMNS = {"all": np.arange(256), "mean": np.r_[0:64, 128:192], "pooled": np.arange(0, 128), "local": np.arange(128, 256)}  # trunk_feats 各選項對應的欄：全部 256 維、兩個平均 128、全盤池化 128、落點 128


def load_store(*dirs):  # 讀入每局的主幹列
    """{game key: [256] float32} from every part file of the given directories."""
    store = {}  # 局鍵 → 主幹列
    for d in dirs:  # 逐一處理每個資料夾
        for f in sorted(glob.glob(f"{d}/*.npz")):  # 依檔名順序讀每個分段檔
            if f.endswith(".tmp.npz"):  # 抽取程式寫入中的暫存檔
                continue  # 略過，避免讀到不完整的資料
            z = np.load(f, allow_pickle=False)  # 讀入分段檔（不允許 pickle）
            for k, row in zip(z["keys"], z["feats"]):  # 每局的鍵與 256 維列
                store[str(k)] = row  # 鍵轉成 Python 字串；同一局重複出現時以後讀的檔為準
    return store  # 回傳完整的對照表


def rows_of(store, keys, cols):  # 取出指定棋局的主幹列
    missing = [k for k in keys if k not in store]  # 找出沒有主幹列的棋局
    if missing:  # 有缺漏時
        raise KeyError(f"{len(missing)} games have no trunk row, e.g. {missing[:3]}")  # 一次報出缺少的局數與前三個局鍵
    return np.stack([store[k][cols] for k in keys]).astype(np.float64)  # 選取欄位後疊成矩陣，轉成 float64 再計算


def fit_trunk(cand_keys_by_player, store, cfg):  # 在候選人的主幹列上擬合風格模型
    """StyleModel on the candidates' trunk rows (cfg keys trunk_feats, trunk_shrink)."""
    cols = COLUMNS[cfg.get("trunk_feats", "all")]  # 依設定選欄位，預設全部 256 維
    model = StyleModel(shrink=cfg.get("trunk_shrink", 0.1)).fit({p: rows_of(store, keys, cols) for p, keys in cand_keys_by_player.items()})  # 每位候選人一個平均，共用收縮後的共變異數
    model.columns = cols  # 記下欄位，計分時用同一組
    return model  # 回傳擬合好的模型


def score_trunk(model, question_keys, store):  # 題目對候選人的主幹風格分數
    """[Q, C] trunk-style log scores for questions given as lists of game keys."""
    if not question_keys:  # 沒有題目時另外處理（空清單無法 np.stack）
        return np.zeros((0, len(model.players)))  # 回傳空的 [0, C] 矩陣
    return np.stack([model.log_scores(rows_of(store, keys, model.columns)) for keys in question_keys])  # 每題一列對數分數
