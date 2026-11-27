"""Reference-player affinity (fork plan line B7, user 2026-11-06): training players as an anonymous basis.

Every game is scored against R reference players (training players outside the pool: their whole-game local-shape
tokens, aicup_pid.local, and orientation tokens, aicup_pid.orient, with the usual Dirichlet smoothing), giving a vector of
R log-likelihoods; the vector is z-scored over the references and averaged over a set of games. A candidate's profile is
the average over its library, a question's over its games, and the component score is the cosine between the two
profiles (the fusion standardises each question's row, so the scale is immaterial). Nobody is identified: the
references only span a space in which "plays like the same people" is measured, which can match a question to its
candidate even when their own opening patterns differ. Reference players are chosen deterministically (min_black Black
games, up to cap games each, seeded) and, in the simulated pools, come from the other fold pair.
"""
import hashlib  # 由玩家 id 衍生亂數種子

import numpy as np  # 陣列運算
import scipy.sparse as sp  # 稀疏的符記計數矩陣

from aicup_pid.local import LocalModel  # 局部棋形模型
from aicup_pid.orient import OrientModel  # 棋盤方向模型


def reference_players(games, folds_df, folds, min_black=30, cap=40, seed=0, exclude=()):  # 挑選參考玩家與其棋局
    """{player_id: [game_key, ...]} for the players of `folds` with >= min_black Black games (not in `exclude`), each with
    up to `cap` games drawn by a per-player seeded shuffle."""
    folds = [int(f) for f in folds]  # 折編號轉成整數
    pl = set(folds_df[folds_df["fold"].isin(folds)]["player_id"].astype(str)) - set(exclude)  # 指定折中、未被排除的玩家
    out = {}  # 參考玩家 -> 棋局鍵
    for p, grp in games[games["player_id"].isin(pl)].groupby("player_id"):  # 逐位玩家
        keys = sorted(grp["game_key"].tolist())  # 排序後的棋局鍵
        if len(keys) < min_black:  # 黑棋局數不夠
            continue  # 略過
        rng = np.random.RandomState(int(hashlib.sha256(f"{seed}:{p}".encode()).hexdigest()[:8], 16))  # 每位玩家固定的亂數
        rng.shuffle(keys)  # 打散棋局
        out[str(p)] = keys[:cap]  # 最多取 cap 局
    return out  # 回傳字典


class RefModel:  # 參考玩家親和度模型
    """fit({reference player: [move lists]}); vectors([move lists]) -> [G, R] z-scored log-likelihoods; profile(...).
    orient_raw=0 drops the orientation block (local-shape tokens only): every token is then invariant under the 8 board
    symmetries, for files whose games may be rotated / mirrored."""

    def __init__(self, alpha=2.0, beta=5.0, local_start=12, orient_depth=12, orient_raw=2):  # 建立局部棋形與方向兩個區塊
        if orient_raw < 0:  # 參數不合法
            raise ValueError("orient_raw must be >= 0")  # 拒絕
        self.local = LocalModel(alpha=alpha, beta=beta, start=local_start)  # 局部棋形區塊
        self.orient = OrientModel(alpha=alpha, beta=beta, depth=orient_depth, n_raw=orient_raw) if orient_raw else None  # 方向區塊（orient_raw=0 時不用）
        self.blocks = (self.local,) + ((self.orient,) if self.orient else ())  # 實際使用的區塊

    def fit(self, games_by_player):  # 以參考玩家的棋局擬合
        for model in self.blocks:  # 每個區塊
            model.fit(games_by_player)  # 各自擬合
        self.players = self.local.players  # 參考玩家的順序
        self.logp = np.hstack([m.logp for m in self.blocks]).astype(np.float32)  # [R, V_local (+ V_orient)]：兩個區塊的對數機率表並排
        return self  # 回傳自身以便串接

    def _counts(self, games):  # 棋局的稀疏符記計數
        """Sparse [G, V_local + V_orient] token counts."""
        rows, cols, vals = [], [], []  # 稀疏矩陣的座標與值
        shifts = [0, self.local.vocabulary + 1][:len(self.blocks)]  # 各區塊在合併表中的欄位起點
        for g, moves in enumerate(games):  # 逐局
            for model, shift in zip(self.blocks, shifts):  # 逐個區塊
                for tok in model.tokenize(moves):  # 逐個符記
                    rows.append(g)  # 這一局
                    cols.append(shift + model.vocab.get(tok, model.vocabulary))  # 符記的欄位（沒見過的放在該區塊最後一欄）
                    vals.append(1.0)  # 計數 1
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(games), self.logp.shape[1]), dtype=np.float32)  # 重複的座標會自動相加

    def vectors(self, games):  # 每局對每位參考玩家的標準化對數概似
        """[G, R] per-game log-likelihoods under every reference, z-scored across the references."""
        ll = np.asarray(self._counts(games) @ self.logp.T, dtype=np.float64)  # 符記計數乘以對數機率
        ll -= ll.mean(axis=1, keepdims=True)  # 每局減去平均
        sd = ll.std(axis=1, keepdims=True)  # 每局的標準差
        sd[sd < 1e-12] = 1.0  # 避免除以 0
        return ll / sd  # 回傳 z 分數

    def profile(self, games):  # 一組棋局的輪廓
        """[R] mean z-scored vector of a set of games (zeros for an empty set)."""
        if not games:  # 沒有棋局
            return np.zeros(len(self.players))  # 零向量
        return self.vectors(games).mean(axis=0)  # 各局向量取平均

    def state(self):  # 輸出可存成 JSON 的狀態
        return {"references": len(self.players), "local": self.local.state(), "orient": self.orient.state() if self.orient else None}  # 參考玩家數與兩個區塊的狀態


def cosine_scores(question_profiles, candidate_profiles):  # 輪廓之間的餘弦相似度
    """[Q, C] cosine similarity between question and candidate profiles."""
    Q = np.asarray(question_profiles, dtype=np.float64)  # 題目的輪廓
    C = np.asarray(candidate_profiles, dtype=np.float64)  # 候選人的輪廓
    qn = np.linalg.norm(Q, axis=1, keepdims=True)  # 題目輪廓的長度
    cn = np.linalg.norm(C, axis=1, keepdims=True)  # 候選人輪廓的長度
    qn[qn < 1e-12] = 1.0  # 零向量不縮放
    cn[cn < 1e-12] = 1.0  # 零向量不縮放
    return (Q / qn) @ (C / cn).T  # 正規化後的內積
