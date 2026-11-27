"""Handicap flag as a per-candidate habit (fork plan line B5): a Dirichlet-multinomial over the record's HD class.

The official records keep `HD[n]` and nothing else besides the moves; on the 11/04 files HD is 0 or 1 only and the
per-candidate HD[1] rate is strongly player-specific (sd 0.22 across players against 0.037 for a binomial at the pool
rate; METHOD_NOTES 2.12). hd_class maps a record to 0 (HD 0 or absent), 1 (HD 1) or 2 (HD >= 2, training set only).
HDModel: p_c(k) = (n_c(k) + alpha p_bg(k)) / (n_c + alpha) with p_bg(k) = (n_bg(k) + beta / K) / (n_bg + beta), the same
shrinkage form as aicup_pid.local; log_scores sums log p_c over the query games' classes, one value per candidate.
"""
import numpy as np  # 陣列運算

from aicup.sgf import handicap  # 讀取棋譜的 HD 值

K = 3  # HD 類別：0、1、>= 2


def hd_class(sgf):  # 棋譜的讓子類別
    """0 for HD[0] or no HD property, 1 for HD[1], 2 for HD[n >= 2]."""
    h = handicap(sgf)  # 讀 HD 值（沒有時為 None）
    return 0 if not h else 1 if h == 1 else 2  # 對應到三個類別


class HDModel:  # 讓子習慣的候選人模型
    """fit({player_id: [sgf strings]}); log_scores([sgf strings]) -> [n_players] in self.players' order."""

    def __init__(self, alpha=2.0, beta=5.0):  # 設定平滑參數
        if alpha <= 0 or beta <= 0:  # 參數必須為正
            raise ValueError("alpha > 0, beta > 0")  # 拒絕不合法的參數
        self.alpha, self.beta = float(alpha), float(beta)  # 個人與背景的平滑強度

    def fit(self, sgfs_by_player):  # 以候選人的棋譜擬合
        self.players = sorted(sgfs_by_player)  # 候選人依 id 排序
        counts = np.zeros((len(self.players), K))  # 每位候選人各類別的局數
        for i, p in enumerate(self.players):  # 逐位候選人
            for s in sgfs_by_player[p]:  # 逐局棋譜
                counts[i, hd_class(s)] += 1  # 累計類別
        bg = counts.sum(axis=0)  # 全題池的類別計數（背景）
        p_bg = (bg + self.beta / K) / (bg.sum() + self.beta)  # 平滑後的背景分布
        self.logp = np.log((counts + self.alpha * p_bg) / (counts.sum(axis=1, keepdims=True) + self.alpha))  # 每位候選人向背景收縮後的對數機率
        self.class_counts = bg.astype(int).tolist()  # 記下背景計數
        return self  # 回傳自身以便串接

    def _counts(self, sgfs):  # 查詢棋局的類別計數
        v = np.zeros(K)  # 各類別的局數
        for s in sgfs:  # 逐局
            v[hd_class(s)] += 1  # 累計類別
        return v  # 回傳計數

    def log_scores(self, sgfs):  # 查詢棋局對每位候選人的對數分數
        return self.logp @ self._counts(sgfs)  # 各類別的 log p 乘以局數後相加

    def state(self):  # 輸出可存成 JSON 的狀態
        return {"alpha": self.alpha, "beta": self.beta, "players": list(self.players), "class_counts": self.class_counts}  # 參數、候選人與背景計數
