"""Local-shape habits over the whole game (fork plan line B2): a per-candidate multinomial over per-move tokens.

Every Black move (from ply `start` on; passes give no token) becomes (displacement class to White's previous move,
displacement class to Black's own previous move, line, phase). A displacement class is (max(|dx|,|dy|), min(|dx|,|dy|))
clipped at 3, FAR when either exceeds 3, NONE when there is no such previous move -- all invariant under the 8 board
symmetries. line is the distance to the nearest edge (1..5, 5 = centre), phase 0 / 1 / 2 for ply < 60 / < 150 / rest.
LocalModel: p_c(tok) = (n_c(tok) + alpha p_bg(tok)) / (n_c + alpha), p_bg(tok) = (n_bg(tok) + beta / V) / (n_bg + beta) with
V = vocabulary + 1 (one slot for unseen tokens); log_scores sums log p_c over the query games' tokens, one value per
candidate (a [C, V] table times the query's token counts).
"""
from collections import Counter  # 符記計數

import numpy as np  # 陣列運算

BOARD = 19  # 棋盤大小
FAR = (4, 4)  # 位移超過 3 時的類別
NONE = (-1, -1)  # 沒有上一手時的類別
MAX_D = 3  # 位移分類的上限
PHASES = (60, 150)  # 階段的手數分界


def displacement(a, b):  # 兩手之間的位移類別
    """Class of the displacement from move a to move b ((x, y) tuples; b None -> NONE)."""
    if b is None or a is None:  # 缺少任一手
        return NONE  # 視為沒有上一手
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])  # 兩軸的距離
    hi, lo = max(dx, dy), min(dx, dy)  # 排序後與方向無關（對稱不變）
    if hi > MAX_D:  # 距離太遠
        return FAR  # 歸為遠距
    return (hi, lo)  # 近距離的類別


def line_of(x, y):  # 第幾線
    return min(x, y, BOARD - 1 - x, BOARD - 1 - y, 4) + 1  # 到最近邊的距離加 1，5 以上都算 5（中央）


def phase_of(t):  # 第 t 手所屬的階段
    return 0 if t < PHASES[0] else 1 if t < PHASES[1] else 2  # 依手數分成三段


def tokens(moves, start=12):  # 一局棋的局部棋形符記
    """[(to White's last move, to Black's last move, line, phase)] for Black's non-pass moves at ply >= start."""
    out, last = [], {"B": None, "W": None}  # 符記清單與雙方的上一手
    for t, (c, x, y) in enumerate(moves):  # 逐手處理
        if x is None:  # 這一手是 pass
            continue  # 不產生符記，也不更新上一手
        if c == "B" and t >= start:  # 黑棋且手數已達 start
            out.append((displacement((x, y), last["W"]), displacement((x, y), last["B"]), line_of(x, y), phase_of(t)))  # 產生一個符記
        last[c] = (x, y)  # 更新該方的上一手
    return out  # 回傳符記清單


class LocalModel:  # 局部棋形的候選人模型
    """fit({player_id: [move lists]}); log_scores([move lists]) -> [n_players] in self.players' order."""

    def __init__(self, alpha=2.0, beta=5.0, start=12):  # 設定平滑參數與起始手數
        if alpha <= 0 or beta <= 0 or start < 0:  # 檢查參數範圍
            raise ValueError("alpha > 0, beta > 0, start >= 0")  # 拒絕不合法的參數
        self.alpha, self.beta, self.start = float(alpha), float(beta), int(start)  # 保存參數

    def tokenize(self, game):  # 一局棋的符記（子類別可覆寫）
        """The tokens of one parsed game (subclasses change the vocabulary, e.g. aicup_pid.orient)."""
        return tokens(game, self.start)  # 預設為局部棋形符記

    def fit(self, games_by_player):  # 以候選人的棋局擬合
        self.players = sorted(games_by_player)  # 候選人依 id 排序
        counts = [Counter(tok for g in games_by_player[p] for tok in self.tokenize(g)) for p in self.players]  # 每位候選人的符記計數
        bg = Counter()  # 全題池的背景計數
        for c in counts:  # 累加每位候選人
            bg.update(c)  # 加進背景
        self.vocab = {tok: i for i, tok in enumerate(sorted(bg))}  # 符記 -> 欄位索引
        self.vocabulary = len(self.vocab)  # 詞彙數
        V = self.vocabulary + 1  # 最後一欄代表所有沒見過的符記
        n_bg = sum(bg.values())  # 背景的總數
        p_bg = np.full(V, self.beta / V)  # 背景分布的平滑項
        for tok, n in bg.items():  # 加上各符記的背景計數
            p_bg[self.vocab[tok]] += n  # 背景計數加到平滑項上
        p_bg /= n_bg + self.beta  # 正規化成機率
        table = np.empty((len(self.players), V))  # [C, V] 對數機率表
        for i, c in enumerate(counts):  # 逐位候選人
            row = self.alpha * p_bg  # 向背景收縮的部分
            for tok, n in c.items():  # 加上個人的計數
                row[self.vocab[tok]] += n  # 個人計數加到收縮項上
            table[i] = np.log(row / (sum(c.values()) + self.alpha))  # 正規化後取對數
        self.logp = table  # 保存對數機率表
        return self  # 回傳自身以便串接

    def _counts(self, games):  # 查詢棋局的符記計數向量
        v = np.zeros(self.vocabulary + 1)  # 含未見符記的那一欄
        for g in games:  # 逐局
            for tok in self.tokenize(g):  # 逐個符記
                v[self.vocab.get(tok, self.vocabulary)] += 1  # 沒見過的符記記在最後一欄
        return v  # 回傳計數

    def log_scores(self, games):  # 查詢棋局對每位候選人的對數分數
        return self.logp @ self._counts(games)  # [C, V] 表格乘以計數向量

    def state(self):  # 輸出可存成 JSON 的狀態
        return {"alpha": self.alpha, "beta": self.beta, "start": self.start, "vocabulary": self.vocabulary, "players": list(self.players)}  # 參數、詞彙數與候選人
