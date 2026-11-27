"""Opening language model: a candidate's Black-move habits in the first moves (docs/records/ARCHITECTURE.md ladder (1)).

Every game is put into a canonical orientation: of the 8 board symmetries the one whose transformed Black coordinates
among the first `depth` moves are lexicographically smallest (ties broken by all the moves), so a game and its mirror
image give the same tokens and the orientation does not depend on White's choices unless Black's own moves leave it
open. The model is a per-candidate back-off model over Black's moves among the first `depth` moves, with two contexts:
the full prefix (every move before it) and Black's own prefix (only Black's moves before it):

  p_c(mv | full) = (n_c(full, mv) + a_full * p_c(mv | black)) / (n_c(full) + a_full)
  p_c(mv | black) = (n_c(black, mv) + alpha * p_bg(mv | black)) / (n_c(black) + alpha)
  p_bg(mv | black) = (n_bg(black, mv) + beta / 362) / (n_bg(black) + beta)        (pool-wide background)

The full prefix captures replies to the opponent's moves when they repeat; Black's own prefix captures the repertoire
regardless of the opponent; the background keeps every probability finite. log_scores(query games) sums log p_c over
the games and their Black opening moves, one value per candidate.
"""
from collections import Counter, defaultdict  # 上下文與落子的計數

import numpy as np  # 陣列運算

from aicup.sgf import parse_moves  # 解析 SGF 落子

BOARD = 19  # 棋盤大小
PASS = (BOARD, BOARD)  # pass 在每種對稱下都保持這個座標
N_POINTS = BOARD * BOARD + 1  # 361 個點加上 pass


def _sym(k, x, y):  # 第 k 種對稱下的座標
    n = BOARD - 1  # 最大的座標值
    if x == BOARD:  # pass：不做轉換
        return x, y  # 原樣回傳
    if k & 1:  # 第 0 位元：鏡射 x
        x = n - x  # 鏡射 x
    if k & 2:  # 第 1 位元：鏡射 y
        y = n - y  # 鏡射 y
    if k & 4:  # 第 2 位元：交換 x 與 y
        x, y = y, x  # 交換兩軸
    return x, y  # 回傳新座標


def coords(moves):  # 落子座標清單
    """[(x, y)] of parsed moves with passes as PASS."""
    return [PASS if x is None else (x, y) for _, x, y in moves]  # pass 換成 PASS


def canonical(moves, depth):  # 把棋局轉到標準方向
    """(canonical coordinates of the whole game, symmetry index): the symmetry whose Black coordinates among the first `depth`
    moves are lexicographically smallest, ties broken by all of the first `depth` coordinates."""
    cs = coords(moves)  # 原始座標
    head = cs[:depth]  # 前 depth 手
    black = [c for (col, *_), c in zip(moves[:depth], head) if col == "B"]  # 前 depth 手中的黑棋座標
    best = min(range(8), key=lambda k: ([_sym(k, x, y) for x, y in black], [_sym(k, x, y) for x, y in head]))  # 先比黑棋、再比全部座標的字典序
    return [_sym(best, x, y) for x, y in cs], best  # 整局套用選定的對稱


def tokens(moves, depth):  # 開局模型的符記
    """[(full context, Black-only context, move)] for Black's moves among the first `depth` moves, canonical coordinates."""
    cs, _ = canonical(moves, depth)  # 標準方向的座標
    out, black_prefix = [], []  # 符記清單與黑棋前綴
    for i, (c, *_rest) in enumerate(moves[:depth]):  # 逐手看前 depth 手
        if c == "B":  # 只為黑棋落子產生符記
            out.append((tuple(cs[:i]), tuple(black_prefix), cs[i]))  # 完整前綴、黑棋前綴與這一手
            black_prefix.append(cs[i])  # 更新黑棋前綴
    return out  # 回傳符記清單


def tokens_raw(moves, depth):  # 原始座標的開局符記
    """tokens() without the symmetry normalisation: raw board coordinates, so a player's orientation habit (which corner
    they actually open in, 3-4 vs 4-3) is part of the pattern (fork plan line B6)."""
    cs = coords(moves)  # 原始座標，不轉方向
    out, black_prefix = [], []  # 符記清單與黑棋前綴
    for i, (c, *_rest) in enumerate(moves[:depth]):  # 逐手看前 depth 手
        if c == "B":  # 只為黑棋落子產生符記
            out.append((tuple(cs[:i]), tuple(black_prefix), cs[i]))  # 完整前綴、黑棋前綴與這一手
            black_prefix.append(cs[i])  # 更新黑棋前綴
    return out  # 回傳符記清單


def parse(sgf):  # 解析棋譜（頂層函式，可交給行程池）
    return parse_moves(sgf)  # 回傳 [(顏色, x, y)]


class OpeningModel:  # 開局退避模型
    """fit({player_id: [move lists]}); log_scores([move lists]) -> [n_players] in self.players' order."""

    def __init__(self, depth=12, alpha=2.0, beta=5.0, alpha_full=1.0, symmetry=True):  # 設定深度與平滑參數
        if depth < 1 or alpha <= 0 or beta <= 0 or alpha_full <= 0:  # 檢查參數範圍
            raise ValueError("depth >= 1, alpha > 0, beta > 0 and alpha_full > 0")  # 拒絕不合法的參數
        self.depth, self.alpha, self.beta, self.alpha_full = int(depth), float(alpha), float(beta), float(alpha_full)  # 保存參數
        self.symmetry = bool(symmetry)  # False：使用原始座標（tokens_raw），方向習慣保留在開局模式中

    def _tokens(self, moves):  # 依設定選擇符記函式
        return tokens(moves, self.depth) if self.symmetry else tokens_raw(moves, self.depth)  # 標準方向或原始座標

    def fit(self, games_by_player):  # 以候選人的棋局擬合
        self.players = sorted(games_by_player)  # 候選人依 id 排序
        self.index = {p: i for i, p in enumerate(self.players)}  # 候選人 -> 索引
        self.full, self.black = [], []  # 每位玩家：{上下文: Counter(落子)}
        self.bg = defaultdict(Counter)  # 全題池的背景計數（以黑棋前綴為上下文）
        for p in self.players:  # 逐位候選人
            full, black = defaultdict(Counter), defaultdict(Counter)  # 兩種上下文的計數
            for moves in games_by_player[p]:  # 逐局
                for fctx, bctx, mv in self._tokens(moves):  # 逐個符記
                    full[fctx][mv] += 1  # 完整前綴下的計數
                    black[bctx][mv] += 1  # 黑棋前綴下的計數
                    self.bg[bctx][mv] += 1  # 背景計數
            self.full.append(full)  # 保存該候選人的完整前綴計數
            self.black.append(black)  # 保存該候選人的黑棋前綴計數
        self.bg_total = {ctx: sum(c.values()) for ctx, c in self.bg.items()}  # 背景各上下文的總數
        self.full_total = [{ctx: sum(c.values()) for ctx, c in t.items()} for t in self.full]  # 每位候選人完整前綴的總數
        self.black_total = [{ctx: sum(c.values()) for ctx, c in t.items()} for t in self.black]  # 每位候選人黑棋前綴的總數
        return self  # 回傳自身以便串接

    def _p_bg(self, bctx, mv):  # 背景機率
        c = self.bg.get(bctx)  # 該上下文的背景計數
        return ((c[mv] if c else 0) + self.beta / N_POINTS) / (self.bg_total.get(bctx, 0) + self.beta)  # 向均勻分布平滑

    def log_scores(self, games):  # 查詢棋局對每位候選人的對數分數
        """Sum over the games' Black opening moves of log p_c(move | contexts), per candidate: [n_players]."""
        out = np.zeros(len(self.players))  # 每位候選人的累計分數
        for moves in games:  # 逐局
            for fctx, bctx, mv in self._tokens(moves):  # 逐個符記
                pb = self._p_bg(bctx, mv)  # 背景機率
                for i in range(len(self.players)):  # 逐位候選人
                    cb = self.black[i].get(bctx)  # 該候選人在黑棋前綴下的計數
                    p_black = ((cb[mv] if cb else 0) + self.alpha * pb) / (self.black_total[i].get(bctx, 0) + self.alpha)  # 黑棋前綴的機率，向背景收縮
                    cf = self.full[i].get(fctx)  # 該候選人在完整前綴下的計數
                    p_full = ((cf[mv] if cf else 0) + self.alpha_full * p_black) / (self.full_total[i].get(fctx, 0) + self.alpha_full)  # 完整前綴的機率，退避到黑棋前綴
                    out[i] += np.log(p_full)  # 累加對數機率
        return out  # 回傳 [n_players]

    def state(self):  # 輸出可存成 JSON 的狀態
        return {"depth": self.depth, "alpha": self.alpha, "beta": self.beta, "alpha_full": self.alpha_full, "symmetry": self.symmetry, "players": list(self.players)}  # 參數與候選人
