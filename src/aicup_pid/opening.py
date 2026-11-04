"""Opening language model: a candidate's Black-move habits in the first moves (docs/ARCHITECTURE.md ladder (1)).

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
from collections import Counter, defaultdict

import numpy as np

from aicup.sgf import parse_moves

BOARD = 19
PASS = (BOARD, BOARD)  # a pass keeps this coordinate under every symmetry
N_POINTS = BOARD * BOARD + 1


def _sym(k, x, y):
    n = BOARD - 1
    if x == BOARD:  # pass
        return x, y
    if k & 1:
        x = n - x
    if k & 2:
        y = n - y
    if k & 4:
        x, y = y, x
    return x, y


def coords(moves):
    """[(x, y)] of parsed moves with passes as PASS."""
    return [PASS if x is None else (x, y) for _, x, y in moves]


def canonical(moves, depth):
    """(canonical coordinates of the whole game, symmetry index): the symmetry whose Black coordinates among the first `depth`
    moves are lexicographically smallest, ties broken by all of the first `depth` coordinates."""
    cs = coords(moves)
    head = cs[:depth]
    black = [c for (col, *_), c in zip(moves[:depth], head) if col == "B"]
    best = min(range(8), key=lambda k: ([_sym(k, x, y) for x, y in black], [_sym(k, x, y) for x, y in head]))
    return [_sym(best, x, y) for x, y in cs], best


def tokens(moves, depth):
    """[(full context, Black-only context, move)] for Black's moves among the first `depth` moves, canonical coordinates."""
    cs, _ = canonical(moves, depth)
    out, black_prefix = [], []
    for i, (c, *_rest) in enumerate(moves[:depth]):
        if c == "B":
            out.append((tuple(cs[:i]), tuple(black_prefix), cs[i]))
            black_prefix.append(cs[i])
    return out


def tokens_raw(moves, depth):
    """tokens() without the symmetry normalisation: raw board coordinates, so a player's orientation habit (which corner
    they actually open in, 3-4 vs 4-3) is part of the pattern (fork plan line B6)."""
    cs = coords(moves)
    out, black_prefix = [], []
    for i, (c, *_rest) in enumerate(moves[:depth]):
        if c == "B":
            out.append((tuple(cs[:i]), tuple(black_prefix), cs[i]))
            black_prefix.append(cs[i])
    return out


def parse(sgf):
    return parse_moves(sgf)


class OpeningModel:
    """fit({player_id: [move lists]}); log_scores([move lists]) -> [n_players] in self.players' order."""

    def __init__(self, depth=12, alpha=2.0, beta=5.0, alpha_full=1.0, symmetry=True):
        if depth < 1 or alpha <= 0 or beta <= 0 or alpha_full <= 0:
            raise ValueError("depth >= 1, alpha > 0, beta > 0 and alpha_full > 0")
        self.depth, self.alpha, self.beta, self.alpha_full = int(depth), float(alpha), float(beta), float(alpha_full)
        self.symmetry = bool(symmetry)  # False: raw coordinates (tokens_raw), the orientation habit stays in the pattern

    def _tokens(self, moves):
        return tokens(moves, self.depth) if self.symmetry else tokens_raw(moves, self.depth)

    def fit(self, games_by_player):
        self.players = sorted(games_by_player)
        self.index = {p: i for i, p in enumerate(self.players)}
        self.full, self.black = [], []  # per player: {context: Counter(move)}
        self.bg = defaultdict(Counter)
        for p in self.players:
            full, black = defaultdict(Counter), defaultdict(Counter)
            for moves in games_by_player[p]:
                for fctx, bctx, mv in self._tokens(moves):
                    full[fctx][mv] += 1
                    black[bctx][mv] += 1
                    self.bg[bctx][mv] += 1
            self.full.append(full)
            self.black.append(black)
        self.bg_total = {ctx: sum(c.values()) for ctx, c in self.bg.items()}
        self.full_total = [{ctx: sum(c.values()) for ctx, c in t.items()} for t in self.full]
        self.black_total = [{ctx: sum(c.values()) for ctx, c in t.items()} for t in self.black]
        return self

    def _p_bg(self, bctx, mv):
        c = self.bg.get(bctx)
        return ((c[mv] if c else 0) + self.beta / N_POINTS) / (self.bg_total.get(bctx, 0) + self.beta)

    def log_scores(self, games):
        """Sum over the games' Black opening moves of log p_c(move | contexts), per candidate: [n_players]."""
        out = np.zeros(len(self.players))
        for moves in games:
            for fctx, bctx, mv in self._tokens(moves):
                pb = self._p_bg(bctx, mv)
                for i in range(len(self.players)):
                    cb = self.black[i].get(bctx)
                    p_black = ((cb[mv] if cb else 0) + self.alpha * pb) / (self.black_total[i].get(bctx, 0) + self.alpha)
                    cf = self.full[i].get(fctx)
                    p_full = ((cf[mv] if cf else 0) + self.alpha_full * p_black) / (self.full_total[i].get(fctx, 0) + self.alpha_full)
                    out[i] += np.log(p_full)
        return out

    def state(self):
        return {"depth": self.depth, "alpha": self.alpha, "beta": self.beta, "alpha_full": self.alpha_full, "symmetry": self.symmetry, "players": list(self.players)}
