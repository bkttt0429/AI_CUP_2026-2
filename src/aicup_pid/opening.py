"""Opening language model: a candidate's habits in the first moves (docs/ARCHITECTURE.md ladder (1)).

The target colour is Black by default (the player-identification task gives Black games only); color="W" models the
same player's White games (AI_CUP_2026-1 issue #22, where the target plays either colour). Every game is put into a
canonical orientation: of the 8 board symmetries the one whose transformed target-colour coordinates among the first
`depth` moves are lexicographically smallest (ties broken by all the moves), so a game and its mirror image give the
same tokens and the orientation does not depend on the opponent's choices unless the target's own moves leave it open.
The model is a per-candidate back-off model over the target's moves among the first `depth` moves, with two contexts:
the full prefix (every move before it) and the target's own prefix (only the target's moves before it):

  p_c(mv | full) = (n_c(full, mv) + a_full * p_c(mv | own)) / (n_c(full) + a_full)
  p_c(mv | own) = (n_c(own, mv) + alpha * p_bg(mv | own)) / (n_c(own) + alpha)
  p_bg(mv | own) = (n_bg(own, mv) + beta / 362) / (n_bg(own) + beta)        (pool-wide background)

The full prefix captures replies to the opponent's moves when they repeat; the own prefix captures the repertoire
regardless of the opponent; the background keeps every probability finite. log_scores(query games) sums log p_c over
the games and their target-colour opening moves, one value per candidate.
"""
import re
from collections import Counter, defaultdict

import numpy as np

from aicup.sgf import parse_moves

_MOVE_RE = re.compile(r";([BW])\[([a-s]{2}|tt)?\]")  # as aicup.sgf, for parse_prefix

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


def canonical(moves, depth, color="B"):
    """(canonical coordinates of the whole game, symmetry index): the symmetry whose `color` coordinates among the first
    `depth` moves are lexicographically smallest, ties broken by all of the first `depth` coordinates."""
    cs = coords(moves)
    head = cs[:depth]
    own = [c for (col, *_), c in zip(moves[:depth], head) if col == color]
    best = min(range(8), key=lambda k: ([_sym(k, x, y) for x, y in own], [_sym(k, x, y) for x, y in head]))
    return [_sym(best, x, y) for x, y in cs], best


def tokens(moves, depth, color="B"):
    """[(full context, own-colour context, move)] for `color`'s moves among the first `depth` moves, canonical coordinates."""
    cs, _ = canonical(moves, depth, color)
    out, own_prefix = [], []
    for i, (c, *_rest) in enumerate(moves[:depth]):
        if c == color:
            out.append((tuple(cs[:i]), tuple(own_prefix), cs[i]))
            own_prefix.append(cs[i])
    return out


def parse(sgf):
    return parse_moves(sgf)


def parse_prefix(sgf, n):
    """The first n moves of an SGF as parse() returns them (the opening model never reads further)."""
    moves = []
    for m in _MOVE_RE.finditer(sgf):
        color, coord = m.group(1), m.group(2)
        moves.append((color, None, None) if coord is None or coord == "tt" else (color, ord(coord[0]) - 97, ord(coord[1]) - 97))
        if len(moves) >= n:
            break
    return moves


class OpeningModel:
    """fit({player_id: [move lists]}); log_scores([move lists]) -> [n_players] in self.players' order."""

    def __init__(self, depth=12, alpha=2.0, beta=5.0, alpha_full=1.0, color="B"):
        if depth < 1 or alpha <= 0 or beta <= 0 or alpha_full <= 0:
            raise ValueError("depth >= 1, alpha > 0, beta > 0 and alpha_full > 0")
        if color not in ("B", "W"):
            raise ValueError("color is 'B' or 'W'")
        self.depth, self.alpha, self.beta, self.alpha_full, self.color = int(depth), float(alpha), float(beta), float(alpha_full), color

    def fit(self, games_by_player):
        self.players = sorted(games_by_player)
        self.index = {p: i for i, p in enumerate(self.players)}
        self.full, self.black = [], []  # per player: {context: Counter(move)}; `black` is the own-colour context (the name predates color=)
        self.bg = defaultdict(Counter)
        for p in self.players:
            full, black = defaultdict(Counter), defaultdict(Counter)
            for moves in games_by_player[p]:
                for fctx, bctx, mv in tokens(moves, self.depth, self.color):
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
        """Sum over the games' target-colour opening moves of log p_c(move | contexts), per candidate: [n_players]."""
        out = np.zeros(len(self.players))
        for moves in games:
            for fctx, bctx, mv in tokens(moves, self.depth, self.color):
                pb = self._p_bg(bctx, mv)
                for i in range(len(self.players)):
                    cb = self.black[i].get(bctx)
                    p_black = ((cb[mv] if cb else 0) + self.alpha * pb) / (self.black_total[i].get(bctx, 0) + self.alpha)
                    cf = self.full[i].get(fctx)
                    p_full = ((cf[mv] if cf else 0) + self.alpha_full * p_black) / (self.full_total[i].get(fctx, 0) + self.alpha_full)
                    out[i] += np.log(p_full)
        return out

    def state(self):
        return {"depth": self.depth, "alpha": self.alpha, "beta": self.beta, "alpha_full": self.alpha_full, "color": self.color, "players": list(self.players)}
