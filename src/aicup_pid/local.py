"""Local-shape habits over the whole game (fork plan line B2): a per-candidate multinomial over per-move tokens.

Every Black move (from ply `start` on; passes give no token) becomes (displacement class to White's previous move,
displacement class to Black's own previous move, line, phase). A displacement class is (max(|dx|,|dy|), min(|dx|,|dy|))
clipped at 3, FAR when either exceeds 3, NONE when there is no such previous move -- all invariant under the 8 board
symmetries. line is the distance to the nearest edge (1..5, 5 = centre), phase 0 / 1 / 2 for ply < 60 / < 150 / rest.
LocalModel: p_c(tok) = (n_c(tok) + alpha p_bg(tok)) / (n_c + alpha), p_bg(tok) = (n_bg(tok) + beta / V) / (n_bg + beta) with
V = vocabulary + 1 (one slot for unseen tokens); log_scores sums log p_c over the query games' tokens, one value per
candidate (a [C, V] table times the query's token counts).
"""
from collections import Counter

import numpy as np

BOARD = 19
FAR = (4, 4)
NONE = (-1, -1)
MAX_D = 3
PHASES = (60, 150)


def displacement(a, b):
    """Class of the displacement from move a to move b ((x, y) tuples; b None -> NONE)."""
    if b is None or a is None:
        return NONE
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    hi, lo = max(dx, dy), min(dx, dy)
    if hi > MAX_D:
        return FAR
    return (hi, lo)


def line_of(x, y):
    return min(x, y, BOARD - 1 - x, BOARD - 1 - y, 4) + 1


def phase_of(t):
    return 0 if t < PHASES[0] else 1 if t < PHASES[1] else 2


def tokens(moves, start=12):
    """[(to White's last move, to Black's last move, line, phase)] for Black's non-pass moves at ply >= start."""
    out, last = [], {"B": None, "W": None}
    for t, (c, x, y) in enumerate(moves):
        if x is None:
            continue
        if c == "B" and t >= start:
            out.append((displacement((x, y), last["W"]), displacement((x, y), last["B"]), line_of(x, y), phase_of(t)))
        last[c] = (x, y)
    return out


class LocalModel:
    """fit({player_id: [move lists]}); log_scores([move lists]) -> [n_players] in self.players' order."""

    def __init__(self, alpha=2.0, beta=5.0, start=12):
        if alpha <= 0 or beta <= 0 or start < 0:
            raise ValueError("alpha > 0, beta > 0, start >= 0")
        self.alpha, self.beta, self.start = float(alpha), float(beta), int(start)

    def tokenize(self, game):
        """The tokens of one parsed game (subclasses change the vocabulary, e.g. aicup_pid.orient)."""
        return tokens(game, self.start)

    def fit(self, games_by_player):
        self.players = sorted(games_by_player)
        counts = [Counter(tok for g in games_by_player[p] for tok in self.tokenize(g)) for p in self.players]
        bg = Counter()
        for c in counts:
            bg.update(c)
        self.vocab = {tok: i for i, tok in enumerate(sorted(bg))}
        self.vocabulary = len(self.vocab)
        V = self.vocabulary + 1  # the last column is every unseen token
        n_bg = sum(bg.values())
        p_bg = np.full(V, self.beta / V)
        for tok, n in bg.items():
            p_bg[self.vocab[tok]] += n
        p_bg /= n_bg + self.beta
        table = np.empty((len(self.players), V))
        for i, c in enumerate(counts):
            row = self.alpha * p_bg
            for tok, n in c.items():
                row[self.vocab[tok]] += n
            table[i] = np.log(row / (sum(c.values()) + self.alpha))
        self.logp = table
        return self

    def _counts(self, games):
        v = np.zeros(self.vocabulary + 1)
        for g in games:
            for tok in self.tokenize(g):
                v[self.vocab.get(tok, self.vocabulary)] += 1
        return v

    def log_scores(self, games):
        return self.logp @ self._counts(games)

    def state(self):
        return {"alpha": self.alpha, "beta": self.beta, "start": self.start, "vocabulary": self.vocabulary, "players": list(self.players)}
