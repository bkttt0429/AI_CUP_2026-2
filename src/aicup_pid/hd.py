"""Handicap flag as a per-candidate habit (fork plan line B5): a Dirichlet-multinomial over the record's HD class.

The official records keep `HD[n]` and nothing else besides the moves; on the 11/04 files HD is 0 or 1 only and the
per-candidate HD[1] rate is strongly player-specific (sd 0.22 across players against 0.037 for a binomial at the pool
rate; METHOD_NOTES 2.12). hd_class maps a record to 0 (HD 0 or absent), 1 (HD 1) or 2 (HD >= 2, training set only).
HDModel: p_c(k) = (n_c(k) + alpha p_bg(k)) / (n_c + alpha) with p_bg(k) = (n_bg(k) + beta / K) / (n_bg + beta), the same
shrinkage form as aicup_pid.local; log_scores sums log p_c over the query games' classes, one value per candidate.
"""
import numpy as np

from aicup.sgf import handicap

K = 3  # HD classes: 0, 1, >= 2


def hd_class(sgf):
    """0 for HD[0] or no HD property, 1 for HD[1], 2 for HD[n >= 2]."""
    h = handicap(sgf)
    return 0 if not h else 1 if h == 1 else 2


class HDModel:
    """fit({player_id: [sgf strings]}); log_scores([sgf strings]) -> [n_players] in self.players' order."""

    def __init__(self, alpha=2.0, beta=5.0):
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha > 0, beta > 0")
        self.alpha, self.beta = float(alpha), float(beta)

    def fit(self, sgfs_by_player):
        self.players = sorted(sgfs_by_player)
        counts = np.zeros((len(self.players), K))
        for i, p in enumerate(self.players):
            for s in sgfs_by_player[p]:
                counts[i, hd_class(s)] += 1
        bg = counts.sum(axis=0)
        p_bg = (bg + self.beta / K) / (bg.sum() + self.beta)
        self.logp = np.log((counts + self.alpha * p_bg) / (counts.sum(axis=1, keepdims=True) + self.alpha))
        self.class_counts = bg.astype(int).tolist()
        return self

    def _counts(self, sgfs):
        v = np.zeros(K)
        for s in sgfs:
            v[hd_class(s)] += 1
        return v

    def log_scores(self, sgfs):
        return self.logp @ self._counts(sgfs)

    def state(self):
        return {"alpha": self.alpha, "beta": self.beta, "players": list(self.players), "class_counts": self.class_counts}
