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
import hashlib

import numpy as np
import scipy.sparse as sp

from aicup_pid.local import LocalModel
from aicup_pid.orient import OrientModel


def reference_players(games, folds_df, folds, min_black=30, cap=40, seed=0, exclude=()):
    """{player_id: [game_key, ...]} for the players of `folds` with >= min_black Black games (not in `exclude`), each with
    up to `cap` games drawn by a per-player seeded shuffle."""
    folds = [int(f) for f in folds]
    pl = set(folds_df[folds_df["fold"].isin(folds)]["player_id"].astype(str)) - set(exclude)
    out = {}
    for p, grp in games[games["player_id"].isin(pl)].groupby("player_id"):
        keys = sorted(grp["game_key"].tolist())
        if len(keys) < min_black:
            continue
        rng = np.random.RandomState(int(hashlib.sha256(f"{seed}:{p}".encode()).hexdigest()[:8], 16))
        rng.shuffle(keys)
        out[str(p)] = keys[:cap]
    return out


class RefModel:
    """fit({reference player: [move lists]}); vectors([move lists]) -> [G, R] z-scored log-likelihoods; profile(...).
    orient_raw=0 drops the orientation block (local-shape tokens only): every token is then invariant under the 8 board
    symmetries, for files whose games may be rotated / mirrored."""

    def __init__(self, alpha=2.0, beta=5.0, local_start=12, orient_depth=12, orient_raw=2):
        if orient_raw < 0:
            raise ValueError("orient_raw must be >= 0")
        self.local = LocalModel(alpha=alpha, beta=beta, start=local_start)
        self.orient = OrientModel(alpha=alpha, beta=beta, depth=orient_depth, n_raw=orient_raw) if orient_raw else None
        self.blocks = (self.local,) + ((self.orient,) if self.orient else ())

    def fit(self, games_by_player):
        for model in self.blocks:
            model.fit(games_by_player)
        self.players = self.local.players
        self.logp = np.hstack([m.logp for m in self.blocks]).astype(np.float32)  # [R, V_local (+ V_orient)]
        return self

    def _counts(self, games):
        """Sparse [G, V_local + V_orient] token counts."""
        rows, cols, vals = [], [], []
        shifts = [0, self.local.vocabulary + 1][:len(self.blocks)]
        for g, moves in enumerate(games):
            for model, shift in zip(self.blocks, shifts):
                for tok in model.tokenize(moves):
                    rows.append(g)
                    cols.append(shift + model.vocab.get(tok, model.vocabulary))
                    vals.append(1.0)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(games), self.logp.shape[1]), dtype=np.float32)

    def vectors(self, games):
        """[G, R] per-game log-likelihoods under every reference, z-scored across the references."""
        ll = np.asarray(self._counts(games) @ self.logp.T, dtype=np.float64)
        ll -= ll.mean(axis=1, keepdims=True)
        sd = ll.std(axis=1, keepdims=True)
        sd[sd < 1e-12] = 1.0
        return ll / sd

    def profile(self, games):
        """[R] mean z-scored vector of a set of games (zeros for an empty set)."""
        if not games:
            return np.zeros(len(self.players))
        return self.vectors(games).mean(axis=0)

    def state(self):
        return {"references": len(self.players), "local": self.local.state(), "orient": self.orient.state() if self.orient else None}


def cosine_scores(question_profiles, candidate_profiles):
    """[Q, C] cosine similarity between question and candidate profiles."""
    Q = np.asarray(question_profiles, dtype=np.float64)
    C = np.asarray(candidate_profiles, dtype=np.float64)
    qn = np.linalg.norm(Q, axis=1, keepdims=True)
    cn = np.linalg.norm(C, axis=1, keepdims=True)
    qn[qn < 1e-12] = 1.0
    cn[cn < 1e-12] = 1.0
    return (Q / qn) @ (C / cn).T
