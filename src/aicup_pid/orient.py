"""Raw board orientation as a per-candidate habit (fork plan line B6).

Every other component is invariant under the 8 board symmetries (the opening model canonicalises, local tokens and
style features are symmetric), so the corner a player actually opens in -- and whether a 3-4 point is recorded as
3-4 or 4-3 -- is thrown away. On the 11/04 official candidates that raw choice is strongly personal: the raw first Black
move carries 1.8 bits per game about the player (label-shuffle null 0.17; log2(400) = 8.64), the symmetry index that
aicup_pid.opening.canonical picks for the first 12 moves 1.1 bits (null 0.05), and 262 / 400 players open on the same
raw point in >= 90 % of their games (METHOD_NOTES 2.14). Whether the habit is the player's or their client's does not
matter for identification as long as the questions come from the same records, which the first-move corner
distribution of the question games (80 % top-right, as the candidates') says they do.

OrientModel is aicup_pid.local.LocalModel with two tokens per game: ("sym", k) -- the symmetry index canonical() picks
for the first `depth` moves, i.e. the orientation habit net of the opening pattern the opening model already scores --
and ("first", raw coordinates of Black's first move). Same shrinkage towards the pool frequencies (alpha, beta).
"""
from aicup_pid.local import LocalModel
from aicup_pid.opening import PASS, canonical, coords


def orient_tokens(moves, depth=12, n_raw=1):
    """Tokens of a parsed game: ("sym", symmetry index), ("first", (x, y)) and, for n_raw >= 2, ("raw", i, (x, y)) for
    Black's 2nd..n_raw-th non-pass moves plus ("pair", first, second); [] when Black never moved. On the official
    candidates Black's raw 1st and 2nd moves jointly carry 2.5 bits per game about the player against 1.7 for the 1st
    alone (METHOD_NOTES 2.16)."""
    black = [c for (col, *_), c in zip(moves, coords(moves)) if col == "B" and c != PASS]
    if not black:
        return []
    _, k = canonical(moves, depth)
    out = [("sym", int(k)), ("first", tuple(black[0]))]
    for i in range(1, min(n_raw, len(black))):
        out.append(("raw", i, tuple(black[i])))
    if n_raw >= 2 and len(black) >= 2:
        out.append(("pair", tuple(black[0]), tuple(black[1])))
    return out


class OrientModel(LocalModel):
    """fit({player_id: [move lists]}); log_scores([move lists]) -> [n_players] in self.players' order."""

    def __init__(self, alpha=2.0, beta=5.0, depth=12, n_raw=1):
        super().__init__(alpha=alpha, beta=beta, start=0)
        if depth < 1 or n_raw < 1:
            raise ValueError("depth >= 1, n_raw >= 1")
        self.depth, self.n_raw = int(depth), int(n_raw)

    def tokenize(self, game):
        return orient_tokens(game, self.depth, self.n_raw)

    def state(self):
        return {"alpha": self.alpha, "beta": self.beta, "depth": self.depth, "n_raw": self.n_raw, "players": list(self.players), "vocabulary": self.vocabulary}
