"""Raw style statistics of a game's Black moves and a Gaussian candidate model (docs/records/ARCHITECTURE.md ladder (2a)).

game_features replays the game on a small board (captures handled) and summarises Black's moves: lines, distances to the
previous Black move and to White's last move, contact moves, captures, the first two Black moves' point types, length.
StyleModel fits one mean per candidate and a pooled within-candidate covariance (shrunk toward its diagonal); the log
score of a query (mean of its games' features) under candidate c is -1/2 d' S^-1 d with d = q - mu_c and S = Sigma x (1/m +
1/n_c): an LDA-style score, one value per candidate.
"""
import numpy as np

BOARD = 19
LINES = (1, 2, 3, 4)  # 5+ is the last bin
POINT_TYPES = ("44", "34", "33", "35", "45", "other")
FEATURE_NAMES = (["log_moves", "black_moves", *[f"line_{k}" for k in (*LINES, "5plus")],
                  *[f"first_{t}" for t in POINT_TYPES], *[f"second_{t}" for t in POINT_TYPES],
                  "prev_le2", "prev_3to5", "prev_ge6", "prev_mean", "opp_le1", "opp_2", "opp_3to5", "opp_ge6", "contact",
                  "captures_made", "stones_lost", "early_line3", "early_line4"])


class Board:
    """Stones with captures (no suicide handling beyond removing a libertyless own group)."""

    def __init__(self):
        self.s = [[0] * BOARD for _ in range(BOARD)]  # 0 empty, 1 Black, 2 White

    def _group(self, x, y):
        colour = self.s[y][x]
        seen, stack, libs = {(x, y)}, [(x, y)], set()
        while stack:
            cx, cy = stack.pop()
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if 0 <= nx < BOARD and 0 <= ny < BOARD:
                    v = self.s[ny][nx]
                    if v == 0:
                        libs.add((nx, ny))
                    elif v == colour and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
        return seen, libs

    def play(self, colour, x, y):
        """Place a stone (1 Black, 2 White); returns the number of opponent stones captured."""
        self.s[y][x] = colour
        other, captured = 3 - colour, 0
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < BOARD and 0 <= ny < BOARD and self.s[ny][nx] == other:
                group, libs = self._group(nx, ny)
                if not libs:
                    for gx, gy in group:
                        self.s[gy][gx] = 0
                    captured += len(group)
        if captured == 0:
            group, libs = self._group(x, y)
            if not libs:  # suicide: not legal in the data, but keep the board consistent
                for gx, gy in group:
                    self.s[gy][gx] = 0
        return captured

    def adjacent_to(self, x, y, colour):
        return any(0 <= nx < BOARD and 0 <= ny < BOARD and self.s[ny][nx] == colour
                   for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))


def line_of(x, y):
    return min(x, BOARD - 1 - x, y, BOARD - 1 - y) + 1


def point_type(x, y):
    a, b = sorted((min(x, BOARD - 1 - x), min(y, BOARD - 1 - y)))
    return {(3, 3): "44", (2, 3): "34", (2, 2): "33", (2, 4): "35", (3, 4): "45"}.get((a, b), "other")


def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def game_features(moves, early=30):
    """[len(FEATURE_NAMES)] floats of a parsed game's Black moves (zeros where Black has no moves)."""
    f = dict.fromkeys(FEATURE_NAMES, 0.0)
    board = Board()
    n_black, prev_black, last_white = 0, None, None
    lines, prev_d, opp_d, early_lines = [], [], [], []
    contact = captures = lost = 0
    firsts = []
    for i, (c, x, y) in enumerate(moves):
        if x is None:
            continue
        colour = 1 if c == "B" else 2
        if colour == 1:
            n_black += 1
            lines.append(line_of(x, y))
            if i < early:
                early_lines.append(line_of(x, y))
            if len(firsts) < 2:
                firsts.append(point_type(x, y))
            if prev_black is not None:
                prev_d.append(_cheb((x, y), prev_black))
            if last_white is not None:
                opp_d.append(_cheb((x, y), last_white))
            if board.adjacent_to(x, y, 2):
                contact += 1
            captures += board.play(1, x, y)
            prev_black = (x, y)
        else:
            lost += board.play(2, x, y)
            last_white = (x, y)
    f["log_moves"] = float(np.log1p(len(moves)))
    f["black_moves"] = float(n_black)
    if n_black == 0:
        return np.array([f[k] for k in FEATURE_NAMES])
    lines = np.asarray(lines)
    for k in LINES:
        f[f"line_{k}"] = float((lines == k).mean())
    f["line_5plus"] = float((lines >= 5).mean())
    for j, name in enumerate(("first", "second")):
        if j < len(firsts):
            f[f"{name}_{firsts[j]}"] = 1.0
    if prev_d:
        d = np.asarray(prev_d)
        f["prev_le2"], f["prev_3to5"], f["prev_ge6"], f["prev_mean"] = float((d <= 2).mean()), float(((d >= 3) & (d <= 5)).mean()), float((d >= 6).mean()), float(d.mean())
    if opp_d:
        d = np.asarray(opp_d)
        f["opp_le1"], f["opp_2"], f["opp_3to5"], f["opp_ge6"] = float((d <= 1).mean()), float((d == 2).mean()), float(((d >= 3) & (d <= 5)).mean()), float((d >= 6).mean())
    f["contact"] = contact / n_black
    f["captures_made"], f["stones_lost"] = float(captures), float(lost)
    if early_lines:
        e = np.asarray(early_lines)
        f["early_line3"], f["early_line4"] = float((e == 3).mean()), float((e == 4).mean())
    return np.array([f[k] for k in FEATURE_NAMES])


class StyleModel:
    """fit({player_id: [n_games, d] feature rows}); log_scores([m, d] query rows) -> [n_players] in self.players' order."""

    def __init__(self, shrink=0.1):
        if not 0.0 <= shrink <= 1.0:
            raise ValueError("shrink in [0, 1]")
        self.shrink = float(shrink)

    def fit(self, feats_by_player):
        self.players = sorted(feats_by_player)
        X = [np.asarray(feats_by_player[p], dtype=float) for p in self.players]
        self.n_games = np.array([len(x) for x in X], dtype=float)
        self.means = np.stack([x.mean(axis=0) for x in X])
        resid = np.vstack([x - x.mean(axis=0) for x in X])
        dof = max(len(resid) - len(X), 1)
        cov = resid.T @ resid / dof
        cov = (1 - self.shrink) * cov + self.shrink * np.diag(np.diag(cov))
        cov += np.eye(cov.shape[0]) * 1e-9 * max(np.trace(cov) / cov.shape[0], 1e-12)
        self.cov = cov
        self.prec = np.linalg.inv(cov)
        return self

    def log_scores(self, query_rows):
        q = np.asarray(query_rows, dtype=float)
        m = max(len(q), 1)
        d = self.means - q.mean(axis=0)
        scale = 1.0 / m + 1.0 / self.n_games  # [C]
        quad = np.einsum("cd,de,ce->c", d, self.prec, d) / scale
        return -0.5 * quad

    def state(self):
        return {"shrink": self.shrink, "players": list(self.players), "features": list(FEATURE_NAMES)}
