"""Synthetic training data with planted per-player habits for the player-identification tests."""
import numpy as np
import pandas as pd

from aicup.metric import LEVELS

LETTERS = "abcdefghijklmnopqrs"
CORNERS = {"44": (15, 3), "34": (15, 2), "33": (16, 2), "35": (14, 2)}  # upper-right points of each type (0-indexed x, y)


def sgf_of(moves):
    return "(;SZ[19]" + "".join(f";{c}[{LETTERS[x]}{LETTERS[y]}]" for c, x, y in moves) + ")"


def random_game(rng, habit, n_moves=60):
    """A game whose Black opening follows `habit` ({first: corner type, second: corner type, line: preferred line 3 or 4})
    and is otherwise random on empty points; White is random. Returns the SGF."""
    occupied, moves = set(), []
    k = int(rng.randint(8))  # a random board symmetry of the planted habit

    def sym(x, y):
        n = 18
        if k & 1:
            x = n - x
        if k & 2:
            y = n - y
        if k & 4:
            x, y = y, x
        return x, y

    def place(c, x, y):
        occupied.add((x, y))
        moves.append((c, x, y))

    def random_point(line=None):
        while True:
            if line is not None and rng.rand() < 0.8:
                side = rng.randint(4)
                a = line - 1
                b = int(rng.randint(a, 19 - a))
                x, y = [(a, b), (18 - a, b), (b, a), (b, 18 - a)][side]
            else:
                x, y = int(rng.randint(19)), int(rng.randint(19))
            if (x, y) not in occupied:
                return x, y

    first, second = CORNERS[habit["first"]], CORNERS[habit["second"]]
    second = (18 - second[0], 18 - second[1])  # the opposite corner
    for i in range(n_moves):
        if i % 2 == 0:
            if i == 0:
                place("B", *sym(*first))
            elif i == 2 and (18 - second[0], 18 - second[1]) in CORNERS.values():
                x, y = sym(*second)
                place("B", x, y) if (x, y) not in occupied else place("B", *random_point())
            else:
                place("B", *random_point(habit["line"] if i < 30 else None))
        else:
            place("W", *random_point())
    return sgf_of(moves)


def habit_of(player_index):
    types = list(CORNERS)
    return {"first": types[player_index % 4], "second": types[(player_index // 4) % 4], "line": 3 if (player_index // 16) % 2 == 0 else 4}


def training_frame(folds=(1, 2, 3, 4), players_per_fold=6, games_per_player=16, seed=0):
    """(frame with player_id, game_id, rank, color, sgf_content, fold; folds table). Half the games are Black."""
    rng = np.random.RandomState(seed)
    rows, fold_rows, g = [], [], 0
    idx = 0
    for fold in folds:
        for p in range(players_per_fold):
            pid = f"{fold:02x}{p:06x}"
            rank = LEVELS[idx % len(LEVELS)]
            habit = habit_of(idx)
            fold_rows.append({"rank": rank, "player_id": pid, "n_games": games_per_player, "fold": fold})
            for j in range(games_per_player):
                colour = "B" if j % 2 == 0 else "W"
                g += 1
                rows.append({"player_id": pid, "game_id": f"g_{g:07d}", "rank": rank, "color": colour,
                             "sgf_content": random_game(rng, habit, n_moves=40 + int(rng.randint(40))), "fold": fold})
            idx += 1
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def black_table(frame):
    df = frame[frame["color"] == "B"].copy()
    df["game_key"] = df["rank"] + ":" + df["game_id"]
    return df[["player_id", "rank", "game_key"]].reset_index(drop=True)


def sgf_lookup(frame):
    keys = frame["rank"] + ":" + frame["game_id"]
    return dict(zip(keys, zip(frame["sgf_content"], frame["color"])))


def official_files(tmp_path, frame, prefix, n_q=3):
    """Candidates (all Black games of each player but the last four) and questions (those four, 2 or 4 games) in the official format."""
    black = frame[frame["color"] == "B"]
    cand_rows, q_rows = [], []
    for i, (pid, g) in enumerate(black.groupby("player_id")):
        games = g["sgf_content"].tolist()
        for s in games[:-4]:
            cand_rows.append({"game_id": f"g_{len(cand_rows) + 1:07d}", "player_id": f"{prefix}p_{pid}", "sgf_content": s, "color": "B"})
        if i < n_q:
            qg = games[-4:] if i % 2 else games[-2:]
            row = {"question_id": f"{prefix}q_{i + 1:04d}", "num_games": len(qg)}
            for j in range(20):
                row[f"sgf_{j + 1}"] = qg[j] if j < len(qg) else ""
                row[f"color_{j + 1}"] = "B" if j < len(qg) else ""
            q_rows.append(row)
    cp, qp = tmp_path / f"cands_{prefix}csv", tmp_path / f"test_{prefix}csv"
    pd.DataFrame(cand_rows).to_csv(cp, index=False)
    pd.DataFrame(q_rows).to_csv(qp, index=False)
    return cp, qp, [f"{prefix}p_{pid}" for pid in black["player_id"].unique()]
