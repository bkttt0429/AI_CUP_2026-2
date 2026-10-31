"""Candidate pools, questions, the official score and the submission file (player identification).

A pool is 400 candidate players with 100 Black games each; a question is 5 / 10 / 15 / 20 Black games of one of them
(docs/TASK.md). make_pool simulates pools from the training set (docs/ARCHITECTURE.md section 3): players of the given
folds with at least 100 + 5 Black games, each one's Black games shuffled with a seed derived from the player id, the
first 100 as the candidate's games and the rest cut into one question per size in ascending order. read_candidates /
read_questions read the official files; question_score is e^-(r-1) for the true player at Top-5 rank r (0 otherwise);
write_submission / validate_submission produce and check submission_player.csv (question_id, top_1..top_5; 800 rows,
five distinct ids from the question's own pool, public before private).
"""
import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from aicup.metric import LEVELS
from aicup.submission import read_test
from aicup_pid.paths import TRAIN_DIR

SIZES = (5, 10, 15, 20)
N_CANDIDATES, CAND_GAMES, TOP_K = 400, 100, 5
PREFIXES = ("pub_", "priv_")
COLUMNS = ["question_id", *[f"top_{i}" for i in range(1, TOP_K + 1)]]


def black_games(train_dir=TRAIN_DIR):
    """(player_id, rank, game_key) of every Black game of the training set, game_key 'RANK:game_id' as aicup.scoring."""
    frames = []
    for r in LEVELS:
        df = pd.read_csv(Path(train_dir) / f"train_{r}.csv", usecols=["player_id", "game_id", "rank", "color"], dtype=str)
        df = df[df["color"] == "B"].copy()
        df["game_key"] = df["rank"] + ":" + df["game_id"]
        frames.append(df[["player_id", "rank", "game_key"]])
    return pd.concat(frames, ignore_index=True)


def eligible_players(games, folds_df, folds, min_black=CAND_GAMES + min(SIZES)):
    """Players of `folds` with >= min_black Black games in `games`: (player_id, rank, n_black), sorted by rank and id."""
    folds = [int(f) for f in folds]
    pl = folds_df[folds_df["fold"].isin(folds)][["player_id", "rank"]].astype({"player_id": str})
    counts = games.groupby("player_id").size().rename("n_black")
    out = pl.merge(counts, left_on="player_id", right_index=True, how="inner")
    return out[out["n_black"] >= min_black].sort_values(["rank", "player_id"], kind="stable").reset_index(drop=True)


def _player_rng(seed, player_id):
    return np.random.RandomState(int(hashlib.sha256(f"{seed}:{player_id}".encode()).hexdigest()[:8], 16))


def make_pool(games, folds_df, folds, n_players=N_CANDIDATES, cand_games=CAND_GAMES, sizes=SIZES, seed=0, pool_id="sim"):
    """A simulated pool: (candidates: player_id, rank, game_key; questions: question_id, player_id, rank, n_games, game_keys;
    info). Up to n_players eligible players (RandomState(seed)); per player the Black games are shuffled with a seed from
    the player id, the first cand_games are the candidate's, the rest give one question per size (ascending) while games
    remain, disjoint. ValueError when no player is eligible."""
    sizes = sorted(int(m) for m in sizes)
    elig = eligible_players(games, folds_df, folds, cand_games + sizes[0])
    if len(elig) == 0:
        raise ValueError(f"folds {list(folds)}: no player has >= {cand_games + sizes[0]} Black games")
    rng = np.random.RandomState(seed)
    chosen = elig.iloc[np.sort(rng.permutation(len(elig))[:n_players])]
    by_player = games.groupby("player_id")["game_key"].apply(sorted).to_dict()
    cand_rows, q_rows = [], []
    for p in chosen.itertuples():
        keys = by_player[p.player_id]
        keys = [keys[i] for i in _player_rng(seed, p.player_id).permutation(len(keys))]
        cand_rows += [{"player_id": p.player_id, "rank": p.rank, "game_key": k} for k in keys[:cand_games]]
        rest = keys[cand_games:]
        for m in sizes:
            if len(rest) < m:
                break
            q_rows.append({"question_id": f"{pool_id}_q_{len(q_rows) + 1:04d}", "player_id": p.player_id, "rank": p.rank,
                           "n_games": m, "game_keys": rest[:m]})
            rest = rest[m:]
    cands = pd.DataFrame(cand_rows, columns=["player_id", "rank", "game_key"])
    qs = pd.DataFrame(q_rows, columns=["question_id", "player_id", "rank", "n_games", "game_keys"])
    info = {"folds": [int(f) for f in folds], "seed": int(seed), "eligible": int(len(elig)), "players": int(len(chosen)),
            "cand_games": int(cand_games), "questions": int(len(qs)), "questions_by_n": {str(m): int((qs["n_games"] == m).sum()) for m in sizes},
            "players_by_rank": {lv: int((chosen["rank"] == lv).sum()) for lv in LEVELS}}
    return cands, qs, info


def read_candidates(path):
    """The official candidates CSV as (player_id, game_id, sgf, color); ValueError on missing columns or a non-Black game."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ("game_id", "player_id", "sgf_content", "color") if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    bad = df[df["color"].str.upper() != "B"]
    if len(bad):
        raise ValueError(f"{path}: {len(bad)} candidate games are not Black (the pools hold Black games only)")
    return df.rename(columns={"sgf_content": "sgf"})[["player_id", "game_id", "sgf", "color"]].reset_index(drop=True)


def read_questions(path):
    """The official test CSV: [{question_id, declared_games, games: [(sgf, colour), ...]}] (aicup.submission.read_test)."""
    return read_test(path)


def question_score(top, truth):
    """e^-(r-1) when `truth` is the r-th of `top` (1-based), else 0."""
    top = list(top)
    return math.exp(-top.index(truth)) if truth in top else 0.0


def summarise(tops, truth, n_games):
    """Official score and auxiliaries of Top-5 lists: {score, top1, top5, mrr, n, by_n: {size: {...}}}."""
    tops, truth, n_games = list(tops), list(truth), np.asarray(n_games)
    s = np.array([question_score(t, y) for t, y in zip(tops, truth)])
    top1 = np.array([len(t) > 0 and t[0] == y for t, y in zip(tops, truth)], dtype=float)
    hit = np.array([y in t for t, y in zip(tops, truth)], dtype=float)
    rr = np.array([1.0 / (list(t).index(y) + 1) if y in t else 0.0 for t, y in zip(tops, truth)])

    def block(m):
        return {"score": float(s[m].mean()), "top1": float(top1[m].mean()), "top5": float(hit[m].mean()), "mrr": float(rr[m].mean()), "n": int(m.sum())}

    out = block(np.ones(len(s), dtype=bool))
    out["by_n"] = {str(m): block(n_games == m) for m in sorted(set(n_games.tolist()))}
    return out


def write_submission(rows, path):
    """rows: [(question_id, [top_1..top_5])] in the inputs' order -> CSV with COLUMNS."""
    table = pd.DataFrame([{"question_id": q, **{f"top_{i + 1}": p for i, p in enumerate(top)}} for q, top in rows], columns=COLUMNS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def validate_submission(path, question_ids, pool_players, expect_rows=None):
    """Check a submission against the questions (ids in input order, public then private) and the pools' player ids
    ({prefix: set of ids}): header, every id once in order, no priv_ before a pub_, five distinct predictions all from the
    question's own pool, and the expected row count. Returns the list of problems ([] when valid)."""
    problems = []
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != COLUMNS:
        problems.append(f"header {list(df.columns)} != {COLUMNS}")
        return problems
    ids = df["question_id"].tolist()
    if ids != list(question_ids):
        counts = Counter(ids)
        dup = [q for q, c in counts.items() if c > 1]
        missing = [q for q in question_ids if q not in counts]
        extra = [q for q in counts if q not in set(question_ids)]
        problems.append(f"question ids differ from the inputs' order: {len(dup)} duplicated, {len(missing)} missing, {len(extra)} unknown")
    seen_priv = False
    for q in ids:
        if q.startswith("priv_"):
            seen_priv = True
        elif q.startswith("pub_") and seen_priv:
            problems.append(f"{q}: a public question after a private one")
            break
    for r in df.itertuples(index=False):
        top = [getattr(r, f"top_{i}") for i in range(1, TOP_K + 1)]
        prefix = next((p for p in PREFIXES if r.question_id.startswith(p)), None)
        if len(set(top)) != TOP_K or any(not t for t in top):
            problems.append(f"{r.question_id}: the five predictions are not distinct non-empty ids")
        if prefix is None:
            problems.append(f"{r.question_id}: no pub_/priv_ prefix")
        elif prefix in pool_players:
            outside = [t for t in top if t not in pool_players[prefix]]
            if outside:
                problems.append(f"{r.question_id}: {len(outside)} predictions are not in the {prefix} pool (e.g. {outside[0]!r})")
    if expect_rows is not None and len(df) != expect_rows:
        problems.append(f"{len(df)} rows, expected {expect_rows}")
    return problems
