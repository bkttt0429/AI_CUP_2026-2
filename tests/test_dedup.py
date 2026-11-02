import hashlib

import numpy as np

from aicup.metric import LEVELS
from aicup.sgf import parse_moves
from aicup_pid import opening
from aicup_pid.dedup import build_training_index, game_hash, hash_games
from tests.synth import sgf_of, training_frame


def _apply_symmetry(moves, k):
    return [(c, *opening._sym(k, x, y)) if x is not None else (c, x, y) for c, x, y in moves]


def _passless_sgf(moves):
    return sgf_of([(c, x, y) for c, x, y in moves if x is not None])


def test_hash_symmetry_and_colour():
    moves = parse_moves(sgf_of([("B", 15, 3), ("W", 3, 15), ("B", 16, 16), ("W", 2, 2)]) .replace(")", ";B[];W[jj])"))
    assert moves[4] == ("B", None, None) and len(moves) == 6
    hs = {game_hash(_apply_symmetry(moves, k)) for k in range(8)}
    assert len(hs) == 1 and len(hs.pop()) == 16
    swapped = [("W" if c == "B" else "B", x, y) for c, x, y in moves]
    assert game_hash(swapped) != game_hash(moves)
    assert game_hash(moves[:-1]) != game_hash(moves)
    assert game_hash([]) == hashlib.blake2b(b"", digest_size=16).digest()


def test_hash_games_counts_and_tolerates_garbage():
    out = hash_games({"a": sgf_of([("B", 3, 3), ("W", 15, 15), ("B", 3, 15)]), "b": "(;SZ[19])", "c": "not an sgf"})
    assert out["a"][1:] == (3, 2) and len(out["a"][0]) == 16
    assert out["b"] == (None, 0, 0) and out["c"] == (None, 0, 0)
    assert hash_games({"a": sgf_of([("B", 3, 3), ("W", 15, 15), ("B", 3, 15)])}, workers=2)["a"] == out["a"]


def _write_training(frame, train_dir):
    train_dir.mkdir(parents=True, exist_ok=True)
    for r in LEVELS:
        sub = frame[frame["rank"] == r]
        if len(sub):
            sub[["player_id", "game_id", "rank", "color", "sgf_content"]].to_csv(train_dir / f"train_{r}.csv", index=False)


def test_index_streams_chunks(tmp_path):
    frame, _ = training_frame(folds=(1, 2), players_per_fold=3, games_per_player=10, seed=2)
    _write_training(frame, tmp_path)
    idx = build_training_index(tmp_path, workers=1, chunksize=7, log=lambda *a, **k: None)
    assert len(idx.meta) == len(frame) and set(idx.meta["rank"]) == set(frame["rank"])
    assert list(idx.meta.columns) == ["player_id", "rank", "game_key", "color", "n_moves", "n_black"]
    some = frame.iloc[5]
    rows = idx.lookup(game_hash(parse_moves(some["sgf_content"])))
    assert [idx.meta.iloc[i]["game_key"] for i in rows] == [f"{some['rank']}:{some['game_id']}"]
    assert idx.lookup(b"\0" * 16) == []
    n_moves, n_black = idx.black_lengths()
    assert len(n_moves) == (frame["color"] == "B").sum() and (n_black * 2 <= n_moves + 1).all() and n_moves.min() >= 40
    small = build_training_index(tmp_path, workers=1, chunksize=7, max_games=4, log=lambda *a, **k: None)
    assert len(small.meta) == 4
    assert len(build_training_index(tmp_path, workers=2, chunksize=7, log=lambda *a, **k: None).meta) == len(frame)
