"""Whole-game duplicate detection (fork plan section 3, checks 1-2).

game_hash puts a game into the canonical orientation of aicup_pid.opening.canonical over all of its moves (so a game and
its 8 symmetric images hash alike), encodes every move as 3 bytes (colour B=0 / W=1, then the point x*19+y big-endian with
pass = 380) and takes a 16-byte blake2b. build_training_index streams the 10 train_<rank>.csv files in chunks, hashes
them in a fork pool and keeps, per game, (player_id, rank, game_key, color, n_moves, n_black) plus hash -> rows.
"""
import hashlib
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aicup.metric import LEVELS
from aicup.sgf import parse_moves
from aicup_pid.opening import BOARD, canonical

PASS_CODE = BOARD * BOARD + BOARD  # 380: (19, 19) under every symmetry
COLUMNS = ["player_id", "rank", "game_key", "color", "n_moves", "n_black"]
USECOLS = ["player_id", "game_id", "rank", "color", "sgf_content"]


def game_hash(moves):
    """16 bytes identifying the game up to the 8 board symmetries (colours kept)."""
    if not moves:
        return hashlib.blake2b(b"", digest_size=16).digest()
    cs, _ = canonical(moves, len(moves))
    buf = bytearray()
    for (c, *_), (x, y) in zip(moves, cs):
        code = PASS_CODE if x == BOARD else x * BOARD + y
        buf += bytes((0 if c == "B" else 1, code >> 8, code & 255))
    return hashlib.blake2b(bytes(buf), digest_size=16).digest()


def _hash_one(sgf):
    moves = parse_moves(sgf)
    if not moves:
        return None, 0, 0
    return game_hash(moves), len(moves), sum(1 for c, *_ in moves if c == "B")


def hash_games(sgfs, workers=1):
    """{key: (hash or None, n_moves, n_black)}; a record without moves gives (None, 0, 0)."""
    keys = list(sgfs)
    texts = [sgfs[k] for k in keys]
    if workers > 1 and len(texts) > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_hash_one, texts, chunksize=256)
    else:
        res = [_hash_one(s) for s in texts]
    return dict(zip(keys, res))


class TrainingIndex:
    """meta: DataFrame with COLUMNS, one row per training game; by_hash: {hash: [row numbers]}."""

    def __init__(self, meta, by_hash, seconds=0.0):
        self.meta, self.by_hash, self.seconds = meta, by_hash, seconds

    def lookup(self, h):
        return list(self.by_hash.get(h, ()))

    def black_lengths(self):
        """(n_moves, n_black) arrays of the Black games."""
        m = self.meta[self.meta["color"] == "B"]
        return m["n_moves"].to_numpy(dtype=int), m["n_black"].to_numpy(dtype=int)


def build_training_index(train_dir, workers=1, chunksize=20000, max_games=None, log=print):
    """Hash every game of train_<rank>.csv (LEVELS order, files that exist) in chunks; max_games stops early (smoke tests)."""
    t0 = time.time()
    pool = mp.get_context("fork").Pool(workers) if workers > 1 else None
    frames, by_hash, n = [], {}, 0
    try:
        for r in LEVELS:
            path = Path(train_dir) / f"train_{r}.csv"
            if not path.exists():
                continue
            t1, n_file = time.time(), 0
            for chunk in pd.read_csv(path, usecols=USECOLS, dtype=str, keep_default_na=False, chunksize=chunksize):
                if max_games is not None:
                    chunk = chunk.iloc[:max(max_games - n, 0)]
                    if len(chunk) == 0:
                        break
                texts = chunk["sgf_content"].tolist()
                res = pool.map(_hash_one, texts, chunksize=256) if pool else [_hash_one(s) for s in texts]
                frames.append(pd.DataFrame({"player_id": chunk["player_id"].to_numpy(), "rank": chunk["rank"].to_numpy(),
                                            "game_key": (chunk["rank"] + ":" + chunk["game_id"]).to_numpy(), "color": chunk["color"].to_numpy(),
                                            "n_moves": np.array([m for _, m, _ in res], dtype=np.int32),
                                            "n_black": np.array([b for _, _, b in res], dtype=np.int32)}))
                for i, (h, _, _) in enumerate(res):
                    if h is not None:
                        by_hash.setdefault(h, []).append(n + i)
                n += len(chunk)
                n_file += len(chunk)
                if max_games is not None and n >= max_games:
                    break
            log(f"training index: {path.name} {n_file} games in {time.time() - t1:.0f} s ({n} so far)", flush=True)
            if max_games is not None and n >= max_games:
                break
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    meta = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return TrainingIndex(meta[COLUMNS], by_hash, time.time() - t0)
