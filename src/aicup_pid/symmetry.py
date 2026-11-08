"""Board symmetries of SGF records, for simulating re-released official files whose games are rotated / mirrored.

transform_sgf(sgf, k) applies the k-th of the 8 symmetries of the 19x19 board to every move and setup stone
(B, W, AB, AW, AE values; passes `[]` / `[tt]` are kept): bit 0 mirrors x, bit 1 mirrors y, bit 2 swaps x and y, so
k = 0 is the identity. sym_of(key, seed) picks one symmetry per game from a hash of (seed, key), the same in every
script (CV pools, trunk extraction), and uniform over the 8 (the identity included, as an organiser's random choice would).
"""
import hashlib
import re

from aicup_pid.hd import hd_class

BOARD = 19
LETTERS = "abcdefghijklmnopqrs"
_PROP = re.compile(r"(?<![A-Za-z])(AB|AW|AE|B|W)((?:\[[^\]]*\])+)")
_VAL = re.compile(r"\[([^\]]*)\]")


def transform_point(x, y, k):
    if k & 1:
        x = BOARD - 1 - x
    if k & 2:
        y = BOARD - 1 - y
    if k & 4:
        x, y = y, x
    return x, y


def _value(v, k):
    if len(v) != 2 or v[0] not in LETTERS or v[1] not in LETTERS:
        return v  # pass ("" or "tt") or anything else: unchanged
    x, y = transform_point(LETTERS.index(v[0]), LETTERS.index(v[1]), k)
    return LETTERS[x] + LETTERS[y]


def transform_sgf(sgf, k):
    """The record with the k-th board symmetry (0..7) applied to every B / W / AB / AW / AE coordinate."""
    if not 0 <= k < 8:
        raise ValueError(f"symmetry index {k} not in 0..7")
    if k == 0:
        return sgf
    return _PROP.sub(lambda m: m.group(1) + _VAL.sub(lambda v: "[" + _value(v.group(1), k) + "]", m.group(2)), sgf)


def sym_of(key, seed):
    """Symmetry index 0..7 of game `key` under simulation seed `seed`."""
    return hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=2).digest()[0] % 8


def game_sym(key, sgf, seed):
    """sym_of(key, seed), except 0 for games of HD >= 2: their records carry no handicap stones and the replay puts them
    on the standard points (not symmetric for every count), so a rotated move list can collide with them. The official
    files hold HD 0 / 1 only."""
    return 0 if hd_class(sgf) >= 2 else sym_of(key, seed)


def transform_games(sgfs, keys, seed):
    """Copy of {key: sgf} with every key in `keys` transformed by game_sym(key, sgf, seed); other keys unchanged."""
    keys = set(keys)
    return {k: (transform_sgf(s, game_sym(k, s, seed)) if k in keys else s) for k, s in sgfs.items()}
