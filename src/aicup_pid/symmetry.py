"""Board symmetries of SGF records, for simulating re-released official files whose games are rotated / mirrored.

transform_sgf(sgf, k) applies the k-th of the 8 symmetries of the 19x19 board to every move and setup stone
(B, W, AB, AW, AE values; passes `[]` / `[tt]` are kept): bit 0 mirrors x, bit 1 mirrors y, bit 2 swaps x and y, so
k = 0 is the identity. sym_of(key, seed) picks one symmetry per game from a hash of (seed, key), the same in every
script (CV pools, trunk extraction), and uniform over the 8 (the identity included, as an organiser's random choice would).
"""
import hashlib  # 由棋局鍵決定對稱
import re  # 找出 SGF 中的座標屬性

from aicup_pid.hd import hd_class  # 判斷讓子類別

BOARD = 19  # 棋盤大小
LETTERS = "abcdefghijklmnopqrs"  # SGF 的座標字母
_PROP = re.compile(r"(?<![A-Za-z])(AB|AW|AE|B|W)((?:\[[^\]]*\])+)")  # 帶座標的 B、W、AB、AW、AE 屬性及其所有值
_VAL = re.compile(r"\[([^\]]*)\]")  # 屬性中的單一值


def transform_point(x, y, k):  # 一個點在第 k 種對稱下的座標
    if k & 1:  # 第 0 位元：鏡射 x
        x = BOARD - 1 - x  # 鏡射 x
    if k & 2:  # 第 1 位元：鏡射 y
        y = BOARD - 1 - y  # 鏡射 y
    if k & 4:  # 第 2 位元：交換 x 與 y
        x, y = y, x  # 交換兩軸
    return x, y  # 回傳新座標


def _value(v, k):  # 轉換一個 SGF 座標值
    if len(v) != 2 or v[0] not in LETTERS or v[1] not in LETTERS:  # 不是一般的棋盤座標
        return v  # pass（"" 或 "tt"）或其他內容：保持不變
    x, y = transform_point(LETTERS.index(v[0]), LETTERS.index(v[1]), k)  # 字母轉成數字後套用對稱
    return LETTERS[x] + LETTERS[y]  # 換回 SGF 字母


def transform_sgf(sgf, k):  # 對整份棋譜套用一種對稱
    """The record with the k-th board symmetry (0..7) applied to every B / W / AB / AW / AE coordinate."""
    if not 0 <= k < 8:  # 對稱索引超出範圍
        raise ValueError(f"symmetry index {k} not in 0..7")  # 拒絕
    if k == 0:  # 恆等變換
        return sgf  # 原樣回傳
    return _PROP.sub(lambda m: m.group(1) + _VAL.sub(lambda v: "[" + _value(v.group(1), k) + "]", m.group(2)), sgf)  # 逐一替換屬性中的每個座標


def sym_of(key, seed):  # 每局固定的對稱索引
    """Symmetry index 0..7 of game `key` under simulation seed `seed`."""
    return hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=2).digest()[0] % 8  # 由種子與棋局鍵的雜湊取模 8


def game_sym(key, sgf, seed):  # 考慮讓子後的對稱索引
    """sym_of(key, seed), except 0 for games of HD >= 2: their records carry no handicap stones and the replay puts them
    on the standard points (not symmetric for every count), so a rotated move list can collide with them. The official
    files hold HD 0 / 1 only."""
    return 0 if hd_class(sgf) >= 2 else sym_of(key, seed)  # 讓 2 子以上的棋局不轉換


def transform_games(sgfs, keys, seed):  # 對指定的棋局套用對稱
    """Copy of {key: sgf} with every key in `keys` transformed by game_sym(key, sgf, seed); other keys unchanged."""
    keys = set(keys)  # 轉成集合以便查詢
    return {k: (transform_sgf(s, game_sym(k, s, seed)) if k in keys else s) for k, s in sgfs.items()}  # 只轉換指定的棋局，其餘原樣
