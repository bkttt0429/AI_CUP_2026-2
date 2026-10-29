import numpy as np
import pytest

from aicup_pid import style
from aicup_pid.opening import parse
from tests.synth import habit_of, random_game


def test_board_captures_single_stones_and_groups():
    b = style.Board()
    # White stone at (1,1) surrounded by Black: the fourth Black move captures it
    b.play(2, 1, 1)
    assert b.play(1, 0, 1) == 0 and b.play(1, 2, 1) == 0 and b.play(1, 1, 0) == 0
    assert b.play(1, 1, 2) == 1 and b.s[1][1] == 0
    # a two-stone White group
    b2 = style.Board()
    for x, y in ((5, 5), (6, 5)):
        b2.play(2, x, y)
    for x, y in ((4, 5), (7, 5), (5, 4), (6, 4), (5, 6)):
        assert b2.play(1, x, y) == 0
    assert b2.play(1, 6, 6) == 2 and b2.s[5][5] == 0 and b2.s[5][6] == 0
    assert b2.adjacent_to(5, 5, 1) and not b2.adjacent_to(0, 0, 2)


def test_point_types_lines_and_game_features():
    assert style.point_type(15, 3) == "44" and style.point_type(3, 15) == "44" and style.point_type(16, 2) == "33"
    assert style.point_type(15, 2) == "34" and style.point_type(2, 15) == "34" and style.point_type(9, 9) == "other"
    assert style.line_of(0, 5) == 1 and style.line_of(3, 10) == 4 and style.line_of(9, 9) == 10
    f = style.game_features(parse("(;SZ[19];B[pd];W[dp];B[qp];W[dc];B[oq];W[qq])"))
    assert f.shape == (len(style.FEATURE_NAMES),)
    names = list(style.FEATURE_NAMES)
    assert f[names.index("black_moves")] == 3 and f[names.index("first_44")] == 1 and f[names.index("second_34")] == 1
    assert abs(f[names.index("line_4")] - 1 / 3) < 1e-12 and abs(f[names.index("line_3")] - 2 / 3) < 1e-12
    assert f[names.index("contact")] == 0 and f[names.index("captures_made")] == 0
    assert style.game_features(parse("(;SZ[19];W[dp])")).sum() == pytest.approx(np.log1p(1))  # no Black move: only the length


def test_style_model_separates_line_preferences():
    rng = np.random.RandomState(3)
    players = {f"p{i}": habit_of(i) for i in (0, 16)}  # same corners, line 3 vs line 4
    cand = {p: np.stack([style.game_features(parse(random_game(rng, h, 60))) for _ in range(40)]) for p, h in players.items()}
    m = style.StyleModel(shrink=0.2).fit(cand)
    assert m.players == ["p0", "p16"] and m.cov.shape == (len(style.FEATURE_NAMES),) * 2
    hits = 0
    for p, h in players.items():
        q = np.stack([style.game_features(parse(random_game(rng, h, 60))) for _ in range(5)])
        s = m.log_scores(q)
        assert s.shape == (2,) and np.all(np.isfinite(s)) and np.all(s <= 0)
        hits += m.players[int(np.argmax(s))] == p
    assert hits == 2
    with pytest.raises(ValueError):
        style.StyleModel(shrink=1.5)
