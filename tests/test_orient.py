import numpy as np
import pytest

from aicup.sgf import parse_moves
from aicup_pid import components, orient
from aicup_pid.components import features_all, fit_from_parsed, fit_orient, parse_all, score_components
from tests.synth import LETTERS, habit_of, random_game


def oriented_game(rng, corner, n_moves=40):
    """A random game whose first Black move is the 4-4 point of `corner` (0..3), the rest random on empty points."""
    cx, cy = [(15, 3), (3, 3), (3, 15), (15, 15)][corner]
    occupied, moves = {(cx, cy)}, [("B", cx, cy)]
    while len(moves) < n_moves:
        x, y = int(rng.randint(19)), int(rng.randint(19))
        if (x, y) in occupied:
            continue
        occupied.add((x, y))
        moves.append(("W" if len(moves) % 2 else "B", x, y))
    return "(;SZ[19]" + "".join(f";{c}[{LETTERS[x]}{LETTERS[y]}]" for c, x, y in moves) + ")"


def test_orient_tokens():
    toks = orient.orient_tokens(parse_moves("(;SZ[19];B[pd];W[dp];B[pq])"), depth=12)
    assert len(toks) == 2 and toks[0][0] == "sym" and 0 <= toks[0][1] < 8 and toks[1] == ("first", (15, 3))
    assert orient.orient_tokens(parse_moves("(;SZ[19];B[];W[dp];B[pq])"), depth=12)[1] == ("first", (15, 16))  # a pass is not a first move
    assert orient.orient_tokens(parse_moves("(;SZ[19];W[dp])"), depth=12) == []
    # the two mirror images of one opening get different symmetry indices but the same canonical pattern
    a, b = parse_moves("(;SZ[19];B[pd];W[dp])"), parse_moves("(;SZ[19];B[dp];W[pd])")
    assert orient.orient_tokens(a)[0] != orient.orient_tokens(b)[0]


def test_orient_model_recovers_planted_corner_habit():
    rng = np.random.RandomState(0)
    lib = {f"p{c}": [parse_moves(oriented_game(rng, c)) for _ in range(20)] for c in range(4)}
    model = orient.OrientModel(alpha=2.0, beta=5.0, depth=12).fit(lib)
    assert model.players == ["p0", "p1", "p2", "p3"] and model.vocabulary >= 8 and model.depth == 12
    for c in range(4):
        q = [parse_moves(oriented_game(rng, c)) for _ in range(3)]
        s = model.log_scores(q)
        assert s.shape == (4,) and np.all(np.isfinite(s)) and int(np.argmax(s)) == c
    assert model.log_scores([]).shape == (4,) and np.all(model.log_scores([]) == 0)
    assert model.state()["depth"] == 12 and model.state()["players"] == model.players
    with pytest.raises(ValueError):
        orient.OrientModel(depth=0)
    # the synthetic random games apply a random symmetry per game, so no orientation habit: scores stay near each other
    flat = {f"q{i}": [parse_moves(random_game(rng, habit_of(i), 40)) for _ in range(20)] for i in range(3)}
    m2 = orient.OrientModel().fit(flat)
    assert m2.players == ["q0", "q1", "q2"]


def test_score_components_with_orient():
    rng = np.random.RandomState(1)
    sgfs, by_player = {}, {}
    for c in range(2):
        for j in range(8):
            k = f"c:{c}:{j}"
            sgfs[k] = oriented_game(rng, c)
            by_player.setdefault(f"p{c}", []).append(k)
    sgfs["q:0"], sgfs["q:1"] = oriented_game(rng, 1), oriented_game(rng, 1)
    moves, feats = parse_all(sgfs), features_all(sgfs)
    opening, style = fit_from_parsed(by_player, moves, feats, {"depth": 8})
    model = fit_orient(by_player, moves, {"orient_alpha": 2.0, "orient_beta": 5.0, "orient_depth": 8})
    assert model.players == opening.players == ["p0", "p1"] and model.depth == 8
    S = score_components(("opening", "style", "orient"), opening, style, [["q:0", "q:1"]], moves, feats, orient=model)
    assert set(S) == {"opening", "style", "orient"} and S["orient"].shape == (1, 2) and S["orient"][0, 1] > S["orient"][0, 0]
    assert components.ALL_COMPONENTS == ("opening", "style", "local", "hd", "orient", "opening_raw")
    with pytest.raises(ValueError):
        score_components(("opening", "style", "orient"), opening, style, [["q:0"]], moves, feats)


def test_orient_tokens_with_more_raw_moves():
    m = parse_moves("(;SZ[19];B[pd];W[dp];B[pq];W[dd];B[fq])")
    t1, t2, t3 = (orient.orient_tokens(m, depth=12, n_raw=n) for n in (1, 2, 3))
    assert t1 == [t1[0], ("first", (15, 3))]
    assert t2 == [*t1, ("raw", 1, (15, 16)), ("pair", (15, 3), (15, 16))]
    assert t3 == [*t1, ("raw", 1, (15, 16)), ("raw", 2, (5, 16)), ("pair", (15, 3), (15, 16))]
    assert orient.orient_tokens(parse_moves("(;SZ[19];B[pd];W[dp])"), n_raw=3) == t1  # no second Black move: no raw / pair tokens
    model = orient.OrientModel(n_raw=2).fit({"a": [m], "b": [parse_moves("(;SZ[19];B[dd];W[pp];B[dp])")]})
    assert model.n_raw == 2 and model.state()["n_raw"] == 2 and model.vocabulary == 8  # 2 x (sym, first, raw, pair)
    assert fit_orient({"a": ["k"]}, {"k": m}, {"orient_raw": 3}).n_raw == 3
    with pytest.raises(ValueError):
        orient.OrientModel(n_raw=0)


def test_score_components_with_opening_raw():
    from aicup_pid.components import fit_opening_raw
    rng = np.random.RandomState(2)
    sgfs, by_player = {}, {}
    for c in range(2):
        for j in range(6):
            sgfs[f"c:{c}:{j}"] = oriented_game(rng, c)
            by_player.setdefault(f"p{c}", []).append(f"c:{c}:{j}")
    sgfs["q:0"] = oriented_game(rng, 1)
    moves, feats = parse_all(sgfs), features_all(sgfs)
    opening, style = fit_from_parsed(by_player, moves, feats, {"depth": 8})
    raw = fit_opening_raw(by_player, moves, {"depth": 8})
    assert raw.players == opening.players and raw.symmetry is False and raw.depth == 8
    S = score_components(("opening", "opening_raw"), opening, style, [["q:0"]], moves, feats, opening_raw=raw)
    assert set(S) == {"opening", "opening_raw"} and S["opening_raw"][0, 1] > S["opening_raw"][0, 0]
    assert components.ALL_COMPONENTS == ("opening", "style", "local", "hd", "orient", "opening_raw")
    with pytest.raises(ValueError):
        score_components(("opening_raw",), opening, style, [["q:0"]], moves, feats)
