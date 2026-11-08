import numpy as np
import pytest

from aicup.sgf import parse_moves
from aicup_pid import symmetry
from aicup_pid.components import features_all, parse_all
from aicup_pid.dedup import game_hash
from aicup_pid.local import LocalModel
from aicup_pid.opening import tokens
from aicup_pid.orient import orient_tokens
from tests.synth import habit_of, random_game


def test_transform_points_and_passes():
    s = "(;SZ[19]HD[2]AB[dp][pd];W[pp];B[];W[tt];B[aa])"
    assert symmetry.transform_sgf(s, 0) == s
    assert symmetry.transform_sgf(s, 1) == "(;SZ[19]HD[2]AB[pp][dd];W[dp];B[];W[tt];B[sa])"
    assert symmetry.transform_sgf(s, 2) == "(;SZ[19]HD[2]AB[dd][pp];W[pd];B[];W[tt];B[as])"
    assert symmetry.transform_sgf(s, 4) == "(;SZ[19]HD[2]AB[pd][dp];W[pp];B[];W[tt];B[aa])"
    assert symmetry.transform_sgf("(;SZ[19]PW[ab];B[ab])", 1) == "(;SZ[19]PW[ab];B[sb])"  # only move / setup properties
    with pytest.raises(ValueError):
        symmetry.transform_sgf(s, 8)


def test_all_eight_distinct_and_hash_invariant():
    rng = np.random.RandomState(0)
    g = random_game(rng, habit_of(0), 40)
    outs = [symmetry.transform_sgf(g, k) for k in range(8)]
    assert len(set(outs)) == 8
    assert len({game_hash(parse_moves(o)) for o in outs}) == 1
    for k in range(8):  # every symmetry is an involution or has an inverse among the 8
        assert any(symmetry.transform_sgf(outs[k], j) == g for j in range(8))


def test_sym_of_deterministic_and_spread():
    ks = [symmetry.sym_of(f"key{i}", 0) for i in range(4000)]
    assert ks == [symmetry.sym_of(f"key{i}", 0) for i in range(4000)]
    counts = np.bincount(ks, minlength=8)
    assert counts.min() > 400 and counts.max() < 600
    assert ks != [symmetry.sym_of(f"key{i}", 1) for i in range(4000)]
    t = symmetry.transform_games({"a": "(;SZ[19];B[aa])", "b": "(;SZ[19];B[aa])"}, ["a"], 0)
    assert t["b"] == "(;SZ[19];B[aa])" and t["a"] == symmetry.transform_sgf("(;SZ[19];B[aa])", symmetry.sym_of("a", 0))


def test_invariant_components_do_not_move_and_orientation_does():
    rng = np.random.RandomState(1)
    games = {f"g{i}": random_game(rng, habit_of(i % 3), 60) for i in range(6)}
    rot = {k: symmetry.transform_sgf(s, 1 + i % 7) for i, (k, s) in enumerate(games.items())}
    f0, f1 = features_all(games), features_all(rot)
    for k in games:
        assert np.allclose(f0[k], f1[k])  # style features
    m0, m1 = parse_all(games), parse_all(rot)
    lm = LocalModel()
    for k in games:
        assert lm.tokenize(m0[k]) == lm.tokenize(m1[k])  # local-shape tokens
        assert tokens(m0[k], 12) == tokens(m1[k], 12)  # symmetry-canonical opening
    assert any(orient_tokens(m0[k])[1:] != orient_tokens(m1[k])[1:] for k in games)  # raw orientation is not invariant


def test_handicap_games_stay_unrotated():
    g = "(;SZ[19]HD[3];W[pd];B[dd])"
    assert symmetry.game_sym("k", g, 0) == 0 and symmetry.transform_games({"k": g}, ["k"], 0)["k"] == g
    keys = [f"key{i}" for i in range(50)]
    assert any(symmetry.game_sym(k, "(;SZ[19]HD[1];B[pd])", 0) for k in keys)
    assert all(symmetry.game_sym(k, "(;SZ[19]HD[1];B[pd])", 0) == symmetry.sym_of(k, 0) for k in keys)
