import numpy as np
import pytest
from katago.game.board import Board

from aicup_pid.tactical_replies import (
    opportunity, play_response, counts, profile, evidence, fit_comparator,
    eligible_players, owner_folds, swapped_ranks,
)


def position(black=(), white=(), turn=Board.BLACK):
    board = Board(19)
    for side, stones in ((Board.BLACK, black), (Board.WHITE, white)):
        for x, y in stones:
            board.set_stone(side, board.loc(x, y))
    board.set_pla(turn)
    return board


def test_pre_move_atari_and_escape_or_tenuki():
    b = position(black=[(0, 0)], white=[(1, 0)])
    prev = b.loc(1, 0)
    before = b.pos_zobrist()
    assert opportunity(b, Board.BLACK, prev) == 0
    assert b.pos_zobrist() == before
    distant = b.copy()
    assert play_response(distant, Board.BLACK, distant.loc(10, 10), prev) == (False,) * 5
    reply = play_response(b, Board.BLACK, b.loc(0, 1), prev)
    assert reply == (True, False, False, False, False)
    assert b.num_liberties(b.loc(0, 0)) >= 2


def test_capture_new_atari_connection_and_pass():
    b = position(black=[(0, 1)], white=[(0, 0)])
    assert opportunity(b, Board.BLACK, b.loc(0, 0)) == 1
    assert play_response(b, Board.BLACK, b.loc(1, 0), b.loc(0, 0))[:3] == (True, True, False)
    b = position(white=[(0, 0)])
    assert play_response(b, Board.BLACK, b.loc(1, 0), b.loc(0, 0))[2]
    b = position(black=[(2, 3), (4, 3)], white=[(8, 8)])
    assert play_response(b, Board.BLACK, b.loc(3, 3), b.loc(8, 8))[3]
    b = position(white=[(0, 1)])
    assert play_response(b, Board.BLACK, b.loc(0, 0), b.loc(0, 1))[4]
    b = position(white=[(8, 8)])
    assert play_response(b, Board.BLACK, Board.PASS_LOC, b.loc(8, 8)) == (False,) * 5


def test_same_margins_opposite_contexts_and_shuffle():
    contexts = np.tile([0, 1], 100)
    phases = np.repeat([0, 1], 100)
    acts = np.repeat((contexts == 0)[:, None], 5, axis=1)
    qa, shuffled = counts(contexts, acts, phases, 27)
    qb, _ = counts(contexts, ~acts, phases, 27)
    np.testing.assert_array_equal(shuffled[:, 0], qa[:, 0])
    np.testing.assert_array_equal(shuffled.sum(0), qa.sum(0))
    assert not np.array_equal(qa, shuffled)
    prior = qa + qb
    p = profile(np.stack([qa, qb]), prior)
    marginal, extra = evidence(qa[None], p)
    np.testing.assert_allclose(marginal[0], marginal[1])
    assert np.all(extra[0] > extra[1])
    assert np.isfinite(profile(np.zeros_like(qa)[None], prior)[1]).all()


def test_pool_local_exclusion_and_antisymmetry():
    folds = owner_folds("p", [str(i) for i in range(20)])
    assert np.bincount(folds).tolist() == [5] * 4
    ok = eligible_players(folds, folds[1], folds[2])
    assert not ok[1] and not ok[2]
    x = np.array([[1.0, 2.1], [-1., 1.3], [0.5, 0.9], [1., 0.]])
    w = fit_comparator(x)
    assert w[0] >= 0 and np.isfinite(w).all()
    np.testing.assert_allclose((-x) @ w, -(x @ w))
    np.testing.assert_array_equal(swapped_ranks(np.arange(4), [True] * 4), [1, 0, 2, 3])
