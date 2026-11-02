import numpy as np

from aicup.sgf import parse_moves
from aicup_pid import local
from tests.synth import habit_of, random_game, sgf_of


def test_displacement_classes_and_tokens():
    assert local.displacement((3, 3), (4, 3)) == (1, 0) and local.displacement((3, 3), (4, 4)) == (1, 1) and local.displacement((3, 3), (5, 4)) == (2, 1)
    assert local.displacement((3, 3), (9, 3)) == local.FAR and local.displacement((3, 3), (3, 7)) == local.FAR and local.displacement((3, 3), None) == local.NONE
    assert local.line_of(0, 5) == 1 and local.line_of(3, 10) == 4 and local.line_of(9, 9) == 5 and local.line_of(16, 2) == 3 and local.line_of(18, 2) == 1
    moves = parse_moves(sgf_of([("B", 15, 3), ("W", 3, 15), ("B", 4, 15), ("W", 4, 14), ("B", 5, 14), ("W", 9, 9)]))
    toks = local.tokens(moves, start=0)
    assert len(toks) == 3  # one per Black move
    assert toks[0] == (local.NONE, local.NONE, 4, 0)  # first move: no previous moves, 4th line, phase 0
    assert toks[1] == ((1, 0), local.FAR, 4, 0)  # B[4,15] is adjacent to W[3,15], far from B[15,3]
    assert toks[2] == ((1, 0), (1, 1), 5, 0)  # B[5,14] adjacent to W[4,14], diagonal to B[4,15], 5th line
    assert local.tokens(moves, start=4) == [toks[2]]
    assert local.tokens(parse_moves("(;SZ[19];B[];W[dd];B[pp])"), start=0) == [(local.FAR, local.NONE, 4, 0)]  # a pass has no token; it is not a previous move either


def test_local_model_recovers_planted_line_habit():
    rng = np.random.RandomState(0)
    players = {f"p{i}": [parse_moves(random_game(rng, habit_of(h), n_moves=120)) for _ in range(15)] for i, h in enumerate((0, 1, 16, 17))}  # line 3, 3, 4, 4
    model = local.LocalModel(alpha=2.0, beta=5.0, start=0).fit(players)
    assert model.players == ["p0", "p1", "p2", "p3"] and model.vocabulary >= 10
    queries = {f"p{i}": [parse_moves(random_game(rng, habit_of(h), n_moves=120)) for _ in range(6)] for i, h in enumerate((0, 1, 16, 17))}
    for p, games in queries.items():
        s = model.log_scores(games)
        assert s.shape == (4,) and np.all(np.isfinite(s))
        same_line = {"p0": ["p0", "p1"], "p1": ["p0", "p1"], "p2": ["p2", "p3"], "p3": ["p2", "p3"]}[p]
        assert model.players[int(np.argmax(s))] in same_line
    assert model.state()["alpha"] == 2.0 and model.state()["players"] == model.players
    assert model.log_scores([]).shape == (4,) and np.all(model.log_scores([]) == 0)
